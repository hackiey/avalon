#!/bin/bash
# ===========================================================================
# Avalon Self-Play 训练循环（Episode-level GAE + External Critic）
# ===========================================================================
# 自动化的自博弈训练流程（8 步）：
#   1. 用当前模型跑游戏 → 产生新轨迹
#   2. 导出 JSONL 轨迹
#   3. Critic 推理 V(s)                   ← 新增
#   4. GAE 计算 advantage                  ← 新增
#   5. 数据预处理（注入 advantage）→ parquet ← 修改
#   6. Verl 训练 actor（precomputed adv）   ← 修改
#   7. 训练 critic                          ← 新增
#   8. 合并 actor checkpoint
#
# Critic 在 Verl 外部独立训练，提供 episode 级别（跨决策点）的 GAE advantage，
# 实现 credit assignment。整体训练仍是 on-policy 的自博弈循环。
#
# 用法:
#   bash training/scripts/self_play.sh
#
# 可通过环境变量配置:
#   ROUNDS=10 GAMES_PER_ROUND=50 bash training/scripts/self_play.sh
# ===========================================================================

set -e

# ===========================================================================
# 配置
# ===========================================================================

# --- 断点续训 ---
# 从第几轮的第几步恢复（默认从头开始）
# 步骤编号: 1-2=采样, 3=Critic推理, 4=GAE, 5=预处理, 6=Actor训练, 7=Critic训练, 8=合并
# 例: RESUME_FROM_ROUND=1 RESUME_FROM_STEP=3  → 从第1轮的Critic推理开始
RESUME_FROM_ROUND="${RESUME_FROM_ROUND:-0}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-0}"

# --- 自博弈轮次 ---
ROUNDS="${ROUNDS:-5}"

# 每轮游戏数量
GAMES_PER_ROUND="${GAMES_PER_ROUND:-50}"

# 初始模型（第一轮使用）
BASE_MODEL="${BASE_MODEL:-/mnt/iem-nas/home/share/llm_share/models/Qwen3-14B}"

# 每轮 Verl 训练步数（较小，因为会频繁刷新数据）
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"

# vLLM 服务配置（用于跑游戏时的推理）
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_TP="${VLLM_TP:-8}"              # tensor-parallel size（14B 模型 TP=2 即可）

# 游戏配置
PLAYER_COUNT="${PLAYER_COUNT:-5}"
PARALLEL="${PARALLEL:-10}"            # 并发游戏数，8卡可以大幅提高

# Verl 训练配置
N_GPUS="${N_GPUS:-8}"                 # 使用全部 GPU 做数据并行训练
BATCH_SIZE="${BATCH_SIZE:-128}"
REWARD_FN_PATH="${REWARD_FN_PATH:-training/reward/avalon_reward.py}"

# Critic / GAE 配置
CRITIC_EPOCHS="${CRITIC_EPOCHS:-3}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
CRITIC_BATCH_SIZE="${CRITIC_BATCH_SIZE:-16}"  # 多卡可增大 batch
GAE_GAMMA="${GAE_GAMMA:-0.99}"
GAE_LAM="${GAE_LAM:-0.95}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-training/self_play}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
CRITIC_DIR="${OUTPUT_DIR}/critic"
DATA_DIR="${OUTPUT_DIR}/data"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${CHECKPOINT_DIR}" "${CRITIC_DIR}" "${DATA_DIR}" "${LOG_DIR}"

# ===========================================================================
# 辅助函数
# ===========================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_vllm() {
    local model_path=$1
    VLLM_MODEL_NAME=$(basename "${model_path}")

    # 先杀掉占用端口的旧进程
    local old_pid
    old_pid=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
    if [ -n "$old_pid" ]; then
        log "Port ${VLLM_PORT} is occupied by pid(s): ${old_pid}, killing..."
        echo "$old_pid" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    log "Starting vLLM server with model: ${model_path} (served as: ${VLLM_MODEL_NAME})"
    # 用 process group 启动，确保 $! 拿到的是 vLLM 进程 PID（而非 tee）
    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --served-model-name "${VLLM_MODEL_NAME}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        --reasoning-parser deepseek_r1 \
        > >(tee "${LOG_DIR}/vllm_round_${CURRENT_ROUND}.log") 2>&1 &
    VLLM_PID=$!

    # 等待服务就绪（检查 /v1/models 确认模型名正确）
    log "Waiting for vLLM server (pid=${VLLM_PID})..."
    for i in $(seq 1 360); do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null | grep -q "${VLLM_MODEL_NAME}"; then
            log "vLLM server ready, serving model: ${VLLM_MODEL_NAME}"
            return 0
        fi
        sleep 2
    done

    log "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        log "Stopping vLLM server (pid=${VLLM_PID})..."

        # 1) 先尝试 SIGTERM 优雅关闭整个进程组
        kill -- -$VLLM_PID 2>/dev/null || kill $VLLM_PID 2>/dev/null || true
        sleep 2

        # 2) 杀掉所有子进程（Ray worker / multiprocessing）
        pkill -P $VLLM_PID 2>/dev/null || true
        sleep 1

        # 3) 如果还活着，SIGKILL 强杀
        if kill -0 $VLLM_PID 2>/dev/null; then
            log "vLLM still alive, sending SIGKILL..."
            kill -9 -- -$VLLM_PID 2>/dev/null || kill -9 $VLLM_PID 2>/dev/null || true
        fi
        wait $VLLM_PID 2>/dev/null || true

        # 4) 清理可能残留的 vLLM/Ray 相关进程占用该端口
        local remaining
        remaining=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            log "Killing remaining processes on port ${VLLM_PORT}: ${remaining}"
            echo "$remaining" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi

        VLLM_PID=""

        # 5) 等待 GPU 显存真正释放（最多等 60 秒）
        log "Waiting for GPU memory to be released..."
        for i in $(seq 1 30); do
            # 检查是否还有 python 进程占用大量 GPU 显存
            local gpu_procs
            gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
            if [ -z "$gpu_procs" ]; then
                log "GPU memory fully released"
                break
            fi
            if [ "$i" -eq 30 ]; then
                log "WARNING: GPU memory not fully released after 60s, proceeding anyway"
                log "Remaining GPU processes: ${gpu_procs}"
            fi
            sleep 2
        done
    fi
}

merge_checkpoint() {
    local ckpt_dir=$1
    local output_dir=$2
    log "Merging checkpoint: ${ckpt_dir} -> ${output_dir}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ckpt_dir}" \
        --target_dir "${output_dir}"
}

# 清理函数
cleanup() {
    log "Cleaning up..."
    stop_vllm
}
trap cleanup EXIT

# ===========================================================================
# 主循环
# ===========================================================================

echo ""
echo "============================================================"
echo "  Avalon Self-Play Training (Episode-level GAE)"
echo "============================================================"
echo "  Rounds:              ${ROUNDS}"
echo "  Games per round:     ${GAMES_PER_ROUND}"
echo "  Base model:          ${BASE_MODEL}"
echo "  Batch size:          ${BATCH_SIZE}"
echo "  Train epochs/round:  ${TRAIN_EPOCHS}"
echo "  Critic epochs/round: ${CRITIC_EPOCHS}"
echo "  GAE gamma:           ${GAE_GAMMA}"
echo "  GAE lambda:          ${GAE_LAM}"
echo "  Output:              ${OUTPUT_DIR}"
if [ "${RESUME_FROM_ROUND}" -gt 0 ]; then
echo "  Resume from:         round ${RESUME_FROM_ROUND}, step ${RESUME_FROM_STEP}"
fi
echo "============================================================"
echo ""

# 当前使用的模型路径
CURRENT_MODEL="${BASE_MODEL}"

# Critic 模型路径（首轮从 base model 初始化）
CURRENT_CRITIC="${BASE_MODEL}"

# 如果是断点续训，恢复之前轮次的模型路径
if [ "${RESUME_FROM_ROUND}" -gt 1 ]; then
    PREV_ROUND=$((RESUME_FROM_ROUND - 1))
    PREV_MERGED="${CHECKPOINT_DIR}/round_${PREV_ROUND}"
    PREV_CRITIC="${CRITIC_DIR}/round_${PREV_ROUND}"
    if [ -d "${PREV_MERGED}" ]; then
        CURRENT_MODEL="${PREV_MERGED}"
        log "Resumed actor model from round ${PREV_ROUND}: ${CURRENT_MODEL}"
    else
        log "WARNING: Previous actor checkpoint not found at ${PREV_MERGED}, using base model"
    fi
    if [ -d "${PREV_CRITIC}" ]; then
        CURRENT_CRITIC="${PREV_CRITIC}"
        log "Resumed critic model from round ${PREV_ROUND}: ${CURRENT_CRITIC}"
    else
        log "WARNING: Previous critic checkpoint not found at ${PREV_CRITIC}, using base model"
    fi
fi

for CURRENT_ROUND in $(seq 1 "${ROUNDS}"); do
    ROUND_TAG="self_play_r${CURRENT_ROUND}"
    ROUND_DATA_DIR="${DATA_DIR}/round_${CURRENT_ROUND}"
    ROUND_JSONL="${ROUND_DATA_DIR}/trajectories.jsonl"
    ROUND_VALUES_JSON="${ROUND_DATA_DIR}/values.json"
    ROUND_ADV_JSON="${ROUND_DATA_DIR}/advantages.json"
    ROUND_PARQUET_DIR="${ROUND_DATA_DIR}/processed"
    ROUND_CRITIC_DIR="${CRITIC_DIR}/round_${CURRENT_ROUND}"

    mkdir -p "${ROUND_DATA_DIR}"

    # 判断当前轮次是否需要跳过（断点续训逻辑）
    if [ "${CURRENT_ROUND}" -lt "${RESUME_FROM_ROUND}" ]; then
        log "Skipping round ${CURRENT_ROUND} (resuming from round ${RESUME_FROM_ROUND})"
        # 恢复该轮的模型路径（供后续轮次使用）
        if [ -d "${CHECKPOINT_DIR}/round_${CURRENT_ROUND}" ]; then
            CURRENT_MODEL="${CHECKPOINT_DIR}/round_${CURRENT_ROUND}"
        fi
        if [ -d "${ROUND_CRITIC_DIR}" ]; then
            CURRENT_CRITIC="${ROUND_CRITIC_DIR}"
        fi
        continue
    fi

    # 判断当前步骤是否应该跳过
    # 仅在 resume 目标轮次才按 step 跳过，之后的轮次全部执行
    should_skip_step() {
        local step=$1
        if [ "${CURRENT_ROUND}" -eq "${RESUME_FROM_ROUND}" ] && [ "$step" -lt "${RESUME_FROM_STEP}" ]; then
            return 0  # true, should skip
        fi
        return 1  # false, should run
    }

    echo ""
    log "=========================================="
    log "  Round ${CURRENT_ROUND}/${ROUNDS}"
    log "  Actor model:  ${CURRENT_MODEL}"
    log "  Critic model: ${CURRENT_CRITIC}"
    log "=========================================="

    # ------------------------------------------------------------------
    # Step 1+2: 用当前模型跑游戏并直接导出轨迹（跳过 MongoDB）
    # ------------------------------------------------------------------
    if should_skip_step 2; then
        log "[Step 1-2/8] SKIPPED (resume mode, trajectories already exist)"
    else
        log "[Step 1-2/8] 启动 vLLM 并运行 ${GAMES_PER_ROUND} 局游戏 (--no-mongo)..."

        start_vllm "${CURRENT_MODEL}"

        # 用 vLLM 作为推理后端跑游戏，--no-mongo 跳过 MongoDB，直接导出 JSONL
        # VLLM_MODEL_NAME 已在 start_vllm() 中设置为 basename

        # 覆盖 .env 中的 VLLM_BASE_URL，确保连接到本脚本启动的 vLLM 服务
        export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
        export AVAILABLE_MODELS="${VLLM_MODEL_NAME}:vllm"

        python -m training.run_batch run \
            --num-games "${GAMES_PER_ROUND}" \
            --player-count "${PLAYER_COUNT}" \
            --models "${VLLM_MODEL_NAME}:vllm" \
            --parallel "${PARALLEL}" \
            --tag "${ROUND_TAG}" \
            --no-mongo \
            --output "${ROUND_JSONL}"

        stop_vllm
    fi

    # ------------------------------------------------------------------
    # Step 3: Critic 推理 V(s)
    # ------------------------------------------------------------------
    if should_skip_step 3; then
        log "[Step 3/8] SKIPPED (resume mode)"
    else
        log "[Step 3/8] Critic 推理 V(s)..."

        python -m training.critic.infer \
            --model_path "${CURRENT_CRITIC}" \
            --input_jsonl "${ROUND_JSONL}" \
            --output_json "${ROUND_VALUES_JSON}" \
            --batch_size "${CRITIC_BATCH_SIZE}" \
            --bf16
    fi

    # ------------------------------------------------------------------
    # Step 4: GAE 计算 advantage
    # ------------------------------------------------------------------
    if should_skip_step 4; then
        log "[Step 4/8] SKIPPED (resume mode)"
    else
        log "[Step 4/8] GAE 计算 advantage (gamma=${GAE_GAMMA}, lambda=${GAE_LAM})..."

        python -m training.advantage.compute \
            --input_jsonl "${ROUND_JSONL}" \
            --values_json "${ROUND_VALUES_JSON}" \
            --output_json "${ROUND_ADV_JSON}" \
            --gamma "${GAE_GAMMA}" \
            --lam "${GAE_LAM}"
    fi

    # ------------------------------------------------------------------
    # Step 5: 数据预处理（注入 advantage）→ parquet
    # ------------------------------------------------------------------
    if should_skip_step 5; then
        log "[Step 5/8] SKIPPED (resume mode)"
    else
        log "[Step 5/8] 数据预处理 (注入预计算 advantage)..."

        python -m training.data.preprocess \
            --input_jsonl "${ROUND_JSONL}" \
            --output_dir "${ROUND_PARQUET_DIR}" \
            --advantages_file "${ROUND_ADV_JSON}" \
            --train_ratio 0.9
    fi

    # ------------------------------------------------------------------
    # Step 6: Verl 训练 actor（使用 precomputed advantage）
    # ------------------------------------------------------------------
    if should_skip_step 6; then
        log "[Step 6/8] SKIPPED (resume mode)"
    else
        log "[Step 6/8] Actor 训练 (precomputed advantage, ${TRAIN_EPOCHS} epochs)..."

        EXPERIMENT_NAME="self-play-gae-r${CURRENT_ROUND}"

        # 通过 wrapper 脚本启动，确保自定义 advantage estimator 在同一进程中注册
        PYTHONUNBUFFERED=1 python training/scripts/run_ppo.py \
            data.train_files="${ROUND_PARQUET_DIR}/train.parquet" \
            data.val_files="${ROUND_PARQUET_DIR}/test.parquet" \
            data.train_batch_size="${BATCH_SIZE}" \
            data.max_prompt_length=24000 \
            data.max_response_length=24000 \
            algorithm.adv_estimator=precomputed \
            algorithm.kl_ctrl.kl_coef=0.001 \
            actor_rollout_ref.model.path="${CURRENT_MODEL}" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.actor.clip_ratio=0.2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.temperature=0.7 \
            actor_rollout_ref.rollout.top_p=0.9 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
            custom_reward_function.path="${REWARD_FN_PATH}" \
            custom_reward_function.name=compute_score \
            trainer.project_name=avalon-self-play \
            trainer.experiment_name="${EXPERIMENT_NAME}" \
            'trainer.logger=["console","wandb"]' \
            trainer.n_gpus_per_node="${N_GPUS}" \
            trainer.nnodes=1 \
            trainer.save_freq="${TRAIN_EPOCHS}" \
            trainer.test_freq=10 \
            trainer.total_epochs="${TRAIN_EPOCHS}" \
            trainer.val_before_train=false \
            2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
    fi

    # ------------------------------------------------------------------
    # Step 7: 训练 Critic
    # ------------------------------------------------------------------
    if should_skip_step 7; then
        log "[Step 7/8] SKIPPED (resume mode)"
    else
        log "[Step 7/8] Critic 训练 (${CRITIC_EPOCHS} epochs, lr=${CRITIC_LR})..."

        python -m training.critic.train \
            --model_path "${CURRENT_CRITIC}" \
            --data_file "${ROUND_PARQUET_DIR}/train.parquet" \
            --output_dir "${ROUND_CRITIC_DIR}" \
            --epochs "${CRITIC_EPOCHS}" \
            --lr "${CRITIC_LR}" \
            --batch_size "${CRITIC_BATCH_SIZE}" \
            --bf16
    fi

    CURRENT_CRITIC="${ROUND_CRITIC_DIR}"
    log "Critic updated: ${CURRENT_CRITIC}"

    # ------------------------------------------------------------------
    # Step 8: 合并 actor checkpoint 作为下一轮模型
    # ------------------------------------------------------------------
    if should_skip_step 8; then
        log "[Step 8/8] SKIPPED (resume mode)"
    else
        log "[Step 8/8] 合并 actor checkpoint..."

        EXPERIMENT_NAME="self-play-gae-r${CURRENT_ROUND}"
        # Verl 默认保存路径: checkpoints/{project}/{experiment}/global_step_{N}/actor
        LATEST_CKPT="checkpoints/avalon-self-play/${EXPERIMENT_NAME}/global_step_${TRAIN_EPOCHS}/actor"
        MERGED_MODEL="${CHECKPOINT_DIR}/round_${CURRENT_ROUND}"

        if [ -d "${LATEST_CKPT}" ]; then
            merge_checkpoint "${LATEST_CKPT}" "${MERGED_MODEL}"
            CURRENT_MODEL="${MERGED_MODEL}"
            log "Round ${CURRENT_ROUND} done. New actor model: ${CURRENT_MODEL}"
        else
            log "WARNING: Checkpoint not found at ${LATEST_CKPT}, reusing previous model"
        fi
    fi

    log "Round ${CURRENT_ROUND}/${ROUNDS} complete"
done

echo ""
echo "============================================================"
echo "  Self-Play Training Complete!"
echo "============================================================"
echo "  Total rounds:    ${ROUNDS}"
echo "  Final actor:     ${CURRENT_MODEL}"
echo "  Final critic:    ${CURRENT_CRITIC}"
echo "  All data:        ${DATA_DIR}/"
echo "  All logs:        ${LOG_DIR}/"
echo "  Checkpoints:     ${CHECKPOINT_DIR}/"
echo "  Critic models:   ${CRITIC_DIR}/"
echo "============================================================"
echo ""
echo "评估最终模型:"
echo "  python -m training.eval.evaluate \\"
echo "      --model_path ${CURRENT_MODEL} \\"
echo "      --num_games 50 --auto_serve"
