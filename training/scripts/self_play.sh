#!/bin/bash
# ===========================================================================
# Avalon Self-Play 训练循环（Episode-level GAE + External Critic）
# ===========================================================================
# 所有参数由 YAML 配置文件定义。新实验只需复制 YAML、修改参数、运行。
# 所有产出物保存在 experiments/<experiment_name>/ 下。
#
# 用法:
#   bash training/scripts/self_play.sh training/configs/ppo_avalon.yaml
#   bash training/scripts/self_play.sh training/configs/exp_lr1e5.yaml
#
# 断点续训 (通过环境变量):
#   RESUME_FROM_ROUND=3 RESUME_FROM_STEP=5 \
#       bash training/scripts/self_play.sh training/configs/ppo_avalon.yaml
# ===========================================================================

set -eo pipefail

# ===========================================================================
# 加载配置
# ===========================================================================

CONFIG_FILE="${1:?用法: bash self_play.sh <config.yaml>}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 从 YAML 读取配置值的辅助函数
# 用法: cfg <key.path> [default_value]
cfg() {
    python3 -c "
import yaml, sys, functools, operator
c = yaml.safe_load(open(sys.argv[1]))
keys = sys.argv[2].split('.')
try:
    print(functools.reduce(operator.getitem, keys, c))
except (KeyError, TypeError):
    if len(sys.argv) > 3:
        print(sys.argv[3])
    else:
        print('', end=''); sys.exit(1)
" "${CONFIG_FILE}" "$1" "${2:-}"
}

# --- 实验名称: YAML 中定义，或使用 YAML 文件名 ---
EXPERIMENT_NAME=$(cfg experiment_name "")
if [ -z "${EXPERIMENT_NAME}" ]; then
    EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)
fi

# --- 实验输出目录 ---
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
CRITIC_DIR="${EXPERIMENT_DIR}/critic"
DATA_DIR="${EXPERIMENT_DIR}/data"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "${CHECKPOINT_DIR}" "${CRITIC_DIR}" "${DATA_DIR}" "${LOG_DIR}"

# 将配置文件复制到实验目录，记录本次实验使用的参数
cp "${CONFIG_FILE}" "${EXPERIMENT_DIR}/config.yaml"

# --- Self-Play 循环配置 ---
BASE_MODEL=$(cfg self_play.base_model)
ROUNDS=$(cfg self_play.rounds)
GAMES_PER_ROUND=$(cfg self_play.games_per_round)
PLAYER_COUNT=$(cfg self_play.player_count)
PARALLEL=$(cfg self_play.parallel)

VLLM_PORT=$(cfg self_play.vllm.port)
VLLM_GPU_UTIL=$(cfg self_play.vllm.gpu_util)
VLLM_TP=$(cfg self_play.vllm.tp)

CRITIC_EPOCHS=$(cfg self_play.critic.epochs)
CRITIC_LR=$(cfg self_play.critic.lr)
CRITIC_BATCH_SIZE=$(cfg self_play.critic.batch_size)
CRITIC_GRAD_ACCUM=$(cfg self_play.critic.gradient_accumulation_steps 1)

GAE_GAMMA=$(cfg self_play.gae.gamma)
GAE_LAM=$(cfg self_play.gae.lam)

LEN_PENALTY_START=$(cfg self_play.length_penalty.start_tokens 5000)
LEN_PENALTY_CAP=$(cfg self_play.length_penalty.cap_tokens 8192)
LEN_PENALTY_MAX=$(cfg self_play.length_penalty.max_penalty "-1.0")
LEN_PENALTY_POWER=$(cfg self_play.length_penalty.power "2.0")

# --- 从 VeRL 配置中读取 step 6/8 需要的值 ---
TRAIN_EPOCHS=$(cfg trainer.total_epochs)
PROJECT_NAME=$(cfg trainer.project_name)

# --- 断点续训 (仅通过环境变量控制) ---
RESUME_FROM_ROUND="${RESUME_FROM_ROUND:-0}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-0}"

# ===========================================================================
# 辅助函数
# ===========================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_vllm() {
    local model_path=$1
    VLLM_MODEL_NAME=$(basename "${model_path}")

    local old_pid
    old_pid=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
    if [ -n "$old_pid" ]; then
        log "Port ${VLLM_PORT} is occupied by pid(s): ${old_pid}, killing..."
        echo "$old_pid" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    log "Starting vLLM server with model: ${model_path} (served as: ${VLLM_MODEL_NAME})"
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
        kill -- -$VLLM_PID 2>/dev/null || kill $VLLM_PID 2>/dev/null || true
        sleep 2
        pkill -P $VLLM_PID 2>/dev/null || true
        sleep 1
        if kill -0 $VLLM_PID 2>/dev/null; then
            log "vLLM still alive, sending SIGKILL..."
            kill -9 -- -$VLLM_PID 2>/dev/null || kill -9 $VLLM_PID 2>/dev/null || true
        fi
        wait $VLLM_PID 2>/dev/null || true

        local remaining
        remaining=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            log "Killing remaining processes on port ${VLLM_PORT}: ${remaining}"
            echo "$remaining" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
        VLLM_PID=""

        log "Waiting for GPU memory to be released..."
        for i in $(seq 1 30); do
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

kill_gpu_processes() {
    log "Killing remaining GPU processes..."
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
    if [ -z "$gpu_pids" ]; then
        log "No GPU processes found"
        return 0
    fi
    log "Found GPU processes: $(echo $gpu_pids | tr '\n' ' ')"
    for pid in $gpu_pids; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    local remaining
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
    if [ -n "$remaining" ]; then
        log "WARNING: Some GPU processes still alive: ${remaining}"
    else
        log "All GPU processes cleaned up"
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
echo "  Experiment:          ${EXPERIMENT_NAME}"
echo "  Config:              ${CONFIG_FILE}"
echo "  Output:              ${EXPERIMENT_DIR}/"
echo "  Rounds:              ${ROUNDS}"
echo "  Games per round:     ${GAMES_PER_ROUND}"
echo "  Base model:          ${BASE_MODEL}"
echo "  Train epochs/round:  ${TRAIN_EPOCHS}"
echo "  Critic epochs/round: ${CRITIC_EPOCHS}"
echo "  GAE gamma/lambda:    ${GAE_GAMMA} / ${GAE_LAM}"
echo "  Length penalty:      start=${LEN_PENALTY_START} cap=${LEN_PENALTY_CAP} max=${LEN_PENALTY_MAX}"
if [ "${RESUME_FROM_ROUND}" -gt 0 ]; then
echo "  Resume from:         round ${RESUME_FROM_ROUND}, step ${RESUME_FROM_STEP}"
fi
echo "============================================================"
echo ""

CURRENT_MODEL="${BASE_MODEL}"
CURRENT_CRITIC="${BASE_MODEL}"

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
    ROUND_TAG="${EXPERIMENT_NAME}_r${CURRENT_ROUND}"
    ROUND_DATA_DIR="${DATA_DIR}/round_${CURRENT_ROUND}"
    ROUND_JSONL="${ROUND_DATA_DIR}/trajectories.jsonl"
    ROUND_VALUES_JSON="${ROUND_DATA_DIR}/values.json"
    ROUND_ADV_JSON="${ROUND_DATA_DIR}/advantages.json"
    ROUND_PARQUET_DIR="${ROUND_DATA_DIR}/processed"
    ROUND_CRITIC_DIR="${CRITIC_DIR}/round_${CURRENT_ROUND}"
    ROUND_EXPERIMENT="${EXPERIMENT_NAME}-r${CURRENT_ROUND}"
    ROUND_STATS_FILE="${EXPERIMENT_DIR}/round_stats.jsonl"

    mkdir -p "${ROUND_DATA_DIR}"

    if [ "${CURRENT_ROUND}" -lt "${RESUME_FROM_ROUND}" ]; then
        log "Skipping round ${CURRENT_ROUND} (resuming from round ${RESUME_FROM_ROUND})"
        if [ -d "${CHECKPOINT_DIR}/round_${CURRENT_ROUND}" ]; then
            CURRENT_MODEL="${CHECKPOINT_DIR}/round_${CURRENT_ROUND}"
        fi
        if [ -d "${ROUND_CRITIC_DIR}" ]; then
            CURRENT_CRITIC="${ROUND_CRITIC_DIR}"
        fi
        continue
    fi

    should_skip_step() {
        local step=$1
        if [ "${CURRENT_ROUND}" -eq "${RESUME_FROM_ROUND}" ] && [ "$step" -lt "${RESUME_FROM_STEP}" ]; then
            return 0
        fi
        return 1
    }

    echo ""
    log "=========================================="
    log "  Round ${CURRENT_ROUND}/${ROUNDS}"
    log "  Actor model:  ${CURRENT_MODEL}"
    log "  Critic model: ${CURRENT_CRITIC}"
    log "=========================================="

    # ------------------------------------------------------------------
    # Step 1+2: 用当前模型跑游戏并直接导出轨迹
    # ------------------------------------------------------------------
    if should_skip_step 2; then
        log "[Step 1-2/8] SKIPPED (resume mode)"
    else
        log "[Step 1-2/8] 启动 vLLM 并运行 ${GAMES_PER_ROUND} 局游戏..."

        start_vllm "${CURRENT_MODEL}"

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
    # Step 2.5: 游戏统计（胜率等）
    # ------------------------------------------------------------------
    if [ -f "${ROUND_JSONL}" ]; then
        log "[Step 2.5/8] 计算游戏统计..."
        python -m training.stats.game_stats \
            --input_jsonl "${ROUND_JSONL}" \
            --round "${CURRENT_ROUND}" \
            --summary_file "${ROUND_STATS_FILE}" \
            --wandb_project "${PROJECT_NAME}" \
            --experiment_name "${EXPERIMENT_NAME}" \
            2>&1 | tee -a "${LOG_DIR}/${ROUND_EXPERIMENT}.log"
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
            --model_path "${CURRENT_MODEL}" \
            --train_ratio 0.9
    fi

    # ------------------------------------------------------------------
    # Step 6: Verl 训练 actor（使用 precomputed advantage）
    # ------------------------------------------------------------------
    if should_skip_step 6; then
        log "[Step 6/8] SKIPPED (resume mode)"
    else
        log "[Step 6/8] Actor 训练 (precomputed advantage, ${TRAIN_EPOCHS} epochs)..."

        # run_ppo.py 自动加载 YAML 中的 VeRL 参数
        # 这里只传每轮变化的运行时路径
        # 注意: PPO 训练可能在 cleanup 阶段 OOM 崩溃（训练本身已完成），
        # 因此捕获退出码并检查 checkpoint 是否已保存来判断是否可以继续。
        local ppo_exit_code=0
        export LEN_PENALTY_START="${LEN_PENALTY_START}"
        export LEN_PENALTY_CAP="${LEN_PENALTY_CAP}"
        export LEN_PENALTY_MAX="${LEN_PENALTY_MAX}"
        export LEN_PENALTY_POWER="${LEN_PENALTY_POWER}"
        PYTHONUNBUFFERED=1 python training/scripts/run_ppo.py \
            --avalon-config "${CONFIG_FILE}" \
            data.train_files="${ROUND_PARQUET_DIR}/train.parquet" \
            data.val_files="${ROUND_PARQUET_DIR}/test.parquet" \
            actor_rollout_ref.model.path="${CURRENT_MODEL}" \
            trainer.experiment_name="${ROUND_EXPERIMENT}" \
            2>&1 | tee "${LOG_DIR}/${ROUND_EXPERIMENT}.log" || ppo_exit_code=$?

        if [ "$ppo_exit_code" -ne 0 ]; then
            log "WARNING: PPO training exited with code ${ppo_exit_code}"
            # 检查最终 checkpoint 是否已保存
            local expected_ckpt_prefix="checkpoints/${PROJECT_NAME}/${ROUND_EXPERIMENT}"
            local latest_ckpt_dir
            latest_ckpt_dir=$(ls -d "${expected_ckpt_prefix}"/global_step_*/actor 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
            if [ -n "$latest_ckpt_dir" ]; then
                log "Checkpoint found at ${latest_ckpt_dir}, treating as successful (likely cleanup OOM)"
            else
                log "ERROR: No checkpoint found under ${expected_ckpt_prefix}/, cannot continue"
                exit 1
            fi
        fi
    fi

    # Step 6 后清理残留 GPU 进程（Ray workers 可能未正常退出）
    kill_gpu_processes

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
            --gradient_accumulation_steps "${CRITIC_GRAD_ACCUM}" \
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

        # 自动查找最新的 global_step checkpoint（VeRL 的 step 编号由实际梯度步数决定）
        CKPT_PREFIX="checkpoints/${PROJECT_NAME}/${ROUND_EXPERIMENT}"
        LATEST_CKPT=$(ls -d "${CKPT_PREFIX}"/global_step_*/actor 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
        MERGED_MODEL="${CHECKPOINT_DIR}/round_${CURRENT_ROUND}"

        if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
            log "Found latest checkpoint: ${LATEST_CKPT}"
            merge_checkpoint "${LATEST_CKPT}" "${MERGED_MODEL}"
            CURRENT_MODEL="${MERGED_MODEL}"
            log "Round ${CURRENT_ROUND} done. New actor model: ${CURRENT_MODEL}"
        else
            log "WARNING: No checkpoint found under ${CKPT_PREFIX}/, reusing previous model"
        fi
    fi

    log "Round ${CURRENT_ROUND}/${ROUNDS} complete"
done

echo ""
echo "============================================================"
echo "  Self-Play Training Complete!"
echo "============================================================"
echo "  Experiment:      ${EXPERIMENT_NAME}"
echo "  Config:          ${CONFIG_FILE}"
echo "  Total rounds:    ${ROUNDS}"
echo "  Final actor:     ${CURRENT_MODEL}"
echo "  Final critic:    ${CURRENT_CRITIC}"
echo "  Experiment dir:  ${EXPERIMENT_DIR}/"
echo "============================================================"
echo ""

# 打印跨轮胜率趋势
ROUND_STATS_FILE="${EXPERIMENT_DIR}/round_stats.jsonl"
if [ -f "${ROUND_STATS_FILE}" ]; then
    echo "  Round Win Rate Trend:"
    echo "  ─────────────────────────────────────────"
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        s = json.loads(line.strip())
        r = s['round']
        g = s['good_win_rate'] * 100
        e = s['evil_win_rate'] * 100
        n = s['total_games']
        bar_g = '█' * int(g / 5)
        print(f'  R{r:>3d}  Good {g:5.1f}% {bar_g:<20s}  Evil {e:5.1f}%  ({n} games)')
" "${ROUND_STATS_FILE}"
    echo "  ─────────────────────────────────────────"
    echo "  Full stats: ${ROUND_STATS_FILE}"
fi
echo ""

echo "评估最终模型:"
echo "  python -m training.eval.evaluate \\"
echo "      --model_path ${CURRENT_MODEL} \\"
echo "      --num_games 50 --auto_serve"
