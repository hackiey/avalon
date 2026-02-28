#!/bin/bash
# ===========================================================================
# NLRL (Natural Language Reinforcement Learning) 训练流水线
# ===========================================================================
# 所有参数由 YAML 配置文件定义。
#
# 用法:
#   bash training/nlrl/pipeline.sh training/nlrl/configs/nlrl_avalon.yaml
#
# 断点续训:
#   RESUME_FROM_ROUND=2 \
#       bash training/nlrl/pipeline.sh training/nlrl/configs/nlrl_avalon.yaml
#
# 完成后继续 RL 训练（修改 base_model 后运行）:
#   bash training/scripts/self_play.sh training/configs/ppo_avalon.yaml
# ===========================================================================

set -eo pipefail

# ===========================================================================
# 加载配置
# ===========================================================================

CONFIG_FILE="${1:?用法: bash pipeline.sh <config.yaml>}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 从 YAML 读取配置值的辅助函数
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

# --- 实验名称 ---
EXPERIMENT_NAME=$(cfg experiment_name "")
if [ -z "${EXPERIMENT_NAME}" ]; then
    EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)
fi

# --- 输出目录 ---
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
STRATEGIES_DIR="${EXPERIMENT_DIR}/strategies"
DATA_DIR="${EXPERIMENT_DIR}/data"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "${CHECKPOINT_DIR}" "${STRATEGIES_DIR}" "${DATA_DIR}" "${LOG_DIR}"
cp "${CONFIG_FILE}" "${EXPERIMENT_DIR}/config.yaml"

# --- Self-Play 参数 ---
BASE_MODEL=$(cfg nlrl.base_model)
ROUNDS=$(cfg nlrl.rounds)
GAMES_PER_ROUND=$(cfg nlrl.games_per_round)
PLAYER_COUNT=$(cfg nlrl.player_count 5)
PARALLEL=$(cfg nlrl.parallel 10)

VLLM_PORT=$(cfg nlrl.vllm.port 8000)
VLLM_GPU_UTIL=$(cfg nlrl.vllm.gpu_util 0.85)
VLLM_TP=$(cfg nlrl.vllm.tp 8)

# --- Teacher LLM 参数 ---
TEACHER_PROVIDER=$(cfg nlrl.teacher.provider openai)
TEACHER_MODEL=$(cfg nlrl.teacher.model gpt-4o)
TEACHER_CONCURRENCY=$(cfg nlrl.teacher.concurrency 10)
ANNOTATE_MAX_PER_ROLE=$(cfg nlrl.teacher.annotate_max_per_role 80)
TEMP_ANNOTATE=$(cfg nlrl.teacher.temperature_annotate 0.3)
TEMP_SYNTH=$(cfg nlrl.teacher.temperature_synthesize 0.7)

# --- Synthesize 参数 ---
SYNTH_CONCURRENCY=$(cfg nlrl.synthesize.concurrency 10)
TRAIN_RATIO=$(cfg nlrl.synthesize.train_ratio 0.9)

# --- SFT 参数 ---
SFT_EPOCHS=$(cfg nlrl.sft.epochs 2)
SFT_LR=$(cfg nlrl.sft.lr 2e-5)
SFT_BATCH=$(cfg nlrl.sft.batch_size 2)
SFT_GRAD_ACCUM=$(cfg nlrl.sft.gradient_accumulation_steps 8)
SFT_MAX_SEQ=$(cfg nlrl.sft.max_seq_length 16384)
SFT_BF16=$(cfg nlrl.sft.bf16 true)

# --- 断点续训 ---
RESUME_FROM_ROUND="${RESUME_FROM_ROUND:-0}"

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
        log "Port ${VLLM_PORT} occupied, killing pid(s): ${old_pid}"
        echo "$old_pid" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    log "Starting vLLM: ${model_path} (as: ${VLLM_MODEL_NAME})"
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

    log "Waiting for vLLM (pid=${VLLM_PID})..."
    for i in $(seq 1 360); do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null | grep -q "${VLLM_MODEL_NAME}"; then
            log "vLLM ready: ${VLLM_MODEL_NAME}"
            return 0
        fi
        sleep 2
    done
    log "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        log "Stopping vLLM (pid=${VLLM_PID})..."
        kill -- -$VLLM_PID 2>/dev/null || kill $VLLM_PID 2>/dev/null || true
        sleep 2
        pkill -P $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        local remaining
        remaining=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            echo "$remaining" | xargs kill -9 2>/dev/null || true
        fi
        VLLM_PID=""
    fi
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
echo "  NLRL Training Pipeline"
echo "============================================================"
echo "  Experiment:     ${EXPERIMENT_NAME}"
echo "  Config:         ${CONFIG_FILE}"
echo "  Output:         ${EXPERIMENT_DIR}/"
echo "  Rounds:         ${ROUNDS}"
echo "  Games/round:    ${GAMES_PER_ROUND}"
echo "  Base model:     ${BASE_MODEL}"
echo "  Teacher:        ${TEACHER_PROVIDER}/${TEACHER_MODEL}"
echo "  SFT epochs:     ${SFT_EPOCHS}, lr=${SFT_LR}"
if [ "${RESUME_FROM_ROUND}" -gt 0 ]; then
echo "  Resume from:    round ${RESUME_FROM_ROUND}"
fi
echo "============================================================"
echo ""

CURRENT_MODEL="${BASE_MODEL}"

# 断点续训：恢复上一轮的模型
if [ "${RESUME_FROM_ROUND}" -gt 1 ]; then
    PREV_ROUND=$((RESUME_FROM_ROUND - 1))
    PREV_CKPT="${CHECKPOINT_DIR}/round_${PREV_ROUND}"
    if [ -d "${PREV_CKPT}" ]; then
        CURRENT_MODEL="${PREV_CKPT}"
        log "Resumed from round ${PREV_ROUND}: ${CURRENT_MODEL}"
    else
        log "WARNING: Checkpoint not found at ${PREV_CKPT}, using base model"
    fi
fi

for CURRENT_ROUND in $(seq 1 "${ROUNDS}"); do
    ROUND_DIR="${DATA_DIR}/round_${CURRENT_ROUND}"
    ROUND_JSONL="${ROUND_DIR}/trajectories.jsonl"
    ROUND_ANNOTATED="${ROUND_DIR}/annotated.jsonl"
    ROUND_STRATEGIES="${STRATEGIES_DIR}/round_${CURRENT_ROUND}"
    PREV_STRATEGIES="${STRATEGIES_DIR}/round_$((CURRENT_ROUND - 1))"
    ROUND_SFT_DIR="${ROUND_DIR}/sft_data"
    ROUND_CKPT="${CHECKPOINT_DIR}/round_${CURRENT_ROUND}"

    mkdir -p "${ROUND_DIR}"

    if [ "${CURRENT_ROUND}" -lt "${RESUME_FROM_ROUND}" ]; then
        log "Skipping round ${CURRENT_ROUND} (resuming from ${RESUME_FROM_ROUND})"
        if [ -d "${ROUND_CKPT}" ]; then
            CURRENT_MODEL="${ROUND_CKPT}"
        fi
        continue
    fi

    echo ""
    log "=========================================="
    log "  NLRL Round ${CURRENT_ROUND}/${ROUNDS}"
    log "  Actor model: ${CURRENT_MODEL}"
    log "=========================================="

    # ------------------------------------------------------------------
    # Step 1: 用当前模型跑游戏，收集轨迹
    # ------------------------------------------------------------------
    log "[Step 1/5] 运行 ${GAMES_PER_ROUND} 局游戏..."

    start_vllm "${CURRENT_MODEL}"

    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export AVAILABLE_MODELS="${VLLM_MODEL_NAME}:vllm"

    python -m training.run_batch run \
        --num-games "${GAMES_PER_ROUND}" \
        --player-count "${PLAYER_COUNT}" \
        --models "${VLLM_MODEL_NAME}:vllm" \
        --parallel "${PARALLEL}" \
        --tag "${EXPERIMENT_NAME}_r${CURRENT_ROUND}" \
        --no-mongo \
        --output "${ROUND_JSONL}" \
        2>&1 | tee -a "${LOG_DIR}/round_${CURRENT_ROUND}.log"

    stop_vllm

    # ------------------------------------------------------------------
    # Step 2: Teacher LLM 标注决策
    # ------------------------------------------------------------------
    log "[Step 2/5] Teacher LLM 标注决策..."

    python -m training.nlrl.annotate \
        --input_jsonl "${ROUND_JSONL}" \
        --output_jsonl "${ROUND_ANNOTATED}" \
        --teacher_provider "${TEACHER_PROVIDER}" \
        --teacher_model "${TEACHER_MODEL}" \
        --concurrency "${TEACHER_CONCURRENCY}" \
        --max_per_role "${ANNOTATE_MAX_PER_ROLE}" \
        --temperature "${TEMP_ANNOTATE}" \
        2>&1 | tee -a "${LOG_DIR}/round_${CURRENT_ROUND}.log"

    # ------------------------------------------------------------------
    # Step 3: 更新角色策略文档
    # ------------------------------------------------------------------
    log "[Step 3/5] 更新策略文档..."

    PREV_STRAT_ARG=""
    if [ "${CURRENT_ROUND}" -gt 1 ] && [ -d "${PREV_STRATEGIES}" ]; then
        PREV_STRAT_ARG="--prev_strategies_dir ${PREV_STRATEGIES}"
    fi

    # shellcheck disable=SC2086
    python -m training.nlrl.strategy update \
        --annotated_jsonl "${ROUND_ANNOTATED}" \
        --strategies_dir "${ROUND_STRATEGIES}" \
        ${PREV_STRAT_ARG} \
        --teacher_provider "${TEACHER_PROVIDER}" \
        --teacher_model "${TEACHER_MODEL}" \
        2>&1 | tee -a "${LOG_DIR}/round_${CURRENT_ROUND}.log"

    # ------------------------------------------------------------------
    # Step 4: Teacher LLM 重新生成决策 → SFT parquet
    # ------------------------------------------------------------------
    log "[Step 4/5] Teacher LLM 合成 SFT 数据..."

    python -m training.nlrl.synthesize \
        --annotated_jsonl "${ROUND_ANNOTATED}" \
        --strategies_dir "${ROUND_STRATEGIES}" \
        --output_dir "${ROUND_SFT_DIR}" \
        --teacher_provider "${TEACHER_PROVIDER}" \
        --teacher_model "${TEACHER_MODEL}" \
        --concurrency "${SYNTH_CONCURRENCY}" \
        --train_ratio "${TRAIN_RATIO}" \
        --temperature "${TEMP_SYNTH}" \
        2>&1 | tee -a "${LOG_DIR}/round_${CURRENT_ROUND}.log"

    # ------------------------------------------------------------------
    # Step 5: SFT 微调
    # ------------------------------------------------------------------
    log "[Step 5/5] SFT 微调..."

    BF16_FLAG=""
    if [ "${SFT_BF16}" = "true" ]; then
        BF16_FLAG="--bf16"
    fi

    python -m training.nlrl.sft_train \
        --train_parquet "${ROUND_SFT_DIR}/train.parquet" \
        --val_parquet "${ROUND_SFT_DIR}/test.parquet" \
        --model_path "${CURRENT_MODEL}" \
        --output_dir "${ROUND_CKPT}" \
        --epochs "${SFT_EPOCHS}" \
        --lr "${SFT_LR}" \
        --batch_size "${SFT_BATCH}" \
        --grad_accum "${SFT_GRAD_ACCUM}" \
        --max_seq_length "${SFT_MAX_SEQ}" \
        ${BF16_FLAG} \
        2>&1 | tee -a "${LOG_DIR}/round_${CURRENT_ROUND}.log"

    CURRENT_MODEL="${ROUND_CKPT}"
    log "Round ${CURRENT_ROUND}/${ROUNDS} complete. Model: ${CURRENT_MODEL}"
done

echo ""
echo "============================================================"
echo "  NLRL Training Complete!"
echo "============================================================"
echo "  Experiment:     ${EXPERIMENT_NAME}"
echo "  Total rounds:   ${ROUNDS}"
echo "  Final model:    ${CURRENT_MODEL}"
echo "  Experiment dir: ${EXPERIMENT_DIR}/"
echo "============================================================"
echo ""
echo "下一步：用 NLRL 训练好的模型继续 RL 训练："
echo "  1. 修改 training/configs/ppo_avalon.yaml 中的 base_model:"
echo "       self_play.base_model: ${CURRENT_MODEL}"
echo "  2. 运行 RL 自博弈训练:"
echo "       bash training/scripts/self_play.sh training/configs/ppo_avalon.yaml"
echo ""
