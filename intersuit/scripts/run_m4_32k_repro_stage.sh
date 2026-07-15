#!/usr/bin/env bash
set -euo pipefail

# 按“先 canary、再正式训练”的顺序启动 16k/32k 复现阶段。
# 示例：
#   bash scripts/run_m4_32k_repro_stage.sh 16k-canary150
#   bash scripts/run_m4_32k_repro_stage.sh 32k-canary20
#   bash scripts/run_m4_32k_repro_stage.sh 32k-canary150
#   bash scripts/run_m4_32k_repro_stage.sh 32k-full

cd "$(dirname "$0")/.."

STAGE="${1:-32k-canary20}"
LOG_DIR="${LOG_DIR:-train_logs}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export M4_CHUNKED_LM_LOSS="${M4_CHUNKED_LM_LOSS:-1}"
export M4_CHUNKED_LM_LOSS_TOKENS="${M4_CHUNKED_LM_LOSS_TOKENS:-512}"
export M4_CHUNKED_LM_LOSS_CHECKPOINT="${M4_CHUNKED_LM_LOSS_CHECKPOINT:-0}"
export M4_FREEZE_LM_HEAD="${M4_FREEZE_LM_HEAD:-0}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_lowmem.json}"
export M4_ALLOW_ZERO3_LOWMEM_RESUME="${M4_ALLOW_ZERO3_LOWMEM_RESUME:-1}"
export SAVE_STEPS="${SAVE_STEPS:-25}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$STAGE" in
  16k-canary150)
    export MODEL_MAX_LENGTH=16384
    export MAX_STEPS=150
    export MID_RUN_NAME="${MID_RUN_NAME:-M4-LongVA-7B-Qwen2-repro-16k-canary150}"
    ;;
  32k-canary20)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS=20
    export MID_RUN_NAME="${MID_RUN_NAME:-M4-LongVA-7B-Qwen2-repro-32k-canary20}"
    ;;
  32k-canary150)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS=150
    export MID_RUN_NAME="${MID_RUN_NAME:-M4-LongVA-7B-Qwen2-repro-32k-canary150}"
    ;;
  32k-full)
    export MODEL_MAX_LENGTH=32000
    unset MAX_STEPS
    export MID_RUN_NAME="${MID_RUN_NAME:-M4-LongVA-7B-Qwen2-repro-32k-full}"
    ;;
  *)
    echo "未知阶段：$STAGE" >&2
    echo "可用阶段：16k-canary150, 32k-canary20, 32k-canary150, 32k-full" >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.log"
CONFIG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.env"

{
  echo "STAGE=$STAGE"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "NUM_GPUS=$NUM_GPUS"
  echo "MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH"
  echo "MAX_STEPS=${MAX_STEPS:-}"
  echo "MID_RUN_NAME=$MID_RUN_NAME"
  echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
  echo "M4_CHUNKED_LM_LOSS=$M4_CHUNKED_LM_LOSS"
  echo "M4_CHUNKED_LM_LOSS_TOKENS=$M4_CHUNKED_LM_LOSS_TOKENS"
  echo "M4_CHUNKED_LM_LOSS_CHECKPOINT=$M4_CHUNKED_LM_LOSS_CHECKPOINT"
  echo "M4_FREEZE_LM_HEAD=$M4_FREEZE_LM_HEAD"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "SAVE_STEPS=$SAVE_STEPS"
} | tee "$CONFIG_FILE"

echo "日志文件：$LOG_FILE"
bash scripts/finetune_m4.sh 2>&1 | tee "$LOG_FILE"
