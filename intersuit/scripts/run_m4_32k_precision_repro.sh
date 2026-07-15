#!/usr/bin/env bash
set -euo pipefail

# 32k 保精度复现入口。
# 目标：
# - 保留已有 12k no-freeze checkpoint，不覆盖、不移动、不作为输出目录。
# - 新 32k 训练自动使用未占用的 v1/v2/... 输出目录。
# - 拒绝冻结 lm_head、降低上下文、换起训权重等会改变保精度语义的配置。
#
# 用法：
#   bash scripts/run_m4_32k_precision_repro.sh 32k-canary20
#   bash scripts/run_m4_32k_precision_repro.sh 32k-canary150
#   bash scripts/run_m4_32k_precision_repro.sh 32k-full
#   DRY_RUN=1 RUN_PREFLIGHT=0 bash scripts/run_m4_32k_precision_repro.sh 32k-canary20

cd "$(dirname "$0")/.."

STAGE="${1:-32k-canary20}"
LOG_DIR="${LOG_DIR:-train_logs}"
mkdir -p "$LOG_DIR"

PROTECTED_RUN_NAME="M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze"
PROTECTED_DIR="checkpoints/${PROTECTED_RUN_NAME}"
BASE_RUN_NAME="M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze"
ORIGINAL_CKPT_PATH="checkpoints/M4-LongVA-7B-Qwen2"

if [ ! -d "$PROTECTED_DIR" ]; then
  echo "错误：未找到需要保留的上一版权重目录：$PROTECTED_DIR" >&2
  exit 2
fi

select_versioned_run_name() {
  local base="$1"
  local candidate
  local n=1
  while true; do
    candidate="${base}-v${n}"
    if [ "$candidate" = "$PROTECTED_RUN_NAME" ]; then
      n=$((n + 1))
      continue
    fi
    if [ ! -e "checkpoints/${candidate}" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    n=$((n + 1))
  done
}

case "$STAGE" in
  32k-canary20)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS=20
    DEFAULT_RUN_NAME="$(select_versioned_run_name "${BASE_RUN_NAME}-canary20")"
    export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
    ;;
  32k-canary150)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS=150
    DEFAULT_RUN_NAME="$(select_versioned_run_name "${BASE_RUN_NAME}-canary150")"
    export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
    ;;
  32k-full)
    export MODEL_MAX_LENGTH=32000
    unset MAX_STEPS
    DEFAULT_RUN_NAME="$(select_versioned_run_name "$BASE_RUN_NAME")"
    export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
    ;;
  *)
    echo "未知阶段：$STAGE" >&2
    echo "可用阶段：32k-canary20, 32k-canary150, 32k-full" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MID_RUN_NAME="${MID_RUN_NAME:-$DEFAULT_RUN_NAME}"
export CKPT_PATH="${CKPT_PATH:-$ORIGINAL_CKPT_PATH}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_lowmem.json}"
export M4_CHUNKED_LM_LOSS="${M4_CHUNKED_LM_LOSS:-1}"
export M4_CHUNKED_LM_LOSS_TOKENS="${M4_CHUNKED_LM_LOSS_TOKENS:-512}"
export M4_CHUNKED_LM_LOSS_CHECKPOINT="${M4_CHUNKED_LM_LOSS_CHECKPOINT:-0}"
export M4_FREEZE_LM_HEAD="${M4_FREEZE_LM_HEAD:-0}"
export M4_ALLOW_ZERO3_LOWMEM_RESUME="${M4_ALLOW_ZERO3_LOWMEM_RESUME:-1}"
export SAVE_STEPS="${SAVE_STEPS:-25}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

if [ "$CUDA_VISIBLE_DEVICES" != "0,2,3,4" ] || [ "$NUM_GPUS" != "4" ]; then
  echo "错误：32k 保精度复现必须使用稳定四卡 CUDA_VISIBLE_DEVICES=0,2,3,4 且 NUM_GPUS=4。" >&2
  exit 2
fi
if [ "$MODEL_MAX_LENGTH" != "32000" ]; then
  echo "错误：32k 保精度复现必须保持 MODEL_MAX_LENGTH=32000。" >&2
  exit 2
fi
if [ "$M4_FREEZE_LM_HEAD" != "0" ]; then
  echo "错误：禁止冻结 lm_head；请保持 M4_FREEZE_LM_HEAD=0。" >&2
  exit 2
fi
if [ "$CKPT_PATH" != "$ORIGINAL_CKPT_PATH" ]; then
  echo "错误：必须从原始完整权重起训：CKPT_PATH=$ORIGINAL_CKPT_PATH。" >&2
  exit 2
fi
if [ "$DEEPSPEED_CONFIG" != "scripts/zero3_lowmem.json" ]; then
  echo "错误：当前计划只允许 DEEPSPEED_CONFIG=scripts/zero3_lowmem.json。" >&2
  exit 2
fi
if [ "$M4_CHUNKED_LM_LOSS" != "1" ] || [ "$M4_CHUNKED_LM_LOSS_CHECKPOINT" != "0" ]; then
  echo "错误：必须保持 M4_CHUNKED_LM_LOSS=1 且 M4_CHUNKED_LM_LOSS_CHECKPOINT=0。" >&2
  exit 2
fi
case "$M4_CHUNKED_LM_LOSS_TOKENS" in
  512|256|128) ;;
  *)
    echo "错误：M4_CHUNKED_LM_LOSS_TOKENS 只允许 512、256 或 128。" >&2
    exit 2
    ;;
esac
if [ "$MID_RUN_NAME" = "$PROTECTED_RUN_NAME" ]; then
  echo "错误：MID_RUN_NAME 不能指向上一版受保护权重目录。" >&2
  exit 2
fi
if [ -e "checkpoints/${MID_RUN_NAME}" ]; then
  echo "错误：输出目录已存在，为避免覆盖请换一个 MID_RUN_NAME：checkpoints/${MID_RUN_NAME}" >&2
  exit 2
fi

if [ "${RUN_PREFLIGHT:-1}" = "1" ]; then
  python scripts/check_m4_repro_preflight.py
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.log"
CONFIG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.env"

{
  echo "STAGE=$STAGE"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "PROTECTED_DIR=$PROTECTED_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "NUM_GPUS=$NUM_GPUS"
  echo "MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH"
  echo "MAX_STEPS=${MAX_STEPS:-}"
  echo "MID_RUN_NAME=$MID_RUN_NAME"
  echo "CKPT_PATH=$CKPT_PATH"
  echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
  echo "M4_CHUNKED_LM_LOSS=$M4_CHUNKED_LM_LOSS"
  echo "M4_CHUNKED_LM_LOSS_TOKENS=$M4_CHUNKED_LM_LOSS_TOKENS"
  echo "M4_CHUNKED_LM_LOSS_CHECKPOINT=$M4_CHUNKED_LM_LOSS_CHECKPOINT"
  echo "M4_FREEZE_LM_HEAD=$M4_FREEZE_LM_HEAD"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
  echo "SAVE_STRATEGY=$SAVE_STRATEGY"
  echo "SAVE_STEPS=$SAVE_STEPS"
} | tee "$CONFIG_FILE"

echo "日志文件：$LOG_FILE"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1：配置检查已通过，未启动训练。"
  exit 0
fi

bash scripts/finetune_m4.sh 2>&1 | tee "$LOG_FILE"
