#!/usr/bin/env bash
set -euo pipefail

# AS-M4 12k staged launcher. Dry-run is intentionally supported so each
# integration phase can be smoke-tested without starting a long training job.

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  STAGE="${2:-12k-smoke}"
else
  STAGE="${1:-12k-smoke}"
fi

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
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-12288}"
export M4_CHUNKED_LM_LOSS="${M4_CHUNKED_LM_LOSS:-1}"
export M4_CHUNKED_LM_LOSS_TOKENS="${M4_CHUNKED_LM_LOSS_TOKENS:-512}"
export M4_CHUNKED_LM_LOSS_CHECKPOINT="${M4_CHUNKED_LM_LOSS_CHECKPOINT:-0}"
export M4_FREEZE_LM_HEAD="${M4_FREEZE_LM_HEAD:-0}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_lowmem.json}"
export AS_M4_ENABLE_SCENE_AUDIO="${AS_M4_ENABLE_SCENE_AUDIO:-1}"
export AS_M4_SCENE_AUDIO_ENCODER_TYPE="${AS_M4_SCENE_AUDIO_ENCODER_TYPE:-dummy}"
export AS_M4_STREAMING_AV_LR="${AS_M4_STREAMING_AV_LR:-1e-4}"
export AS_M4_SCENE_AUDIO_PROJECTOR_LR="${AS_M4_SCENE_AUDIO_PROJECTOR_LR:-1e-4}"
export AS_M4_FORCE_AUDIO_GATE="${AS_M4_FORCE_AUDIO_GATE:-}"
export MM_TUNABLE_PARTS="${MM_TUNABLE_PARTS:-mm_vision_tower,mm_mlp_adapter,mm_language_model,streaming_av_module}"
export DATA_PATH="${DATA_PATH:-inputs/texts/m4-it-qwen.json}"
export VIDEO_FEATURE_FOLDER="${VIDEO_FEATURE_FOLDER:-}"
export SAVE_STEPS="${SAVE_STEPS:-25}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$STAGE" in
  12k-smoke)
    export MAX_STEPS="${MAX_STEPS:-2}"
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-12k-smoke}"
    ;;
  12k-canary20)
    export MAX_STEPS="${MAX_STEPS:-20}"
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-12k-canary20}"
    ;;
  12k-canary150)
    export MAX_STEPS="${MAX_STEPS:-150}"
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-12k-canary150}"
    ;;
  12k-full)
    unset MAX_STEPS
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-12k-full}"
    ;;
  32k-canary20)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS="${MAX_STEPS:-20}"
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-32k-canary20}"
    ;;
  32k-canary150)
    export MODEL_MAX_LENGTH=32000
    export MAX_STEPS="${MAX_STEPS:-150}"
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-32k-canary150}"
    ;;
  32k-full)
    export MODEL_MAX_LENGTH=32000
    unset MAX_STEPS
    export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-32k-full}"
    ;;
  *)
    echo "未知阶段：$STAGE" >&2
    echo "可用阶段：12k-smoke, 12k-canary20, 12k-canary150, 12k-full, 32k-canary20, 32k-canary150, 32k-full" >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.env"
LOG_FILE="$LOG_DIR/${MID_RUN_NAME}_${STAMP}.log"

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
  echo "AS_M4_ENABLE_SCENE_AUDIO=$AS_M4_ENABLE_SCENE_AUDIO"
  echo "AS_M4_SCENE_AUDIO_ENCODER_TYPE=$AS_M4_SCENE_AUDIO_ENCODER_TYPE"
  echo "AS_M4_STREAMING_AV_LR=$AS_M4_STREAMING_AV_LR"
  echo "AS_M4_SCENE_AUDIO_PROJECTOR_LR=$AS_M4_SCENE_AUDIO_PROJECTOR_LR"
  echo "AS_M4_FORCE_AUDIO_GATE=$AS_M4_FORCE_AUDIO_GATE"
  echo "MM_TUNABLE_PARTS=$MM_TUNABLE_PARTS"
  echo "DATA_PATH=$DATA_PATH"
  echo "VIDEO_FEATURE_FOLDER=$VIDEO_FEATURE_FOLDER"
  echo "SAVE_STEPS=$SAVE_STEPS"
} | tee "$CONFIG_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1，仅写入配置：$CONFIG_FILE"
  exit 0
fi

if [[ -e "checkpoints/$MID_RUN_NAME" ]]; then
  echo "拒绝覆盖已有输出目录：checkpoints/$MID_RUN_NAME" >&2
  exit 3
fi

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  python scripts/check_m4_repro_preflight.py --allowed_power_limits "${ALLOWED_POWER_LIMITS:-300.00 W,450.00 W}"
fi

echo "日志文件：$LOG_FILE"
bash scripts/finetune_m4.sh 2>&1 | tee "$LOG_FILE"
