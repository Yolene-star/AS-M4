#!/usr/bin/env bash
set -euo pipefail

# 按音频复现阶段启动训练。数据下载/构建完成后再运行对应阶段。
# 示例：
#   bash scripts/run_m4_audio_repro_stage.sh voiceassistant
#   bash scripts/run_m4_audio_repro_stage.sh llavanext-speech
#   bash scripts/run_m4_audio_repro_stage.sh m4-audio

cd "$(dirname "$0")/.."

STAGE="${1:-m4-audio}"
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

case "$STAGE" in
  voiceassistant)
    export MID_RUN_NAME="${MID_RUN_NAME:-LongVA-7B-Qwen2-VoiceAssistant-repro}"
    TRAIN_SCRIPT="scripts/finetune_voiceassistant.sh"
    ;;
  llavanext-speech)
    export MID_RUN_NAME="${MID_RUN_NAME:-LongVA-7B-Qwen2-LLaVA-NeXT-Speech-repro}"
    TRAIN_SCRIPT="scripts/finetune_llavanextaudio.sh"
    ;;
  m4-audio)
    export MID_RUN_NAME="${MID_RUN_NAME:-M4-Audio-LongVA-7B-Qwen2-repro}"
    TRAIN_SCRIPT="scripts/finetune_m4_audio.sh"
    ;;
  *)
    echo "未知音频阶段：$STAGE" >&2
    echo "可用阶段：voiceassistant, llavanext-speech, m4-audio" >&2
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
  echo "MID_RUN_NAME=$MID_RUN_NAME"
  echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
} | tee "$CONFIG_FILE"

echo "日志文件：$LOG_FILE"
bash "$TRAIN_SCRIPT" 2>&1 | tee "$LOG_FILE"
