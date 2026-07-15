#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export HF_HOME="${HF_HOME:-/home/yjm/M4-main/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/yjm/M4-main/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/yjm/M4-main/.cache/matplotlib}"
export PYTHONPATH="${PYTHONPATH:-/home/yjm/M4-main/intersuit}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/home/yjm/miniconda3/envs/M4/bin/python}"
MODE="${1:-turntaking-valid}"
MODEL_PATH="${MODEL_PATH:-checkpoints/M4-LongVA-7B-Qwen2}"

case "$MODE" in
  proactive)
    "$PYTHON_BIN" -m local_demo.proactive_cli \
      --model_path "$MODEL_PATH" \
      --frame_fps 1 \
      --video_file local_demo/assets/water.mp4
    ;;
  turntaking-valid)
    "$PYTHON_BIN" -m local_demo.turntaking_cli \
      --video_path local_demo/assets/water.mp4 \
      --model_path "$MODEL_PATH" \
      --question "Can you describe the video?" \
      --new_query "How many people in the video?" \
      --new_query_pos 20 \
      --num_sampled_frames 8
    ;;
  turntaking-interrupt)
    "$PYTHON_BIN" -m local_demo.turntaking_cli \
      --video_path local_demo/assets/water.mp4 \
      --model_path "$MODEL_PATH" \
      --question "Can you describe the video?" \
      --new_query "Sorry to interrupt?" \
      --new_query_pos 20 \
      --num_sampled_frames 8
    ;;
  turntaking-noise)
    "$PYTHON_BIN" -m local_demo.turntaking_cli \
      --video_path local_demo/assets/water.mp4 \
      --model_path "$MODEL_PATH" \
      --question "Can you describe the video?" \
      --new_query "Okay, I see." \
      --new_query_pos 20 \
      --num_sampled_frames 8
    ;;
  baseline-audio-file|baseline-audio-tts|turntaking-audio-file|turntaking-audio-tts)
    ./scripts/run_audio_demos_stable.sh "$MODE"
    ;;
  *)
    echo "未知模式：$MODE" >&2
    echo "可用模式：proactive, turntaking-valid, turntaking-interrupt, turntaking-noise, baseline-audio-file, baseline-audio-tts, turntaking-audio-file, turntaking-audio-tts" >&2
    exit 2
    ;;
esac
