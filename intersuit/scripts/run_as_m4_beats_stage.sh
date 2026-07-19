#!/usr/bin/env bash
set -euo pipefail

# 第一版 BEATs 最终回答链路：
# 冻结 BEATs -> 可训练 audio projector -> 固定窗口线性映射
# -> 可训练 residual fusion -> M4。
# 不启用动态对齐、复杂 Gate、事件检测辅助训练或 MUG。

INTERSUIT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$INTERSUIT_ROOT/.." && pwd)"
cd "$INTERSUIT_ROOT"

export AS_M4_ENABLE_SCENE_AUDIO=1
export AS_M4_SCENE_AUDIO_ENCODER_TYPE=beats
export CKPT_PATH="${CKPT_PATH:-$INTERSUIT_ROOT/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze}"
export AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT="${AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT:-$INTERSUIT_ROOT/checkpoints/BEATs_iter3_plus_AS2M.pt}"
export AS_M4_SCENE_AUDIO_BEATS_CODE_ROOT="${AS_M4_SCENE_AUDIO_BEATS_CODE_ROOT:-$REPO_ROOT/third_party/OmniMMI/baselines/videollama2/model}"
if [[ ! -f "$AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT" ]]; then
  echo "错误：本地 BEATs checkpoint 不存在：$AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$AS_M4_SCENE_AUDIO_BEATS_CODE_ROOT/beats/BEATs.py" ]]; then
  echo "错误：本地 BEATs 源码不存在：$AS_M4_SCENE_AUDIO_BEATS_CODE_ROOT/beats/BEATs.py" >&2
  exit 2
fi
export AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT_SHA256="${AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT_SHA256:-$(sha256sum "$AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT" | awk '{print $1}')}"
export AS_M4_FUSION_MODE=beats_simple_residual
export AS_M4_SIMPLE_AUDIO_GATE="${AS_M4_SIMPLE_AUDIO_GATE:-1.0}"
export AS_M4_FUSION_INIT="${AS_M4_FUSION_INIT:-zero}"
export AS_M4_ENABLE_AUDIO_CONFIDENCE_GATE_V1=0
export AS_M4_ENABLE_AUDIO_EVENT_ALIGNER_V1=0
export MM_TUNABLE_PARTS=scene_audio_projector,streaming_av_fusion
export FRAMES_UPBOUND="${FRAMES_UPBOUND:-32}"
export VIDEO_FPS="${VIDEO_FPS:-1}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-none}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"
export MID_RUN_NAME="${MID_RUN_NAME:-AS-M4-BEATs-simple-residual}"

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "错误：必须显式设置 DATA_PATH 为与冻结 300 条评测集无重叠的训练 manifest。" >&2
  exit 2
fi
FROZEN_EVAL="$INTERSUIT_ROOT/harness/artifacts/frozen_av_task_eval_dev300/frozen_eval.json"
DATA_PATH_FROM_INTERSUIT="$(realpath -m "$DATA_PATH")"
DATA_PATH_FROM_REPO="$(realpath -m "$REPO_ROOT/$DATA_PATH")"
if [[ "$DATA_PATH_FROM_INTERSUIT" == "$(realpath -m "$FROZEN_EVAL")" || "$DATA_PATH_FROM_REPO" == "$(realpath -m "$FROZEN_EVAL")" ]]; then
  echo "错误：禁止使用冻结 300 条评测 manifest 训练 BEATs projector/fusion。" >&2
  exit 2
fi

exec bash scripts/run_as_m4_stage.sh "${1:-12k-smoke}"
