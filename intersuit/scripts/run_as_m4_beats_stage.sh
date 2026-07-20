#!/usr/bin/env bash
set -euo pipefail

# 第一版 BEATs 最终回答链路：
# 冻结 BEATs -> 可训练 audio projector -> 固定窗口线性映射
# -> 可训练 residual fusion -> M4。
# 不启用动态对齐、复杂 Gate、事件检测辅助训练或 MUG。

INTERSUIT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$INTERSUIT_ROOT/.." && pwd)"
cd "$INTERSUIT_ROOT"

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "错误：必须显式设置 DATA_PATH 为与全部排除集无重叠的训练 manifest。" >&2
  exit 2
fi
if [[ -z "${TRAIN_MANIFEST_AUDIT:-}" || ! -f "$TRAIN_MANIFEST_AUDIT" ]]; then
  echo "错误：必须设置 TRAIN_MANIFEST_AUDIT，且审计文件必须存在。" >&2
  exit 2
fi
FROZEN_EVAL="$INTERSUIT_ROOT/harness/artifacts/frozen_av_task_eval_dev300/frozen_eval.json"
DATA_PATH_FROM_INTERSUIT="$(realpath -m "$DATA_PATH")"
DATA_PATH_FROM_REPO="$(realpath -m "$REPO_ROOT/$DATA_PATH")"
if [[ "$DATA_PATH_FROM_INTERSUIT" == "$(realpath -m "$FROZEN_EVAL")" || "$DATA_PATH_FROM_REPO" == "$(realpath -m "$FROZEN_EVAL")" ]]; then
  echo "错误：禁止使用冻结 300 条评测 manifest 训练 BEATs projector/fusion。" >&2
  exit 2
fi
python - "$DATA_PATH" "$TRAIN_MANIFEST_AUDIT" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]).resolve()
audit_path = pathlib.Path(sys.argv[2]).resolve()
if not manifest.is_file():
    raise SystemExit(f"错误：训练 manifest 不存在：{manifest}")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
required_zero = ("video_id_overlap_count", "youtube_id_overlap_count", "media_sha256_overlap_count", "error_count")
if audit.get("status") != "PASS":
    raise SystemExit("错误：训练 manifest 内容审计不是 PASS")
if pathlib.Path(audit.get("manifest_path", "")).resolve() != manifest:
    raise SystemExit("错误：审计记录的 manifest 路径与 DATA_PATH 不符")
if audit.get("manifest_sha256") != actual:
    raise SystemExit("错误：manifest 已在审计后变化，SHA256 不符")
if any(int(audit.get(key, -1)) != 0 for key in required_zero):
    raise SystemExit("错误：审计包含泄漏、媒体或字段错误")
if not audit.get("exclusion_set_sha256"):
    raise SystemExit("错误：审计缺少排除集合 SHA256")
print("训练 manifest 内容门禁：PASS")
PY

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

exec bash scripts/run_as_m4_stage.sh "${1:-12k-smoke}"
