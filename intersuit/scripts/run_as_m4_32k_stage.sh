#!/usr/bin/env bash
set -euo pipefail

# AS-M4 32k staged launcher. 32k must go through canaries again even though the
# original M4 32k baseline has completed. It forwards the same rollback knobs
# used by run_as_m4_stage.sh and records them in the generated .env file:
#   AS_M4_ROLLBACK_MODE=behavior   disable scene_audio and force gate=0
#   AS_M4_ROLLBACK_MODE=gate0      keep probes but force residual gate=0
#   AS_M4_ROLLBACK_MODE=weights12k load saved 12k baseline and disable audio
#   AS_M4_ROLLBACK_MODE=weights32k load saved 32k baseline and disable audio
# Freezing modules only stops future updates; it does not restore changed
# checkpoint weights. Use weights12k/weights32k or an explicit CKPT_PATH for
# true weight rollback.

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  STAGE="${2:-32k-canary20}"
else
  STAGE="${1:-32k-canary20}"
fi

case "$STAGE" in
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
    echo "可用阶段：32k-canary20, 32k-canary150, 32k-full" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec bash scripts/run_as_m4_stage.sh --dry-run "$STAGE"
fi

exec bash scripts/run_as_m4_stage.sh "$STAGE"
