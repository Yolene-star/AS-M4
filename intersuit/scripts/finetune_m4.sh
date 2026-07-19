#!/bin/bash
#SBATCH --job-name=m4
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/finetune-m4.out
#SBATCH --error=./slurm_logs/finetune-m4.error.out


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0


export NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT=$(expr $RANDOM + 1000)
export PORT=$MASTER_PORT
export TORCHRUN="${TORCHRUN:-/home/yjm/miniconda3/envs/M4/bin/torchrun}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export M4_CHUNKED_LM_LOSS="${M4_CHUNKED_LM_LOSS:-1}"
export M4_CHUNKED_LM_LOSS_TOKENS="${M4_CHUNKED_LM_LOSS_TOKENS:-512}"
export M4_CHUNKED_LM_LOSS_CHECKPOINT="${M4_CHUNKED_LM_LOSS_CHECKPOINT:-0}"
export M4_FREEZE_LM_HEAD="${M4_FREEZE_LM_HEAD:-0}"


export PYTHONPATH=$(pwd)
export HF_HOME="${HF_HOME:-/home/yjm/M4-main/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/yjm/M4-main/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/yjm/M4-main/.cache/matplotlib}"
echo $PYTHONPATH

LLM_VERSION="checkpoints/M4-LongVA-7B-Qwen2"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"


PROMPT_VERSION=qwen_1_5

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="${MID_RUN_NAME:-M4-LongVA-7B-Qwen2-train}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


CKPT_PATH="${CKPT_PATH:-checkpoints/M4-LongVA-7B-Qwen2}" # this could also be the previous stage checkpoint
DATA_PATH="${DATA_PATH:-inputs/texts/m4-it-qwen.json}"
VIDEO_FEATURE_FOLDER="${VIDEO_FEATURE_FOLDER:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3.json}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-12288}"
DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MM_TUNABLE_PARTS="${MM_TUNABLE_PARTS:-mm_vision_tower,mm_mlp_adapter,mm_language_model}"
AS_M4_ENABLE_SCENE_AUDIO_BOOL=False
if [ "${AS_M4_ENABLE_SCENE_AUDIO:-0}" = "1" ] || [ "${AS_M4_ENABLE_SCENE_AUDIO:-false}" = "true" ] || [ "${AS_M4_ENABLE_SCENE_AUDIO:-False}" = "True" ]; then
    AS_M4_ENABLE_SCENE_AUDIO_BOOL=True
fi
AS_M4_ROLLBACK_MODE="${AS_M4_ROLLBACK_MODE:-none}"
echo "AS_M4_ROLLBACK_MODE=${AS_M4_ROLLBACK_MODE}"
if [ "${AS_M4_ROLLBACK_MODE}" = "behavior" ]; then
    AS_M4_ENABLE_SCENE_AUDIO_BOOL=False
    export AS_M4_FORCE_AUDIO_GATE="${AS_M4_FORCE_AUDIO_GATE:-0}"
    echo "AS-M4 behavior rollback: disable scene_audio path and force audio gate to ${AS_M4_FORCE_AUDIO_GATE}."
elif [ "${AS_M4_ROLLBACK_MODE}" = "gate0" ]; then
    export AS_M4_FORCE_AUDIO_GATE="${AS_M4_FORCE_AUDIO_GATE:-0}"
    echo "AS-M4 gate rollback: keep scene_audio probes but force residual fusion gate to ${AS_M4_FORCE_AUDIO_GATE}."
elif [ "${AS_M4_ROLLBACK_MODE}" = "weights12k" ]; then
    CKPT_PATH="${AS_M4_BASELINE_CKPT_12K:-checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze}"
    AS_M4_ENABLE_SCENE_AUDIO_BOOL=False
    echo "AS-M4 weight rollback: loading 12k baseline checkpoint ${CKPT_PATH} and disabling scene_audio."
elif [ "${AS_M4_ROLLBACK_MODE}" = "weights32k" ]; then
    CKPT_PATH="${AS_M4_BASELINE_CKPT_32K:-checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1}"
    AS_M4_ENABLE_SCENE_AUDIO_BOOL=False
    echo "AS-M4 weight rollback: loading 32k baseline checkpoint ${CKPT_PATH} and disabling scene_audio."
elif [ "${AS_M4_ROLLBACK_MODE}" != "none" ]; then
    echo "错误：未知 AS_M4_ROLLBACK_MODE=${AS_M4_ROLLBACK_MODE}。可用值：none, behavior, gate0, weights12k, weights32k。" >&2
    exit 2
fi

if [ -n "${RESUME_FROM_CHECKPOINT:-}" ] && [ "${DEEPSPEED_CONFIG}" = "scripts/zero3_lowmem.json" ] && [ "${M4_ALLOW_ZERO3_LOWMEM_RESUME:-0}" != "1" ]; then
    echo "错误：不能用 scripts/zero3_lowmem.json 直接恢复已有 ZeRO-3 checkpoint。"
    echo "原因：该 lowmem 配置会改变 DeepSpeed 优化器/参数分组，加载 checkpoint 里的 optimizer state 时会触发 parameter group mismatch。"
    echo "建议：从 scripts/zero3.json 生成的 checkpoint 继续训练时继续使用 DEEPSPEED_CONFIG=scripts/zero3.json。"
    echo "如果 checkpoint 本身就是 scripts/zero3_lowmem.json 生成的，可显式设置 M4_ALLOW_ZERO3_LOWMEM_RESUME=1。"
    exit 2
fi

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ -n "${AS_M4_FORCE_AUDIO_GATE:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--force_audio_gate "${AS_M4_FORCE_AUDIO_GATE}")
fi
if [ -n "${AS_M4_AUDIO_DELTA_RATIO_CAP:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--audio_delta_ratio_cap "${AS_M4_AUDIO_DELTA_RATIO_CAP}")
fi
if [ -n "${VIDEO_FOLDER:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--video_folder "${VIDEO_FOLDER}")
fi
if [ -n "${FRAMES_UPBOUND:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--frames_upbound "${FRAMES_UPBOUND}")
fi
if [ -n "${VIDEO_FPS:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--video_fps "${VIDEO_FPS}")
fi
if [ -n "${SCENE_AUDIO_FOLDER:-}" ]; then
    EXTRA_TRAIN_ARGS+=(--scene_audio_folder "${SCENE_AUDIO_FOLDER}")
fi
EXTRA_TRAIN_ARGS+=(--as_m4_fusion_init "${AS_M4_FUSION_INIT:-zero}")
EXTRA_TRAIN_ARGS+=(--as_m4_gate_logit_bias "${AS_M4_GATE_LOGIT_BIAS:--5.0}")
EXTRA_TRAIN_ARGS+=(--as_m4_fusion_mode "${AS_M4_FUSION_MODE:-aligned_gated}")
EXTRA_TRAIN_ARGS+=(--as_m4_simple_audio_gate "${AS_M4_SIMPLE_AUDIO_GATE:-1.0}")

DEEPSPEED_ARGS=()
if [ -n "${DEEPSPEED_CONFIG:-}" ] && [ "${DEEPSPEED_CONFIG}" != "none" ] && [ "${DEEPSPEED_CONFIG}" != "NONE" ]; then
    DEEPSPEED_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
else
    echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}: 不传 --deepspeed，使用普通 DDP/单卡训练路径。"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
    echo "TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE：DDP-only smoke 默认关闭 TorchDynamo，避免 DDPOptimizer higher-order op 失败。"
    export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-True}"
    echo "DDP_FIND_UNUSED_PARAMETERS=$DDP_FIND_UNUSED_PARAMETERS：AS-only smoke 允许尚未接入辅助 loss 的 head 暂时无梯度。"
fi

if command -v module >/dev/null 2>&1; then
    module add cuda11.8
fi

ACCELERATE_CPU_AFFINITY=1 "$TORCHRUN" --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    intersuit/train/train_mem.py \
    "${DEEPSPEED_ARGS[@]}" \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder inputs/images/llava-next \
    --video_feature_folder "${VIDEO_FEATURE_FOLDER}" \
    --mm_tunable_parts "${MM_TUNABLE_PARTS}" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --enable_scene_audio "${AS_M4_ENABLE_SCENE_AUDIO_BOOL}" \
    --scene_audio_encoder_type "${AS_M4_SCENE_AUDIO_ENCODER_TYPE:-dummy}" \
    --scene_audio_torchaudio_bundle "${AS_M4_SCENE_AUDIO_TORCHAUDIO_BUNDLE:-WAV2VEC2_BASE}" \
    --scene_audio_sample_rate "${AS_M4_SCENE_AUDIO_SAMPLE_RATE:-16000}" \
    --scene_audio_beats_checkpoint "${AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT:-intersuit/checkpoints/BEATs_iter3_plus_AS2M.pt}" \
    --scene_audio_beats_code_root "${AS_M4_SCENE_AUDIO_BEATS_CODE_ROOT:-third_party/OmniMMI/baselines/videollama2/model}" \
    --scene_audio_beats_checkpoint_sha256 "${AS_M4_SCENE_AUDIO_BEATS_CHECKPOINT_SHA256:-}" \
    --num_audio_events "${AS_M4_NUM_AUDIO_EVENTS:-25}" \
    --audio_quality_dim "${AS_M4_AUDIO_QUALITY_DIM:-1}" \
    --max_av_offset_sec "${AS_M4_MAX_AV_OFFSET_SEC:-1.5}" \
    --streaming_av_lr "${AS_M4_STREAMING_AV_LR:-1e-4}" \
    --scene_audio_projector_lr "${AS_M4_SCENE_AUDIO_PROJECTOR_LR:-1e-4}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (336, 2352), (336, 2688), (336, 3024), (336, 3360), (336, 3696), (336, 4032), (336, 4368), (336, 4704), (336, 5040), (336, 5376), (336, 5712), (336, 6048), (336, 6384), (336, 6720), (336, 7056), (336, 7392), (336, 7728), (336, 8064), (336, 8400), (336, 8736), (336, 9072), (336, 9408), (336, 9744), (336, 10080), (336, 10416), (336, 10752), (336, 11088), (336, 11424), (336, 11760), (336, 12096), (336, 12432), (336, 12768), (336, 13104), (336, 13440), (336, 13776), (336, 14112), (336, 14448), (336, 14784), (336, 15120), (336, 15456), (336, 15792), (336, 16128), (336, 16464), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (672, 2352), (672, 2688), (672, 3024), (672, 3360), (672, 3696), (672, 4032), (672, 4368), (672, 4704), (672, 5040), (672, 5376), (672, 5712), (672, 6048), (672, 6384), (672, 6720), (672, 7056), (672, 7392), (672, 7728), (672, 8064), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1008, 2352), (1008, 2688), (1008, 3024), (1008, 3360), (1008, 3696), (1008, 4032), (1008, 4368), (1008, 4704), (1008, 5040), (1008, 5376), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1344, 2352), (1344, 2688), (1344, 3024), (1344, 3360), (1344, 3696), (1344, 4032), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (1680, 2352), (1680, 2688), (1680, 3024), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016), (2016, 2352), (2016, 2688), (2352, 336), (2352, 672), (2352, 1008), (2352, 1344), (2352, 1680), (2352, 2016), (2352, 2352), (2688, 336), (2688, 672), (2688, 1008), (2688, 1344), (2688, 1680), (2688, 2016), (3024, 336), (3024, 672), (3024, 1008), (3024, 1344), (3024, 1680), (3360, 336), (3360, 672), (3360, 1008), (3360, 1344), (3696, 336), (3696, 672), (3696, 1008), (3696, 1344), (4032, 336), (4032, 672), (4032, 1008), (4032, 1344), (4368, 336), (4368, 672), (4368, 1008), (4704, 336), (4704, 672), (4704, 1008), (5040, 336), (5040, 672), (5040, 1008), (5376, 336), (5376, 672), (5376, 1008), (5712, 336), (5712, 672), (6048, 336), (6048, 672), (6384, 336), (6384, 672), (6720, 336), (6720, 672), (7056, 336), (7056, 672), (7392, 336), (7392, 672), (7728, 336), (7728, 672), (8064, 336), (8064, 672), (8400, 336), (8736, 336), (9072, 336), (9408, 336), (9744, 336), (10080, 336), (10416, 336), (10752, 336), (11088, 336), (11424, 336), (11760, 336), (12096, 336), (12432, 336), (12768, 336), (13104, 336), (13440, 336), (13776, 336), (14112, 336), (14448, 336), (14784, 336), (15120, 336), (15456, 336), (15792, 336), (16128, 336), (16464, 336)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 1 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --ddp_timeout "${DDP_TIMEOUT}" \
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-False}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile "${TORCH_COMPILE:-False}" \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
