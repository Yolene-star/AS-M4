#!/bin/bash
#SBATCH --job-name=m4-audio
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/finetune-m4-audio.out
#SBATCH --error=./slurm_logs/finetune-m4-audio.error.out


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0


export NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT=$(expr $RANDOM + 1000)
export PORT=$MASTER_PORT
export TORCHRUN="${TORCHRUN:-/home/yjm/miniconda3/envs/M4/bin/torchrun}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"


export PYTHONPATH=$(pwd)
export HF_HOME="${HF_HOME:-/home/yjm/M4-main/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/yjm/M4-main/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/yjm/M4-main/.cache/matplotlib}"
echo $PYTHONPATH

LLM_VERSION="${LLM_VERSION:-checkpoints/LongVA-7B-Qwen2-VoiceAssistant/checkpoint-3000}"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
SPEECH_MODEL_VERSION="${SPEECH_MODEL_VERSION:-checkpoints/whisper/large-v3.pt}"
# SPEECH_MODEL_VERSION="checkpoints/whisper/whisper-large-v3"
SPEECH_MODEL_VERSION_CLEAN="whisper-large"

PROMPT_VERSION=qwen_1_5

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="${MID_RUN_NAME:-M4-Audio-LongVA-7B-Qwen2-train}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH="${CKPT_PATH:-$LLM_VERSION}" # this could also be the previous stage checkpoint
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero1.json}"
DATA_PATH="${DATA_PATH:-inputs/texts/m4-it-qwen-audio.generated.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-inputs/images/llava-next}"
SPEECH_FOLDER="${SPEECH_FOLDER:-inputs/images/llava-next}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-16384}"
SAVE_STEPS="${SAVE_STEPS:-500}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

if command -v module >/dev/null 2>&1; then
    module add cuda11.8
fi


ACCELERATE_CPU_AFFINITY=1 "$TORCHRUN" --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    intersuit/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --speech_folder "${SPEECH_FOLDER}" \
    --mm_tunable_parts "speech_projector,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --speech_encoder ${SPEECH_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --speech_projector_type linear \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (336, 2352), (336, 2688), (336, 3024), (336, 3360), (336, 3696), (336, 4032), (336, 4368), (336, 4704), (336, 5040), (336, 5376), (336, 5712), (336, 6048), (336, 6384), (336, 6720), (336, 7056), (336, 7392), (336, 7728), (336, 8064), (336, 8400), (336, 8736), (336, 9072), (336, 9408), (336, 9744), (336, 10080), (336, 10416), (336, 10752), (336, 11088), (336, 11424), (336, 11760), (336, 12096), (336, 12432), (336, 12768), (336, 13104), (336, 13440), (336, 13776), (336, 14112), (336, 14448), (336, 14784), (336, 15120), (336, 15456), (336, 15792), (336, 16128), (336, 16464), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (672, 2352), (672, 2688), (672, 3024), (672, 3360), (672, 3696), (672, 4032), (672, 4368), (672, 4704), (672, 5040), (672, 5376), (672, 5712), (672, 6048), (672, 6384), (672, 6720), (672, 7056), (672, 7392), (672, 7728), (672, 8064), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1008, 2352), (1008, 2688), (1008, 3024), (1008, 3360), (1008, 3696), (1008, 4032), (1008, 4368), (1008, 4704), (1008, 5040), (1008, 5376), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1344, 2352), (1344, 2688), (1344, 3024), (1344, 3360), (1344, 3696), (1344, 4032), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (1680, 2352), (1680, 2688), (1680, 3024), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016), (2016, 2352), (2016, 2688), (2352, 336), (2352, 672), (2352, 1008), (2352, 1344), (2352, 1680), (2352, 2016), (2352, 2352), (2688, 336), (2688, 672), (2688, 1008), (2688, 1344), (2688, 1680), (2688, 2016), (3024, 336), (3024, 672), (3024, 1008), (3024, 1344), (3024, 1680), (3360, 336), (3360, 672), (3360, 1008), (3360, 1344), (3696, 336), (3696, 672), (3696, 1008), (3696, 1344), (4032, 336), (4032, 672), (4032, 1008), (4032, 1344), (4368, 336), (4368, 672), (4368, 1008), (4704, 336), (4704, 672), (4704, 1008), (5040, 336), (5040, 672), (5040, 1008), (5376, 336), (5376, 672), (5376, 1008), (5712, 336), (5712, 672), (6048, 336), (6048, 672), (6384, 336), (6384, 672), (6720, 336), (6720, 672), (7056, 336), (7056, 672), (7392, 336), (7392, 672), (7728, 336), (7728, 672), (8064, 336), (8064, 672), (8400, 336), (8736, 336), (9072, 336), (9408, 336), (9744, 336), (10080, 336), (10416, 336), (10752, 336), (11088, 336), (11424, 336), (11760, 336), (12096, 336), (12432, 336), (12768, 336), (13104, 336), (13440, 336), (13776, 336), (14112, 336), (14448, 336), (14784, 336), (15120, 336), (15456, 336), (15792, 336), (16128, 336), (16464, 336)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 1 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
