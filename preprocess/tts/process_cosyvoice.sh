#!/bin/bash
set -euo pipefail
#SBATCH --job-name=preproc
#SBATCH --partition=HGX,DGX
#SBATCH --account=research
#SBATCH --qos=lv0a
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=tts.out
#SBATCH --error=tts.error.out

project_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$project_root"

export PYTHONPATH="$project_root/third_party/CosyVoice:$project_root/third_party/Matcha-TTS:${PYTHONPATH:-}"
if command -v module >/dev/null 2>&1; then
	module load cuda11.8
fi

# 批次数量
num_batches=5


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python 脚本路径
python_script="$project_root/preprocess/tts/process_cosyvoice.py"

for batch_id in $(seq 0 $((num_batches - 1))); do
	python3 "$python_script" --batch_id "$batch_id" --num_batch "$num_batches"
done