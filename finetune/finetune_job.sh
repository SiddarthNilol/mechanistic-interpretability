#!/bin/bash
#SBATCH --job-name=finetune-heads
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
# Fine-tuning on one GPU. Submit from the repository root, and give the run's directory as OUT:
#
#   OUT=$SCRATCH/mechanistic-interpretability/finetune/unhook_selected sbatch finetune/finetune_job.sh \
#       --task "unhook ring" --selection $SCRATCH/mechanistic-interpretability/pi05_base_denoised/selected_heads.json
#
# Extra arguments go to finetune_heads.py. The time limit is a guess, not measured.
set -euo pipefail

source "$SCRATCH/lair-openpi/.venv/bin/activate"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$SCRATCH/.cache/openpi}  # where pi05_base is cached
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
DATA=${DATA:-/scratch/sk12590/ultrarobot-data/unhook-hook-data}

nvidia-smi -L
python finetune/check_finetune.py
python finetune/finetune_heads.py --dataset "$DATA" --out "$OUT" --rtc-train "$@"
