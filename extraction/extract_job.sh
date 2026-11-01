#!/bin/bash
#SBATCH --job-name=extract-heads
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h100_tandon,h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# Head extraction on one GPU. Submit from the repository root:
#
#   sbatch extraction/extract_job.sh
#
# DATA and OUT can be overridden: DATA=... OUT=... sbatch extraction/extract_job.sh
# Extra arguments go to extract_heads.py, e.g. --batch-size 32.
set -euo pipefail

source "$SCRATCH/lair-openpi/.venv/bin/activate"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$SCRATCH/.cache/openpi}  # where pi05_base is cached
export PYTHONUNBUFFERED=1
DATA=${DATA:-/scratch/sk12590/ultrarobot-data/unhook-hook-data}
OUT=${OUT:-$SCRATCH/mechanistic-interpretability/pi05_base}

nvidia-smi -L
python extraction/check_capture.py
python extraction/extract_heads.py --dataset "$DATA" --frames prepared/hook_unhook/frames.csv --out "$OUT" "$@"
