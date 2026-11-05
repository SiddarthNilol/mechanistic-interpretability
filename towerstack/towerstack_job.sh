#!/bin/bash
#SBATCH --job-name=towerstack-heads
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
# The paper's head selection on the episodes of prepare_episodes.py. Submit from the repository root:
#
#   DATA=/path/to/towerstack-data sbatch towerstack/towerstack_job.sh
#
# Extra arguments go to extract_heads.py. The time limit is a guess: hook and unhook ring's 6,192
# frames were given 4 hours, and the number of frames here depends on the episodes picked.
set -euo pipefail

source "$SCRATCH/lair-openpi/.venv/bin/activate"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$SCRATCH/.cache/openpi}  # where pi05_base is cached
export PYTHONUNBUFFERED=1
OUT=${OUT:-$SCRATCH/mechanistic-interpretability/towerstack_denoised}

nvidia-smi -L
python extraction/check_capture.py
python extraction/extract_heads.py --dataset "$DATA" --frames towerstack/prepared/frames.csv --out "$OUT" --denoised "$@"
python analysis/select_heads.py --dataset "$DATA" --heads "$OUT" | tee "$OUT/selection_report.txt"
