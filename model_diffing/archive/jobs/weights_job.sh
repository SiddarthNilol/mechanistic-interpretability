#!/bin/bash
#SBATCH --job-name=md-weights
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# F1 and F2 (weights only, no GPU). Submit from the model_diffing directory:
#
#   sbatch jobs/weights_job.sh
set -euo pipefail
source jobs/env.sh
python src/f1_f2_weights.py "$@"
