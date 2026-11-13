#!/bin/bash
#SBATCH --job-name=fc-gpu
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log
# One command in openpi's environment on a GPU node (the policy side).
# Submit from flow_control:
#
#   sbatch jobs/gpu_job.sh python src/policy/check_server.py
set -euo pipefail
source jobs/env.sh
nvidia-smi -L
"$@"
