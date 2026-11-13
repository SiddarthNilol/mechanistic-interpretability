#!/bin/bash
#SBATCH --job-name=fc-sim
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log
# One simulator-side command on a GPU node (MuJoCo renders off-screen with EGL on the GPU).
# Submit from flow_control:
#
#   sbatch jobs/sim_job.sh python src/sim/probe.py
set -euo pipefail
source jobs/sim_env.sh
nvidia-smi -L
"$@"
