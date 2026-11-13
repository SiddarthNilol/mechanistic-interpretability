#!/bin/bash
#SBATCH --job-name=fc-rollout
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# Closed-loop rollouts on one GPU node: the policy server (openpi env, GPU) in the background,
# and the simulator workers (LIBERO env, CPUs, EGL rendering on the same GPU) in front.
# Submit from flow_control with the simulator-side script and its arguments:
#
#   sbatch jobs/rollout_job.sh src/sim/rollout.py --out results/data/stage1.jsonl --workers 12
#   sbatch jobs/rollout_job.sh src/sim/demo.py --pairs 3
set -euo pipefail
nvidia-smi -L
PORT=$((8000 + SLURM_JOB_ID % 1000))

( source jobs/env.sh && exec python src/policy/server.py --port "$PORT" ) > "server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do  # the checkpoint loads and compiles in a few minutes
    if (exec 3<>/dev/tcp/localhost/$PORT) 2>/dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server died, see server_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
echo "server up on port $PORT"

( source jobs/sim_env.sh && python "$@" --port "$PORT" )
