#!/bin/bash
#SBATCH --job-name=md-pilots
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
# The GPU pilots on one H200 (whichever of the two H200 partitions starts first). Submit from
# the model_diffing directory:
#
#   sbatch jobs/gpu_job.sh            # all steps
#   sbatch jobs/gpu_job.sh f3 f4      # some steps
#
# Steps: f3 f4 f5 f6 f7. The sanity check (check_pair.py --real) always runs first, and the job
# stops if it fails. Both models in float32 take about 27 GB of GPU memory.
# Rough time on one H200, not measured yet: about 1 hour for all steps, most of it loading the
# two checkpoints once per step.
set -euo pipefail
source jobs/env.sh
nvidia-smi -L

STEPS=${*:-f3 f4 f5 f6 f7}
python src/check_pair.py --real
for step in $STEPS; do
    case $step in
        f3) python src/f3_module_patching.py ;;
        f4) python src/f4_layer_patching.py ;;
        f5) python src/f5_progress_probe.py ;;
        f6) python src/f6_readable_spectrum.py ;;
        f7) python src/f7_readable_fraction.py ;;
        *) echo "unknown step: $step" >&2; exit 1 ;;
    esac
done
