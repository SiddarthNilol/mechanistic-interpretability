# Commands, step by step

Everything runs on NYU HPC from the `model_diffing` directory, in the lair-openpi environment.
The Mac is only used to edit code (push to GitHub, `git pull` on HPC) and to look at figures.

## 0. Setup (every new shell)

```bash
cd $SCRATCH/mechanistic-interpretability
git pull
cd model_diffing
source jobs/env.sh     # activates $SCRATCH/lair-openpi/.venv, sends every cache to $SCRATCH/.cache
```

## 1. Downloads (once, on a login node, needs internet)

The state list is already in `results/states.json` (960 states, 240 marked for patching).
This downloads its 80 episode files (2.5 GB), both checkpoints (25 GB) and the tokenizer:

```bash
python src/prepare.py
```

## 2. Test the code (optional, CPU, about 2 minutes)

Two small random models, no checkpoints needed. Run after any change to `src/pair.py`.
Not on a login node: the process gets killed there for using too much memory (the vision encoder
is full size even in the small models). Use an interactive session:

```bash
srun --account=torch_pr_564_tandon_advanced --cpus-per-task=8 --mem=32G --time=0:30:00 --pty bash
cd $SCRATCH/mechanistic-interpretability/model_diffing && source jobs/env.sh
python src/check_pair.py
```

## 3. F1 + F2: model map and where weights moved (CPU job)

```bash
sbatch jobs/weights_job.sh
#   -> results/figures/F1.png, F2.png, results/data/f1_f2.json, f1_table.md
```

## 4. F3 to F7: the GPU pilots (one H200, about 1 hour, not measured yet)

The job first runs the sanity check on the real models (`check_pair.py --real`) and stops if it fails.

```bash
sbatch jobs/gpu_job.sh              # all steps: f3 f4 f5 f6 f7
sbatch jobs/gpu_job.sh f3 f4        # only some steps
sbatch jobs/gpu_job.sh f6 f7        # f7 needs f5 and f6 to have run before
```

Follow a job:

```bash
squeue -u $USER
tail -f md-pilots_<job id>.log
grep -n "SURPRISING\|Error\|ok:" md-pilots_<job id>.log     # after it finishes
```

A quick test of one step on a few states, in an interactive GPU session:

```bash
srun --account=torch_pr_564_tandon_advanced --partition=h200_tandon,h200_public --gres=gpu:1 \
    --cpus-per-task=8 --mem=96G --time=1:00:00 --pty bash
source jobs/env.sh
python src/check_pair.py --real
python src/f3_module_patching.py --limit 16
```

## 5. Redraw figures without a GPU

Every script keeps its numbers in `results/data/`:

```bash
python src/f1_f2_weights.py --plot-only
python src/f3_module_patching.py --plot-only
python src/f4_layer_patching.py --plot-only
python src/f5_progress_probe.py --plot-only      # --probe-only refits the probes from the saved features
python src/f6_readable_spectrum.py --plot-only
python src/f7_readable_fraction.py
```

## 6. Results to the Mac

The figures, the small json files and the run log are tracked by git; the large `.npz` files are
ignored and stay on HPC.

```bash
git add results/figures results/data/*.json results/data/f1_table.md results/run_log.md
git commit -m "model_diffing: pilot results F1 to F7"
git push
```

Then `git pull` on the Mac and fill in the TODO lines of `results/summary.md`.

## 7. F8: closed-loop steering in LIBERO (optional, not run yet)

Only after F1 to F7. Needs the LIBERO simulator in its own Python 3.8 environment, set up as in
`examples/libero/README.md` of openpi ("Without Docker"). Get a time estimate before the full run.

```bash
# terminal 1, GPU node, lair-openpi environment
source jobs/env.sh
python src/f8_steer_server.py --layer 9 --port 8000

# terminal 2, same node, LIBERO environment: first a smoke test with one episode per condition
python src/f8_rollout.py --out results/data/f8_smoke.jsonl --episodes 1
python src/f8_rollout.py --out results/data/f8_rollouts.jsonl

# afterwards, lair-openpi environment
python src/f8_plot.py
```
