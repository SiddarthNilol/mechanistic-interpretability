# Commands, step by step

Everything runs on NYU HPC from the repository root, in the lair-openpi environment.
The Mac is only used to edit code (push to GitHub, `git pull` on HPC) and to look at figures.

## 0. Setup (every new shell)

```bash
cd $SCRATCH/mechanistic-interpretability
git pull
source $SCRATCH/lair-openpi/.venv/bin/activate
DATA=/scratch/sk12590/ultrarobot-data/unhook-hook-data
OUT=prepared/hook_unhook
```

## 1. Dataset preparation (`data_prep/`)

```bash
# per-episode labels (open_end, distractors) renumbered for the hook + unhook dataset
python data_prep/cycle_labels.py --dataset $DATA --out $OUT/labels.csv \
    --labels $SCRATCH/lair-train-pipeline/labels.csv

# 20 random episodes per task (seed 0), skipping excluded and unlabelled episodes
python data_prep/select_episodes.py --dataset $DATA --out $OUT --labels $OUT/labels.csv

# automatic phase boundaries from the gripper (t_pipe is left empty)
python data_prep/bootstrap_phases.py --dataset $DATA --out $OUT
```

Annotate `t_pipe` and check the other boundaries in the browser GUI. The GUI runs on an HPC
login node and is reached from the Mac through lair-pc:

```bash
# on the HPC login node
hostname                                             # e.g. torch-login-a-3
python data_prep/annotate.py --dataset $DATA --out $OUT   # serves localhost:8765

# in the VS Code terminal on lair-pc (VS Code forwards the port on to the Mac)
ssh -L 8765:localhost:8765 sk12590@<login node from hostname>
# then open http://localhost:8765 on the Mac
```

After all 40 episodes were reviewed, build the per-frame labels:

```bash
python data_prep/build_labels.py --dataset $DATA --out $OUT   # writes $OUT/frames.csv (6192 frames)
git add prepared/ && git commit -m "Annotated phase boundaries and frame labels" && git push
```

## 2. Extracting the heads (`extraction/`, GPU job)

`extract_job.sh` first runs `check_capture.py` (checks the recording method against openpi
on a tiny model), then `extract_heads.py`.

```bash
# normal run: each frame's own task text and state in the prompt
sbatch extraction/extract_job.sh
#   -> $SCRATCH/mechanistic-interpretability/pi05_base

# images-only run: the same prompt text and the mean state for every frame
OUT=$SCRATCH/mechanistic-interpretability/pi05_base_images \
    sbatch extraction/extract_job.sh --prompt ring --mean-state
```

Follow a job with `squeue -u $USER` and `tail -f extract-heads_<job id>.log`.

## 3. Scoring and plots (`analysis/`, CPU)

Run in an interactive job, not on the login node (the files are large):

```bash
srun --account=torch_pr_564_tandon_advanced --cpus-per-task=8 --mem=16G --time=1:00:00 --pty bash
source $SCRATCH/lair-openpi/.venv/bin/activate
cd $SCRATCH/mechanistic-interpretability

for H in $SCRATCH/mechanistic-interpretability/pi05_base $SCRATCH/mechanistic-interpretability/pi05_base_images; do
    python analysis/score_heads.py      --heads $H   # scores.csv, recalls.csv
    python analysis/plot_heads.py       --heads $H   # figures/heatmaps.png, phases_*.png, overlap.png, top_heads.csv
    python analysis/attention_groups.py --heads $H   # attention_groups.csv, figures/attention_groups.png
done
```

## 4. The paper's head score: predicting the action (`analysis/`, CPU)

Uses the head outputs already extracted, no GPU. About 30-60 minutes per run:

```bash
srun --account=torch_pr_564_tandon_advanced --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash
source $SCRATCH/lair-openpi/.venv/bin/activate
cd $SCRATCH/mechanistic-interpretability
DATA=/scratch/sk12590/ultrarobot-data/unhook-hook-data

for H in $SCRATCH/mechanistic-interpretability/pi05_base $SCRATCH/mechanistic-interpretability/pi05_base_images; do
    python analysis/action_scores.py --dataset $DATA --heads $H   # action_scores.csv, action_splits.csv
    python analysis/action_report.py --heads $H | tee $H/action_report.txt   # the tests, figures/action_*.png
    python analysis/action_specificity.py --heads $H   # action_specificity.csv: do heads prefer a task or phase?
done
```

## 5. The paper's head selection, as in its released code

First a new extraction on the GPU, feeding the action tokens as the paper's code does. It denoises
every frame, so it takes longer than step 2, hence the longer time limit:

```bash
OUT=$SCRATCH/mechanistic-interpretability/pi05_base_denoised \
    sbatch --time=04:00:00 extraction/extract_job.sh --denoised
```

Then the selection, on the CPU:

```bash
srun --account=torch_pr_564_tandon_advanced --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
source $SCRATCH/lair-openpi/.venv/bin/activate
cd $SCRATCH/mechanistic-interpretability
DATA=/scratch/sk12590/ultrarobot-data/unhook-hook-data
H=$SCRATCH/mechanistic-interpretability/pi05_base_denoised

python analysis/select_heads.py --dataset $DATA --heads $H | tee $H/selection_report.txt
```

## 6. Fine-tuning with the selected heads (`finetune/`, GPU job)

The job first runs `check_finetune.py`, then trains. Run the selected heads and the random control,
so the result can be compared:

```bash
H=$SCRATCH/mechanistic-interpretability/pi05_base_denoised
RUNS=$SCRATCH/mechanistic-interpretability/finetune

OUT=$RUNS/unhook_selected sbatch finetune/finetune_job.sh --task "unhook ring" --selection $H/selected_heads.json
OUT=$RUNS/unhook_random   sbatch finetune/finetune_job.sh --task "unhook ring" --heads random
```

Each run saves 3 checkpoints of about 12 GB each. A new run refuses a directory that already has
one. `served_check.json` compares the held-out loss of the trained model and of the last checkpoint
loaded as the server loads it; the two should be close.

To serve a checkpoint on the robot's inference machine, from the `policy-inference-spec` root in the
lair-openpi environment:

```bash
PYTHONPATH=. $SCRATCH/lair-openpi/.venv/bin/python -m server.server_openpi \
    --checkpoint-path $RUNS/unhook_selected/3000 --task "unhook ring" --rtc-train --host 0.0.0.0 --port 18090
```

`finetune_job.sh` always passes `--rtc-train`, so serve its checkpoints with `--rtc-train` as above.
Delays are 0 to 14 control steps by default, the same as the server's. Compare held-out losses only
between runs that both use RTC or both don't.

## 7. Results to the Mac

Copy `scores.csv`, `attention_groups.csv`, `action_report.txt`, `action_report.csv`, `stable_heads.csv`
and `figures/` of each run into
`pi05_base/` and `pi05_base_images/` in the local repository, `selection_report.txt`,
`selected_heads.json` and `selection_scores.csv` into `pi05_base_denoised/`, and each fine-tuning run's
`run.json` and `metrics.csv` into `finetune_runs/<run>/` (the large `.npy` files and the checkpoints
stay on HPC). The conclusions are in `RESULTS.md`.
