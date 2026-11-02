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

## 5. Results to the Mac

Copy `scores.csv`, `attention_groups.csv`, `action_report.txt`, `action_report.csv`, `stable_heads.csv`
and `figures/` of each run into
`pi05_base/` and `pi05_base_images/` in the local repository (the large `.npy` files stay on
HPC). The conclusions are in `RESULTS.md`.
