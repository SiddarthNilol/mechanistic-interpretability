# Head specialization in a fine-tuned pi0.5

Which of PaliGemma's 144 attention heads (18 layers x 8 heads) carry action, phase, predicate
and task information in a pi0.5 policy fine-tuned on ring hooking/unhooking, following
[arXiv:2511.22697](https://arxiv.org/abs/2511.22697).

Commands for the whole experiment: [COMMANDS.md](COMMANDS.md). Results: [RESULTS.md](RESULTS.md).

Plan:

1. **Dataset**: 20 random episodes per task (hook ring, unhook ring), every 5th
   frame, with task / phase / predicate labels per frame. `data_prep/`
2. **Extraction**: every head's output and attention on every frame, from the base pi0.5
   (no fine-tuning, as the paper selects heads before fine-tuning). `extraction/`
3. **Scores per head**: task, phase, predicates by k-NN; shuffled-label control. `analysis/`
4. **Plots**: 18x8 heatmaps, top-20 overlap, per-phase maps. `analysis/`

## The data

`unhook-hook-data` (226 episodes, 50 fps) records cycles: episode 2k unhooks the ring from the
pipe and drops it on the table, episode 2k+1 picks it up and hooks it back on, starting exactly
where 2k ended. Its unhook episodes are the `unhook-ring-sept25-data` episodes (sept25 episode
k = episode 2k here, identical data), so the per-episode labels in
`lair-train-pipeline/labels.csv` (`open_end`, `distractors`, numbered like sept25) hold for
both halves of a cycle. `cycle_labels.py` converts them.

## Step 1: preparing the dataset

Run on HPC with the lair-openpi environment (it has pandas, pyarrow, av and pillow):

```bash
source $SCRATCH/lair-openpi/.venv/bin/activate
DATA=/scratch/sk12590/ultrarobot-data/unhook-hook-data
OUT=prepared/hook_unhook

python data_prep/cycle_labels.py     --dataset $DATA --out $OUT/labels.csv \
    --labels $SCRATCH/lair-train-pipeline/labels.csv          # labels.csv, combined numbering
python data_prep/select_episodes.py  --dataset $DATA --out $OUT --labels $OUT/labels.csv  # episodes.csv
python data_prep/bootstrap_phases.py --dataset $DATA --out $OUT  # phase_boundaries.csv
python data_prep/annotate.py         --dataset $DATA --out $OUT  # GUI -> phase_boundaries.csv, frames.csv
```

Adjust `DATA` to wherever the dataset lives on HPC. `select_episodes.py` skips
`meta/excluded_episodes.json` and episodes without labels, and picks 20 per task with
`--seed 0`, so the selection is the same on every machine:

- hook ring: 13, 23, 27, 49, 61, 67, 99, 111, 119, 123, 127, 129, 133, 137, 149, 153, 157, 187, 203, 207
- unhook ring: 10, 14, 34, 58, 76, 78, 82, 84, 96, 102, 122, 126, 148, 152, 174, 190, 198, 210, 214, 222

Commit the CSVs in `prepared/`, especially your hand-edited `phase_boundaries.csv`.

### The annotation GUI

`annotate.py` serves a page on `localhost:8765` of the machine it runs on. To open it from
your laptop, start it on an HPC login node, note that node's name (`hostname`), and tunnel to
that same node from your laptop:

```bash
ssh -L 8765:localhost:8765 <netid>@<login node from hostname>
```

then open http://localhost:8765. If the port is taken (login nodes are shared), pass
`--port 8877` and use that number in the tunnel too.

For each episode the page shows the head camera, the active arm's wrist camera (best for
seeing the ring come off or go onto the pipe), and a timeline with the phases, the gripper
trace (top = open, bottom = closed) and the five boundaries. It opens at `t_closed`, after
which `t_pipe` is to be found.

| key | action |
|---|---|
| `←` `→` (`shift`: ×10) | step one frame (ten frames) |
| `space` | play / pause |
| `1`–`5` | set t_align / t_grasp / t_closed / t_pipe / t_release to the current frame |
| `enter` | save this episode and go to the next unreviewed one |

You can also click or drag on the timeline to seek, drag a boundary marker, or type a frame
number. "Save" stays disabled until all five boundaries are set and in order.

Each save writes `phase_boundaries.csv` immediately (the episode's `reviewed` becomes 1), so you
can stop and come back later. Once all 40 are reviewed, the GUI writes `frames.csv` itself (the
summary appears at the top of the page and in the terminal), and rewrites it after any later
edit. `build_labels.py` does the same from the command line, e.g. with another `--stride`.

The first view of an episode takes a few seconds while its video is decoded (the next one is
decoded in the background).

### Phases

Boundaries are frame indices within an episode. Four are bootstrapped from the active arm's
gripper state (1 open, 0 closed) and commanded end-effector position; `t_pipe` is annotated
by hand. The two tasks share every boundary and every phase but one:

| frames                  | unhook ring | hook ring | rule                                                  |
|-------------------------|-------------|-----------|-------------------------------------------------------|
| `[0, t_align)`          | approach    | approach  | arm moving toward the ring                            |
| `[t_align, t_grasp)`    | align       | align     | end effector stays within 5 cm of where it grasps     |
| `[t_grasp, t_closed)`   | grasp       | grasp     | gripper goes from < 0.9 to < 0.2                      |
| `[t_closed, t_pipe)`    | **unhook**  | carry     | ring held; unhook: on the pipe, hook: off it          |
| `[t_pipe, t_release)`   | carry       | **hook**  | ring held; unhook: off the pipe, hook: on it          |
| `[t_release, length)`   | retract     | retract   | gripper back > 0.5: ring let go, arm moves away       |

**`t_pipe`** is the first frame where the ring is off the pipe (unhook) or on it (hook). It has
no automatic value; the GUI won't save an episode, and `build_labels.py` won't run, without it.

If the gripper closes more than once, the boundaries follow the **longest** closure, the one
that moves the ring: the others are failed grabs before it (they fall in align) or a brief
re-touch after the release (falls in retract). These episodes have a `!` in the GUI's episode
list and a note in `phase_boundaries.csv`.

### `frames.csv`

One row per kept frame (about 6,200). Labels for the analysis:

- `task`: hook ring / unhook ring
- `phase`, `phase_id`: the phases above (`phase_id` is the position 0-5 within the task)
- `grasped`: ring held (`t_closed <= frame < t_release`)
- `ring_on_pipe`: ring on the pipe
- `active_arm`, `open_end`, `distractors`: per-episode predicates
- `cycle`: the unhook/hook pair an episode belongs to

`episode_index` + `frame_index` (or the global `index`) identify the frame in the source
dataset. The extraction step runs the model on exactly these rows. Adjacent frames are strongly
correlated and the two halves of a cycle share a scene, so cross-validate grouped by `cycle`,
not by frame. Hook episodes are longer (~1.5x the frames), so balance or weight the tasks when
scoring task identity.

## Step 2: extracting the heads

pi0.5 is two transformers with 18 layers of 8 heads each that share one attention per
layer: PaliGemma (Gemma 2B) processes the prefix (three camera images of 256 tokens each,
then the prompt `Task: <task>, State: <25 numbers>;\nAction: `), and the action expert
(Gemma 300M) processes the suffix (50 noisy action tokens). The prefix attends only to
itself; the action tokens attend to the prefix and to each other.

`extract_heads.py` runs the released `pi05_base` on every frame of `frames.csv` (one
prefix pass and the first denoising step of the suffix, with the same noise for every
frame) and records, for every layer and head, the attention weights over the keys and the
head output `softmax(q kᵀ / √d) v` (the paper's eq. 1, 256 numbers) at two tokens:

- `vlm`: PaliGemma's heads at the last prompt token, which sees images, task and state
- `action`: the action expert's heads at the first action token (the paper's choice)

```bash
sbatch extraction/extract_job.sh     # from the repository root; ~5 GB to $SCRATCH
```

The job first runs `check_capture.py`, which checks on a tiny random pi0.5 that the
recorded head outputs equal openpi's own. Outputs (row i = row i of `frames.csv`):
`heads_{vlm,action}.npy` `(frames, 18, 8, 256)`, `attn_{vlm,action}.npy`
`(frames, 18, 8, keys)`, `tokens.npy`, and `info.json` with the key layout.

The base model has never seen this robot. The state is normalized like openpi does
(1st/99th percentile of each dim over the dataset to -1/1) and written into the prompt;
the cameras fill the slots the lair-train configs use (head, left wrist, right wrist).

## Steps 3-4: scoring the heads and plotting them

```bash
H=$SCRATCH/mechanistic-interpretability/pi05_base
python analysis/score_heads.py --heads $H    # scores.csv, recalls.csv (CPU, ~1 GB RAM, minutes)
python analysis/plot_heads.py  --heads $H    # figures/
```

For every head and label, each frame is classified by majority vote of the 20 frames with
the most similar head output (cosine similarity), never taken from the frame's own cycle.
The score is balanced accuracy. Labels: `task`, the phases within each task
(`phase_hook`, `phase_unhook`), `grasped`, `ring_on_pipe`.

The **control** reruns the same vote with each episode's labels replaced by another
episode's at the same progress through the episode. It keeps the label proportions and
their timing, so a head that only tracks time since the start (or wins by chance) scores
as well on the control. The maps show **score above control**.

Note that the task name is in the prompt, so heads that read the prompt can tell the
tasks apart without looking at the scene.

`figures/`: `heatmaps.png` (label x stream, top 20 outlined), `phases_{hook,unhook}.png`
(each phase's recall above control, per head), `overlap.png` (top-20 heads shared by
two labels), `top_heads.csv`.

## Removing the text shortcuts

The task and the state are both text in the prompt, so heads can score on task and phase
by reading rather than looking. Two checks:

- `analysis/attention_groups.py --heads $H` splits each head's attention into cameras,
  task words, state numbers, the rest of the prompt (incl. `<bos>`, often a "sink" that
  absorbs attention) and action tokens, and compares each label's top heads with all heads.
- `extract_heads.py --prompt ring --mean-state` gives every frame the same prompt text
  and the dataset's mean state, so only the images differ between frames. Rerun steps 3-4
  on its output and compare with the normal run.

```bash
OUT=$SCRATCH/mechanistic-interpretability/pi05_base_images sbatch extraction/extract_job.sh --prompt ring --mean-state
```

## The paper's head score

`action_scores.py` scores heads as the paper does (its eq. 2-5): per task, each frame's action
is predicted as the mean action of the k frames with the most similar head output, from the
task's other episodes, and a head's score is the squared error (lower is better). Two targets:
`position` (the action, joint position commands) and `motion` (how far the command moves the
joints in the next 0.1 s, the closest we have to the paper's joint velocities).

`action_report.py` then tests whether specialized heads exist; the five criteria are in its
docstring, and were fixed before seeing results. The key ones split a task's episodes into two
halves with no episode in common: does the top 20 repeat, and do heads picked on one half still
beat the typical head on the other?

`action_specificity.py` looks closer at whether heads prefer a task or phase, and at how much of
a head's rank comes from its layer. It was added after seeing the results of `action_report.py`.

These scripts follow the paper's text, and differ from its released code in how the action tokens
are fed, which actions are used, and how k is chosen. `extract_heads.py --denoised` and
`select_heads.py` follow the code instead, except that they use all frames rather than keyframes;
the docstring of `select_heads.py` lists what we had to adapt to our robot.

## Fine-tuning

`finetune/finetune_heads.py` fine-tunes pi0.5 on one task as the paper's code does: it trains the
LoRA of the chosen heads (Q and O only, in PaliGemma), the FFN LoRA of every PaliGemma layer, the time
MLP and the action projections, and freezes the rest. `--heads random` and `--heads all` are the
controls. The docstring lists what we had to adapt. Checkpoints have their LoRA merged in, so
`policy-inference-spec/server/server_openpi.py` can serve them. `finetune/check_finetune.py` tests
that only the intended parameters change and that the served checkpoint gives the trained model's
actions. Commands: [COMMANDS.md](COMMANDS.md), step 6.
