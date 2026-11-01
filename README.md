# Head specialization in a fine-tuned pi0.5

Which of PaliGemma's 144 attention heads (18 layers x 8 heads) carry action, phase, predicate
and task information in a pi0.5 policy fine-tuned on ring hooking/unhooking, following
[arXiv:2511.22697](https://arxiv.org/abs/2511.22697).

Plan:

1. **Dataset** (this step): 20 random episodes per task (hook ring, unhook ring), every 5th
   frame, with task / phase / predicate labels per frame. `data_prep/`
2. Extraction: per-head activations at the first suffix token, `(num_frames, 144, head_dim)`.
3. Scores per head: action, phase, predicates, task; shuffled-label control.
4. Plots: 18x8 heatmaps, top-k overlap, per-phase maps, unhook vs hook, attention rollout.

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
