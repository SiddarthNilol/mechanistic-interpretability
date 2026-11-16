# Results: tracker-driven adaptive Flow Control

## Stage 1: Flow Control reproduction (gate)

Setup: `pi05_libero` on LIBERO-90 tasks 12 and 60, which are held out from its fine-tuning. Each
task's 50 fixed initial states, unsteered and steered, paired (same scene, same noise seed at
every replan). Steering is the paper's baseline (Appendix G.1): the scripted controller's
command written into the first 5 steps of the starting noise (sigma 1.5), with privileged
simulator state.

| task | unsteered, ours | steered, ours | McNemar p | unsteered, paper | steered, paper |
|---|---|---|---|---|---|
| 12: bowl at the back onto the plate | 20/50 | 35/50 | 6.1e-05 | 23/50 | 33/50 |
| 60: left bowl into the tray | 26/50 | 50/50 | 1.2e-07 | 27/50 | 47/50 |

- Shows: the gate passes. Steering raises success on both tasks, in the paper's direction and by a
  similar amount (+15 and +24 episodes against the paper's +10 and +20). In no pair did the
  unsteered episode succeed and the steered one fail. Our unsteered rates are within 3 episodes of
  the paper's. Every failure in both conditions is a timeout (400 steps), and in 48 of 54
  unsteered failures and 14 of 15 steered failures the bowl was never held at a replan, so the
  failures are mostly failed or missing grasps. Steering made task 60 slower: successful steered
  episodes took 229 steps on average against 124 unsteered.
- Does not show: that the remaining 15 failures on task 12 cannot be fixed (the controller never
  commands the gripper, and `held` is read only at replans); robustness to disturbances (Stage 2);
  anything without privileged state (the oracle controller reads the simulator). One noise
  stream per scene, so run-to-run variation is not measured. "50/50" on task 60 is this run's
  count; its 95% interval is 93% to 100%, not a guarantee.
- Numbers: `results/data/stage1_summary.json`; figure `results/figures/stage1.png`.

## Stage 2: forced-failure benchmark

Setup: the Stage 1 episodes (same 50 initial states per task, same noise seeds) under five
conditions applied by the simulator (`src/sim/disturbances.py`), each with unsteered and oracle
Flow Control (the privileged scripted controller):

- none: reference.
- drop: once the bowl is carried (lifted 3 cm, gripper within 8 cm) and before it is over the
  goal, after a random 0 to 20 step delay, the gripper is forced open for 10 steps.
- shift: when the gripper first comes within 12 cm of the bowl (or at step 60), the bowl is
  teleported 3 to 6 cm in a random direction.
- distractor: at the start, the nearest other bowl is placed 12 cm from the target, keeping the
  target the one the instruction names.
- moved goal: with the drop's trigger, the plate or tray is teleported 5 to 8 cm.

Disturbance parameters come from a seed per (task, scene, kind), so both conditions get the same
ones, but each fires when its own episode reaches the trigger. A drop or goal move cannot fire if
the bowl is never carried, so those rows also give the pairs where both episodes were disturbed.

| task | disturbance | unsteered | steered | McNemar p | pairs disturbed in both: unsteered / steered |
|---|---|---|---|---|---|
| 12 | none | 21/50 | 35/50 | 1.2e-04 | |
| 12 | drop | 22/50 | 32/50 | 0.013 | 19/19 vs 17/19 |
| 12 | shift | 16/50 | 28/50 | 4.9e-04 | |
| 12 | distractor | 44/50 | 48/50 | 0.22 | |
| 12 | moved goal | 19/50 | 31/50 | 0.0075 | 16/20 vs 18/20 |
| 60 | none | 26/50 | 48/50 | 4.8e-07 | |
| 60 | drop | 22/50 | 42/50 | 1.1e-05 | 22/28 vs 26/28 |
| 60 | shift | 25/50 | 48/50 | 1.5e-06 | |
| 60 | distractor | 0/50 | 9/50 | 0.0039 | |
| 60 | moved goal | 25/50 | 48/50 | 2.4e-07 | 25/28 vs 28/28 |

Commitment after a drop (`src/policy/commitment.py`): at each saved frame, 8 noise seeds, the
mean of the first 5 actions along the axis from the gripper to the bowl, with +command minus
-command (on-axis response) and the unsteered value signed toward the bowl.

| task | frames | on-axis response, median | unsteered toward the bowl, median |
|---|---|---|---|
| 12 | early (step 10), 100 | 0.17 | 0.20 |
| 12 | after the drop, 110 | 0.21 | 0.49 |
| 60 | early (step 10), 100 | 0.18 | 0.22 |
| 60 | after the drop, 152 | 0.18 | 0.57 |

- Shows:
  - The policy is not committed to carrying on after a drop. It heads back to the fallen bowl
    on its own (unsteered motion toward the bowl is 2.5 times larger than early in the episode),
    a command moves it as much as early in the episode, and it grasps the bowl again in most
    dropped episodes. Continuing to the goal with an empty gripper happened in 1 to 5 episodes
    per group.
  - Once a drop happens, oracle steering does not clearly help recovery (task 12: 17 of 19
    pairs vs 19 of 19 unsteered; task 60: 26 of 28 vs 22 of 28). Its gain in the drop rows comes
    mostly from getting the bowl grasped in the first place.
  - A moved goal costs nothing: both conditions follow the plate or tray to its new place.
  - Shift costs oracle steering on task 12 only (35 to 28, paired p = 0.016).
  - The dominant failure is grasping the wrong bowl, and direction-only steering barely fixes it.
    With a look-alike bowl next to the target in task 60, success falls to 0/50 unsteered and
    9/50 with oracle steering. Even without a disturbance, about half of task 60 episodes grasp
    the other bowl at some point: the 24 steered successes that did took 261 to 388 steps, the
    24 that did not took 100 to 156, with no overlap.
- Does not show:
  - That drops are harmless in general. In task 12 the plate is about 16 cm from the bowl, so the
    drop always fires early in the carry and the bowl falls near where it started. Drops only
    fire in carried episodes (20 to 49 of 50 per group), so the dropped subsets are selected.
  - A valid distractor test on task 12. Success rose (21 to 44 unsteered, 35 to 48 steered):
    moving the nearest bowl beside the target also removed it from where it confused the policy,
    so this condition changes the scene rather than adding difficulty.
  - That the "grasped wrong" label is exact: it is robosuite's two-finger contact check on the
    other bowls at replans, validated only by the step counts above.
  - Run-to-run variation: one noise stream. The no-disturbance row is a rerun of Stage 1 and
    differs by 1 to 2 episodes (task 12 unsteered 21 vs 20, task 60 steered 48 vs 50), the GPU
    arithmetic not being bit-exact between runs.
- What it implies for later stages: the Stage 3 premise (the policy resists correction after a
  drop) does not hold on these two tasks. The failure that steering cannot fix is choosing the
  wrong object, which a direction command cannot express.
- Numbers: `results/data/stage2_summary.json`, `stage2_commitment.json`, per-episode traces in
  `results/data/stage2_<disturbance>.jsonl`; figures `results/figures/stage2_success.png`,
  `stage2_behavior.png`, `stage2_commitment.png`.

## Stage 3 diagnostic: commitment and gripper control at wrong grasps

Setup: new rollouts of the frozen `pi05_libero` (task 60: none and distractor; task 12: none;
both conditions; 300 episodes) saved the policy request at the 2 replans before the first
wrong-bowl grasp (wrong2, wrong1), before the first correct grasp (right2, right1), and at step 10
(early): 940 frames. On each frame, 8 noise seeds: translation commands toward the target
(+/- on its dominant axis), and a gripper command written into the gripper dim of the starting
noise for the first 5 chunk steps (open = -1.5, close = +1.5; the checkpoint's quantile
normalization maps the gripper's range to exactly [-1, 1], so these are beyond fully open and
fully closed).

| task | frames | n | on-axis response | unsteered toward target | chunks that close: unsteered / written open / written close |
|---|---|---|---|---|---|
| 12 | early | 100 | 0.17 | 0.20 | 0% / 0% / 0% |
| 12 | right1 | 49 | 0.07 | 0.00 | 100% / 98% / 100% |
| 12 | wrong1 | 41 | 0.11 | -0.02 | 100% / 99% / 100% |
| 60 | early | 200 | 0.12 | 0.59 | 0% / 0% / 0% |
| 60 | right1 | 81 | 0.11 | -0.23 | 100% / 100% / 100% |
| 60 | wrong1 | 149 | 0.09 | -0.02 | 99% / 96% / 99% |

Medians in LIBERO action units (mean of the first 5 actions); closing = any of the first 5
actions' gripper value above 0. Change in the gripper output, close minus open: 0.002 to 0.004
in every group.

- Shows:
  - The gripper output ignores a gripper command written into the starting noise. Writing
    "open" changes the share of chunks that close by at most 4 points, and "close" never makes
    the policy close early (0% at early frames). So the planned gripper veto, built the paper's
    way (noise only), cannot stop a wrong grasp.
  - Translation commands still act just before a wrong grasp (on-axis 0.09 and 0.11, 66% to 71%
    of the early response), and about as much as before a correct grasp. So the policy is not
    specifically committed at the wrong grasp; responses are smaller near any grasp.
  - But the effect is small for the distance involved: at wrong1 the gripper is 3 cm above and
    4 to 5 cm from the wrong bowl and 12 to 15 cm from the target, while one replan's command
    moves the output by about 0.03 to 0.05 units toward the target. The mistake is made earlier,
    during the descent.
- Does not show: whether a stronger push (Appendix D: shifting the sample during the first M
  Euler steps) moves the gripper or translation more; we only tested the starting noise (M = 0).
  Why the gripper is insensitive (likely the flow pulls it to a near-binary value set by the
  wrist image). One noise stream, frames from one run.
- Implication: a gripper command needs either the stronger push (to be tested offline) or a
  different mechanism (overriding the executed gripper action, which is no longer steering
  through the noise). Translation steering has to act before the descent, not at the grasp.
- Numbers: `results/data/stage3_diagnostic.json` (per frame), `stage3_diagnostic_summary.json`;
  figure `results/figures/stage3_diagnostic.png`.
