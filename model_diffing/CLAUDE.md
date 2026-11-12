# CLAUDE.md: Tracker-driven adaptive Flow Control (version 1)

This replaces the earlier pilot CLAUDE.md (patching and readable-subspace figures). Keep that code and its results; reuse its model loading, paired-noise and hook utilities where useful.

## Goal

Extend Flow Control (Kao, Chan, Wang, CoRL 2026, arXiv 2606.10180) so a frozen, fine-tuned VLA can recover forward from mid-task failures such as a dropped object, without a human at the keyboard.

Two additions to their method:

1. A small learned state tracker that replaces their privileged simulator state. It reads the frozen VLA's features and outputs: is the target held, where the target object is, where the goal is.
2. Adaptive steering strength plus a gripper command, so a correction still works after the policy has committed to the wrong action (for example, carrying on with an empty gripper).

Version 1 is post-hoc: the VLA (`pi05_libero`) is never retrained. Co-training the tracker during fine-tuning is version 2 and is out of scope here.

## How Flow Control works (from the paper, verify against code)

- At each replan, sample the action expert's starting noise x0 from N(0, I). If a command is active, overwrite part of x0 with the command, then run the usual flow integration.
- LIBERO setup in the paper (Appendix G): `pi05_libero`, chunks of 10 steps x 7 dims (6 end-effector deltas + gripper), starting noise of shape 10 x 32 in normalized, zero-padded action space, replan every 5 steps at 20 Hz, 10 Euler steps.
- Injection: for the first tau = 5 chunk steps, set the commanded translation dim to +/- sigma with sigma = 1.5, set the other five translation and rotation dims to 0, leave gripper and padding dims as noise.
- Scripted controller: at each replan, command the dominant axis of the vector from end effector to target. Before the grasp the target is the task object; after the grasp it is a point 12 cm above the goal region. It uses privileged simulator state.
- Reported results, LIBERO-90 task 60 ("pick up the black bowl on the left and put it in the tray") and task 12 ("put the black bowl at the back on the plate"), 50 paired trials each: unsteered 27/50 and 23/50, steered 47/50 and 33/50.
- Stronger push (their Appendix D): besides the starting noise, also shift the sample after each of the first M Euler steps by sigma / sqrt(K) along the command (K = number of Euler steps). Larger M changes the policy's mind later in a motion but moves actions away from the policy's normal ones.

## Environment

- NYU HPC Torch cluster, H200 GPUs, SLURM. Reuse the account and partition already recorded in `results/run_log.md`. If missing, ask me.
- openpi in `third_party/openpi` (do not edit it). Our code in `src/`. LIBERO simulator for all rollouts.
- Keep each job under about 2 H200-hours unless I approve more. Rollouts are the bottleneck: run environments in parallel on CPU workers with the policy served from the GPU.

## Stage 0: verify before building (write answers at the top of `results/run_log.md`)

1. Is Flow Control code public? Check the project page https://jcykao.github.io/flowcontrol/ . If yes, use it as the reference. If no, reimplement from the paper and say so.
2. Confirm in openpi the `pi05_libero` chunk length, action dims, noise shape, number of Euler steps and replan interval. Note the flow-time convention (which end is noise) so "the first M integration steps" means the steps nearest the noise.
3. Was `pi05_libero` trained on LIBERO-90? The paper says it was held out. Check the training config or data list.
4. In LIBERO, find how to read: end-effector pose, gripper opening, object and goal poses, and a reliable "object is held" check. Find how to override the gripper command for a few steps and how to move an object.
5. Confirm the task indices: LIBERO-90 tasks 60 and 12 match the two instructions above.

If any of these contradicts this file, stop and tell me.

## Stage 1: reproduce Flow Control (gate)

Implement injection and the scripted controller exactly as above. Run 50 trials per task on each task's 50 fixed initial states, steered and unsteered sharing the same initial state and noise stream.

Gate: steered success is clearly above unsteered on both tasks, in the same direction as the paper. Report our numbers next to theirs. If the gate fails, stop and debug before any later stage.

## Stage 2: forced-failure benchmark

Add automatic disturbances to rollouts:

- Drop: while the object is held and before it reaches the goal, at a random time, override the gripper command to open for a few steps so the object falls, then return control.
- Shift: before the grasp, move the target object a few centimeters.

Run three policies under each disturbance, 50 paired trials per task: unsteered, Flow Control with the privileged scripted controller (oracle tracker), and no disturbance as reference.

Record for every episode: success, and what the policy did after the disturbance (re-grasped, kept moving to the goal with an empty gripper, stalled, other).

Also measure commitment after a drop: at frames just after the drop, inject +command and -command toward the object and report how much the output action changes along that axis (their on-axis response), compared with frames early in the episode.

This stage answers: can steering fix a drop at all, and how committed is the policy to carrying on.

## Stage 3: steering extensions

1. Gripper command: also write the gripper dim of the starting noise (open or close) for the first tau steps. Test with the same +/- response measure whether the gripper output follows.
2. Adaptive strength: at each replan with an active command, generate candidates in one batch with M in {0, 2, 4, 6, 8} (M = 0 is plain Flow Control). Compute fidelity = cosine similarity between the command direction and the mean translation of the first 5 actions of each candidate. Execute the candidate with the smallest M whose fidelity passes a threshold; if none passes, use the largest M. Sweep the threshold over a few values.

Evaluate offline first: on drop frames from Stage 2, plot fidelity against conformity for each fixed M and for the adaptive rule. Conformity = the frozen policy's flow-matching loss on the chunk (paper Eq. 10, fixed set of 32 time and noise draws). Conformity is for evaluation only, not for online selection.

## Stage 4: state tracker (post-hoc, VLA frozen)

Data: rollouts of the frozen `pi05_libero` on the two tasks, nominal and disturbed, with labels read from the simulator at every replan. Split by episode and initial state. Hold out some disturbance timings and shift sizes for testing.

Inputs: frozen features plus proprioception (end-effector position, gripper opening). Compare two feature sources:

- VLA backbone prefix features at one middle layer (choose the layer on validation data).
- The VLA's vision encoder output tokens.

Pool tokens with a small attention-pooling head, not a plain mean (mean pooling has been reported to lose action-relevant detail).

Outputs: held (yes or no), target object position (3D), goal position (3D). Per-task heads are fine for version 1; note this as a limitation.

Train three versions to measure the blind spot: nominal frames only, nominal plus disturbed frames, and (optional) a separate frozen image encoder not taken from the VLA.

Report: held accuracy and AUROC, position error in cm, each split into nominal frames and post-disturbance frames, plus added latency per replan. The tracker must never receive simulator state at test time.

## Stage 5: closed loop

Rule (same as the scripted controller, fed by the tracker): if not held, target = object position and command the gripper open if it is closed; if held, target = goal + 12 cm. Command = dominant axis of end effector to target.

Conditions, 50 paired trials per task, for no disturbance, drop, and shift:

1. Unsteered
2. Flow Control + oracle state
3. Flow Control + tracker
4. Adaptive strength and gripper command + oracle state
5. Adaptive strength and gripper command + tracker (ours)

Statistics: exact McNemar tests on paired episodes (as in the paper) and Wilson 95% intervals.

Failure breakdown for every failed episode: wrong target, drop not recovered, grasp failure, tracker error (tracker disagreed with simulator state at the key moment), timeout, other.

## Stretch (ask me before starting)

- Fine-tune `pi05_libero` with LoRA on about 60 steered recovery successes and test recovery with steering off.
- A vision-language-model tracker as a baseline (the paper's Appendix G.3).
- A multi-stage, order-sensitive task.

## Deliverables

- `results/summary.md`: one short section per stage with what it shows, what it does not show, and the numbers.
- `results/figures/`: Stage 1 bar chart (ours vs paper), Stage 2 behavior breakdown and commitment plot, Stage 3 fidelity vs conformity plot, Stage 4 tracker accuracy (nominal vs disturbed), Stage 5 success table and failure breakdown.
- `results/run_log.md`: every run with command, checkpoint, seeds, trial counts and wall time.

## Rules

- Stage order matters. Do not start a stage until the one before it has results in `summary.md`.
- Privileged simulator state is allowed only for labels, the oracle conditions and scoring.
- Fix and log all seeds. Use the same initial states and noise streams across conditions so comparisons are paired.
- Never overclaim. Do not describe a result as closing the gap to 100%. Report the gap that remains and why.
- Do not modify downloaded checkpoints or commit them to git.
- Plots: matplotlib, one finding per figure, the title states the finding, axes labeled, PNG at 200 dpi and PDF.
- Do not use em dashes in any written text.

## References

- Flow Control: https://arxiv.org/abs/2606.10180
- Flow Reversal Steering: https://arxiv.org/abs/2606.13675
- Assistron (possible overlap, read before writing related work): https://arxiv.org/abs/2606.23147
- openpi: https://github.com/Physical-Intelligence/openpi
- LIBERO: https://arxiv.org/abs/2306.03310