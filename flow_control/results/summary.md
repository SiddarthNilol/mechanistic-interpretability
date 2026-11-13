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
  the paper's. Every failure in both conditions is a timeout (400 steps), and in 37 of 39
  unsteered failures and all 15 steered failures the bowl was never held at a replan, so the
  failures are mostly failed or missing grasps. Steering made task 60 slower: successful steered
  episodes took 229 steps on average against 124 unsteered.
- Does not show: that the remaining 15 failures on task 12 cannot be fixed (the controller never
  commands the gripper, and `held` is read only at replans); robustness to disturbances (Stage 2);
  anything without privileged state (the oracle controller reads the simulator). One noise
  stream per scene, so run-to-run variation is not measured. "50/50" on task 60 is this run's
  count; its 95% interval is 93% to 100%, not a guarantee.
- Numbers: `results/data/stage1_summary.json`; figure `results/figures/stage1.png`.
