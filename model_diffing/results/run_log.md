# Run log: tracker-driven adaptive Flow Control

The earlier pilot (patching and readable subspace) is in `archive/`, with its own run log at
`archive/results/run_log.md`.

## Stage 0: verification (2026-11-12)

Sources: the Flow Control paper (arXiv 2606.10180, HTML version), its project page, openpi at
`aec5d4d` (`/scratch/sk12590/lair-openpi`), and LIBERO's source on GitHub.

1. **Is Flow Control code public?** No. The project page (https://jcykao.github.io/flowcontrol/)
   links only the paper, a video and the lab page; the arXiv page lists no repository. We
   reimplement from the paper (Sections 3 and 4, Appendices D and G).

   Which model: the paper uses two. The real-robot experiments (Franka, joint velocities, 16-step
   chunks, tau = 8, sigma = 1, gain 15) use pi0.5-DROID. The LIBERO simulation experiments
   (Appendix G.1) use "the public pi0.5-LIBERO checkpoint", which is openpi's
   `gs://openpi-assets/checkpoints/pi05_libero`. Our study runs in LIBERO, so we use
   `pi05_libero`, as this file says. pi0.5-DROID cannot be evaluated in LIBERO (different robot
   action space and cameras).

2. **openpi facts for `pi05_libero`** (`src/openpi/training/config.py`, `models/pi0.py`,
   `examples/libero/main.py`):
   - chunk length (action horizon) 10; model action dim 32, of which LIBERO uses the first 7
     (6 end-effector deltas + gripper);
   - starting noise shape (10, 32), drawn from N(0, I) in the normalized, zero-padded action
     space (actions are quantile-normalized with the checkpoint's statistics);
   - 10 Euler steps (`sample_actions(num_steps=10)`);
   - replan every 5 steps (`replan_steps = 5` in `examples/libero/main.py`), LIBERO control at
     20 Hz;
   - `Policy.infer(obs, noise=...)` accepts a starting noise, so plain Flow Control (M = 0) needs
     no change to openpi's model code.
   - **Flow-time convention differs from the paper.** openpi integrates from t = 1 (noise) to
     t = 0 (action), with dt = -0.1 (comment in `sample_actions`). The paper writes t = 0 for
     noise and t = 1 for action. So "the first M integration steps" (Appendix D) means openpi's
     steps starting at t = 1.0, 0.9, ..., the ones nearest the noise.

3. **Was `pi05_libero` trained on LIBERO-90?** No. Its config trains on the Hugging Face dataset
   `physical-intelligence/libero`, whose `meta/tasks.jsonl` has 40 tasks: LIBERO-10, Goal,
   Object and Spatial. Neither instruction below is among them. The paper says the same (G.1:
   "held out from the checkpoint's fine-tuning data"). Note: 10 similar tasks are in training
   ("pick up the black bowl ... and place it on the plate", LIBERO-Spatial), so task 12 is close
   to the training distribution; no tray task is in training.

4. **Reading state in LIBERO.** From openpi's `examples/libero/main.py`: end-effector position
   `obs["robot0_eef_pos"]`, orientation `obs["robot0_eef_quat"]`, gripper finger positions
   `obs["robot0_gripper_qpos"]`. Object poses, a "held" check, overriding the gripper and moving
   an object are NOT verified yet: LIBERO is not installed on Torch (the `third_party/libero`
   submodule of lair-openpi is empty). To be checked in the simulator in Module 1.

5. **Task indices** (LIBERO `benchmark/libero_suite_task_map.py`; LIBERO-90 uses its listed order,
   with no task permutation, `benchmark/__init__.py`):
   - task 12: `KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate`. Target
     `akita_black_bowl_3`, goal `(On akita_black_bowl_3 plate_1)`.
   - task 60: `LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray`.
     Target `akita_black_bowl_1`, goal `(In akita_black_bowl_1 wooden_tray_1_contain_region)`.

   Both match this file.

Paper numbers to reproduce (G.1, 50 paired trials, each task's 50 fixed initial states): task 60
unsteered 27/50, steered 47/50; task 12 unsteered 23/50, steered 33/50 (exact McNemar p < 0.01).
G.3 reports 28/50 and 22/50 unsteered in separate runs. Not stated for LIBERO in the paper: the
number of Euler steps (we use openpi's 10) and the episode step limit (we use openpi's 400 for
LIBERO-90, plus 10 settling steps).

Nothing contradicts CLAUDE.md.

## Runs
