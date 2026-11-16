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

### Module 1 probe (simulator access), 2026-11-12

- command: `sbatch jobs/sim_job.sh python src/sim/probe.py`
- jobs: 22929482 (H100; grasp aimed at the bowl center and gripped nothing) and 22935674 (H100,
  grasp at the rim); also 22928294, a duplicate of the first on an H200
- tasks 12 and 60, initial state 0, env seed 0; no policy
- results: action dims 0 to 2 move the end effector along world x, y, z (4.4 to 5.4 cm in 5 steps
  at action 1.0); rim grasp at +5 cm in y holds the bowl (rose 11.3 cm, `is_held` True after
  lifting, False right after closing); opening the gripper drops it (`is_held` False);
  teleporting by 5 cm in x moved task 60's bowl 5.0 cm, task 12's 3.7 cm plus 1.1 cm in y
- output: `results/probe/` (images, `probe_log.txt`)
- wall time: about 1 minute each

### Module 2 check (on-axis response), 2026-11-12

- command: `sbatch jobs/gpu_job.sh python src/policy/check_server.py`
- job: 22935702 (H200); checkpoint `gs://openpi-assets/checkpoints/pi05_libero`, openpi default
  precision (bfloat16); noise seeds 0 to 7; injection tau 5, sigma 1.5
- result: on the probe's saved observation of each task, all 6 commands shift the mean of the
  first 5 actions along their axis in their sign (x, y: 0.10 to 0.17; z: 0.01 to 0.04, LIBERO
  action units), 4 to 10 times the seed-to-seed s.d. Output: `results/probe/steering_check.txt`
- wall time: 1.9 minutes

### Stage 1: Flow Control reproduction, 2026-11-12

- command: `sbatch jobs/rollout_job.sh --out results/data/stage1.jsonl --tasks 12 60 --conditions unsteered steered --states 0-49 --workers 12`
  (submitted before rollout_job.sh took the script name as its first argument)
- job: 22936817 (H100); checkpoint `gs://openpi-assets/checkpoints/pi05_libero`, openpi default
  precision; env seed 0; noise stream 0 (seed per task, initial state, replan); tau 5, sigma 1.5;
  replan every 5 steps; at most 400 steps after 10 settling steps
- trials: LIBERO-90 tasks 12 and 60, initial states 0 to 49, unsteered and steered: 200 episodes
- analysis: `python src/analysis/stage1.py` (exact McNemar, Wilson intervals)
- result: task 12 unsteered 20/50, steered 35/50 (McNemar p = 6.1e-05); task 60 unsteered 26/50,
  steered 50/50 (p = 1.2e-07). Paper: 23 -> 33 and 27 -> 47.
- wall time: 11.4 min (10.0 min of episodes)

### Stage 2 probe (scene facts), 2026-11-16

- command: `sbatch jobs/sim_job.sh python src/sim/probe_stage2.py`; job 23318775 (H200), no policy
- result: task 60 target is the smallest-y bowl in 50/50 states ("left"); task 12 target is the
  smallest-x and smallest-y bowl in 50/50 ("at the back", nearest the robot) and always the bowl
  nearest the gripper's start. Two bowls need 12 cm center to center (8 cm: overlap, 10 cm: pushed
  about 10 cm). Task 12's plate collides with bowls in some directions. Output:
  `results/probe/stage2_probe.json`
- check of the chosen distractor rule: `sbatch jobs/sim_job.sh python src/sim/disturbances.py`,
  job 23320329: placement error 0.00 cm, target keeps its role in 50/50 states for both tasks
- wall time: about 4 minutes each

### Stage 2 smoke tests, 2026-11-16

- jobs 23320330 (3 states, all disturbances) and 23377398 (5 states, drop, shift, moved goal),
  outputs in `/scratch/sk12590/fc/smoke*` (not committed)
- found and fixed before the full run: a drop "halfway" landed on task 12's plate; the grasp
  check was not always True during a carry so some drops never fired; the shift fired at step 0
  in task 12 (horizontal trigger, the gripper starts above the bowl). Commit 5bbb7d6.

### Stage 2 rollouts, 2026-11-16

- command, one job per disturbance k: `sbatch --job-name=fc-s2-$k jobs/rollout_job.sh src/sim/rollout.py --out results/data/stage2_$k.jsonl --tasks 12 60 --states 0-49 --disturbances $k --workers 14`
  (drop also `--frames /scratch/sk12590/fc/stage2_frames`)
- jobs: none 23378218, drop 23378219, shift 23378220, distractor 23378221 (H100, 11.6 to 12.4
  min each). moved_goal: 23378222 crashed after 27 episodes (KeyError when the move was skipped,
  fixed in d4884fb); 23379584 hung on node gh004 (websocket keepalive timeout, no episodes);
  23380986 resumed and finished (10.5 min)
- checkpoint `gs://openpi-assets/checkpoints/pi05_libero`; env seed 0; noise stream 0; tau 5,
  sigma 1.5; disturbance seed per (stream, task, state, kind); at most 400 steps
- trials: 2 tasks x 50 initial states x 2 conditions x 5 disturbances = 1000 episodes
- analysis: `python src/analysis/stage2.py`

### Stage 2 commitment test, 2026-11-16

- command: `sbatch --time=01:00:00 jobs/gpu_job.sh python src/policy/commitment.py --frames /scratch/sk12590/fc/stage2_frames --out results/data/stage2_commitment.json`
- job 23379590 (H100); noise seeds 0 to 7; frames: 200 drop episodes, early (replan 2) and the
  first two replans after the forced opening: 462 frames
- wall time: 9.7 minutes

### Stage 3 diagnostic, 2026-11-16

- rollouts (frames only): `sbatch --job-name=fc-s3-t60 jobs/rollout_job.sh src/sim/rollout.py --out /scratch/sk12590/fc/stage3_diag/task60.jsonl --tasks 60 --states 0-49 --disturbances none distractor --grasp-frames /scratch/sk12590/fc/stage3_diag/frames --workers 14`
  (job 23384698, 13.4 min) and the same for task 12 with `--disturbances none` (job 23384699,
  6.6 min); episode outcomes not committed (they repeat Stage 2's conditions)
- diagnostic: `sbatch --time=02:00:00 jobs/gpu_job.sh python src/policy/grasp_diagnostic.py --frames /scratch/sk12590/fc/stage3_diag/frames --out results/data/stage3_diagnostic.json`
  (job 23386179, H100, 34.5 min); noise seeds 0 to 7; tau 5, sigma 1.5 for translation and gripper
- frames: 940 (300 episodes); analysis `python src/analysis/stage3_diagnostic.py`

### Stage 3 stronger push (offline), 2026-11-16

- rollouts (frames only, 6 replans kept before each grasp): task 60, `--disturbances none distractor --grasp-frames /scratch/sk12590/fc/stage3_push/frames --grasp-history 6` (job 23392221, 13.5 min; 200 frame files); episode outcomes not committed
- smoke test: job 23392522 (`--limit`), M = 0 matched openpi `Policy.infer` (max difference 0.00)
- test: `sbatch --dependency=afterok:23392221 --time=02:00:00 --job-name=fc-push jobs/gpu_job.sh python src/policy/push_test.py --frames /scratch/sk12590/fc/stage3_push/frames --out results/data/stage3_push.json`
  (job 23393845, H200, 74.6 min); noise seeds 0 to 3; M in {0, 2, 4, 6, 8}; tau 5, sigma 1.5,
  push sigma / sqrt(10); conformity 32 draws, key seed 0
- frames: 240 (40 per group); analysis `python3 src/analysis/stage3_push.py`

### Stage 3 closed loop, 2026-11-16

- smoke: job 23404015 (task 60 distractor, states 0-3, steered/veto/adaptive/both, 1.9 min)
- command (one job per setting): `sbatch --time=02:00:00 --job-name=fc-s3cl-<task>-<kind> jobs/rollout_job.sh src/sim/rollout.py --out results/data/stage3_cl_task<task>_<kind>.jsonl --tasks <task> --states 0-49 --disturbances <kind> --workers 14`
  (all five conditions by default)
- jobs: 23404358 (60 distractor, 21.3 min), 23404360 (60 none, 12.8 min), 23404363 (12 none,
  13.0 min), 23404368 (12 shift, 14.1 min)
- checkpoint `gs://openpi-assets/checkpoints/pi05_libero`; env seed 0; noise stream 0; tau 5,
  sigma 1.5; gate 8 cm; veto gripper M 6; adaptive M in {0, 2, 4, 6, 8}, threshold 0.8; at most
  400 steps; trials 4 settings x 5 conditions x 50 = 1000 episodes
- analysis: `python3 src/analysis/stage3_closed_loop.py`
