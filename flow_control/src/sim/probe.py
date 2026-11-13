"""Module 1 check: does libero_env.py read and change the simulator correctly? No policy.

    source jobs/sim_env.sh && python src/sim/probe.py

For both tasks, on initial state 0:
  1. builds the env, checks the task name, prints the observation keys and the object names;
  2. prints the privileged state (end effector, target, goal, held) and saves both camera
     images, rotated as the policy sees them, and the whole policy request to results/probe/;
  2b. pushes each action axis and reports where the end effector moves (the action frame);
  3. moves the gripper to the target with a hand-written script (open, go above, descend,
     close, lift) and reports whether is_held turns True and the object rises with the hand;
  4. opens the gripper while lifted (the Stage 2 "drop") and reports that held turns False and
     the object falls;
  5. moves the target by 5 cm (the Stage 2 "shift") and reports the measured displacement.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import libero_env as L  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "probe"


def drive_to(env, obs, goal, gripper, steps=60, gain=10.0):
    """Proportional control of the end-effector position (LIBERO's actions are scaled deltas,
    clipped to [-1, 1])."""
    for _ in range(steps):
        delta = np.clip(gain * (goal - obs["robot0_eef_pos"]), -1, 1)
        obs, _, _, _ = env.step(list(delta) + [0.0, 0.0, 0.0, gripper])
    return obs


def probe(task_index: int):
    task = L.TASKS[task_index]
    env, instruction, initial_states = L.make_env(task_index)
    print(f"\n== task {task_index}: {instruction!r}, {len(initial_states)} initial states")
    obs = L.reset(env, initial_states[0])
    print("observation keys:", sorted(obs.keys()))
    print("objects:", sorted(L._inner(env).objects_dict.keys()))
    s = L.state(env, obs, task)
    print("state:", {k: np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()})

    request = L.policy_input(obs, instruction)
    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for key in ("observation/image", "observation/wrist_image"):
        Image.fromarray(request[key]).save(OUT / f"task{task_index}_{key.split('/')[1]}.png")
    print("policy input:", {k: (v.shape, v.dtype) if hasattr(v, "shape") else v for k, v in request.items()})
    np.savez(OUT / f"task{task_index}_request.npz", **request)  # for src/policy/check_server.py

    # 2b. action frame: does action dim i move the end effector along world axis i?
    for axis in range(3):
        obs = L.reset(env, initial_states[0])
        start = np.array(obs["robot0_eef_pos"])
        action = [0.0] * 7
        action[axis], action[6] = 1.0, -1.0
        for _ in range(5):
            obs, _, _, _ = env.step(action)
        print(f"action +{'xyz'[axis]} for 5 steps moves the end effector by "
              f"{np.round(100 * (obs['robot0_eef_pos'] - start), 2).tolist()} cm (world x, y, z)")
    obs = L.reset(env, initial_states[0])

    # 3. grasp by script
    target = s["target"]
    obs = drive_to(env, obs, target + [0, 0, 0.10], -1.0)
    obs = drive_to(env, obs, target + [0, 0, 0.01], -1.0)
    for _ in range(15):
        obs, _, _, _ = env.step([0.0] * 6 + [1.0])
    z_before = L.body_position(env, task.target)[2]
    obs = drive_to(env, obs, obs["robot0_eef_pos"] + [0, 0, 0.12], 1.0, steps=40)
    z_after = L.body_position(env, task.target)[2]
    print(f"grasp: held = {L.is_held(env, task.target)}, target rose {100 * (z_after - z_before):.1f} cm "
          f"(gripper qpos {np.round(obs['robot0_gripper_qpos'], 3).tolist()})")

    # 4. drop
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    for _ in range(20):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    print(f"drop: held = {L.is_held(env, task.target)}, target z {100 * L.body_position(env, task.target)[2]:.1f} cm "
          f"(was {100 * z_after:.1f} cm)")

    # 5. shift
    obs = L.reset(env, initial_states[0])
    before = L.body_position(env, task.target)
    L.move_object(env, task.target, [0.05, 0.0, 0.0])
    for _ in range(10):
        obs, _, _, _ = env.step(L.SETTLE_ACTION)
    after = L.body_position(env, task.target)
    print(f"shift by 5 cm in x: moved {np.round(100 * (after - before), 1).tolist()} cm, success = {L.is_success(env)}")
    env.close()


if __name__ == "__main__":
    for index in L.TASKS:
        probe(index)
