"""F8 (optional), part 2 of 3: closed-loop rollouts in LIBERO with steering. CPU, next to the server.

    python src/f8_rollout.py --out results/data/f8_rollouts.jsonl

NOT RUN YET (see f8_steer_server.py). Runs in the LIBERO environment of openpi's
examples/libero (Python 3.8, see its README), not in openpi's main environment. It therefore
imports nothing from this repository. The loop follows examples/libero/main.py.

One task, 10 conditions: no steering, and 3 directions x 3 strengths. Every condition runs the
same 10 initial states, with the same flow noise (the server seeds it by episode and step). One
line per episode is appended to --out: success, and the mean end-effector speed in m/s.

Rough time, not measured: 100 episodes of at most 220 steps, about 1 hour next to one GPU.
Say so before running, it is the longest pilot.
"""

import argparse
import collections
import json
import math
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # as in openpi
CONTROL_HZ = 20  # LIBERO's control frequency
DIRECTIONS = ("progress", "progress_readable", "random")


def quat2axisangle(quat):
    """Copied from openpi's examples/libero/main.py (from robosuite)."""
    quat = np.array(quat)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rollout(env, client, task, init_state, condition, episode, args):
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    positions = []
    done = False
    for t in range(MAX_STEPS[args.suite] + args.wait_steps):
        if t < args.wait_steps:  # let the objects settle, as openpi does
            obs, _, done, _ = env.step(DUMMY_ACTION)
            continue
        positions.append(np.array(obs["robot0_eef_pos"]))
        if not plan:
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # rotate 180 degrees, as in training
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            chunk = client.infer({
                "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224)),
                "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": str(task.language),
                "steer/direction": condition["direction"],
                "steer/alpha": condition["alpha"],
                "steer/episode": episode,
                "steer/step": t,
            })["actions"]
            plan.extend(chunk[: args.replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
    steps = np.linalg.norm(np.diff(np.stack(positions), axis=0), axis=1) if len(positions) > 1 else np.zeros(1)
    return {"success": bool(done), "mean_speed_m_per_s": float(steps.mean() * CONTROL_HZ), "steps": len(positions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--suite", default="libero_spatial", choices=sorted(MAX_STEPS))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                        help="steering strengths, in units of a typical token's residual norm")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7, help="environment seed, openpi's default")
    args = parser.parse_args()

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    conditions = [{"direction": "none", "alpha": 0.0}]
    conditions += [{"direction": d, "alpha": a} for d in DIRECTIONS for a in args.alphas]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for condition in conditions:
            for episode in range(args.episodes):
                result = rollout(env, client, task, init_states[episode], condition, episode, args)
                record = dict(condition, episode=episode, suite=args.suite, task_id=args.task_id,
                              task=str(task.language), seed=args.seed, **result)
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(record, flush=True)


if __name__ == "__main__":
    main()
