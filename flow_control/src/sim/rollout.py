"""Module 4: closed-loop episodes in LIBERO against the policy server. CPU workers, LIBERO env.

    source jobs/sim_env.sh
    python src/sim/rollout.py --out results/data/stage1.jsonl --tasks 12 60 \
        --conditions unsteered steered --states 0-49 --workers 8

The loop is openpi's examples/libero/main.py: reset to one of the task's 50 fixed initial
states, 10 settling steps, then at most 400 steps; every 5 steps ask the server for a 10-step
chunk and execute its first 5 actions (20 Hz). Success is LIBERO's own goal check.

Conditions:
    unsteered   command None
    steered     at every replan, the scripted controller's command (controller.py), computed
                from privileged state

Pairing: replan k of the episode on (task, initial state) uses noise seed
episode_seed(task, state, k, stream) in every condition (injection.py). The two conditions
start from the same scene and the same noise; they diverge once their actions differ.

One JSON line per episode is appended to --out; episodes already in the file are skipped, so a
killed run can be resumed. Workers run episodes in parallel; the server answers one request at
a time, so the GPU is shared.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "policy"))
import controller  # noqa: E402
import injection  # noqa: E402
import libero_env as L  # noqa: E402

REPLAN = 5  # actions executed per chunk, as in openpi and the paper
CONDITIONS = ("unsteered", "steered")

_worker: dict = {}


def _init(host: str, port: int):
    from openpi_client import websocket_client_policy

    _worker["client"] = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    _worker["envs"] = {}


def _env(task_index: int):
    if task_index not in _worker["envs"]:
        _worker["envs"][task_index] = L.make_env(task_index)
    return _worker["envs"][task_index]


def episode(job: tuple, record: bool = False) -> dict:
    """One episode. With record=True the result also has "frames": the front camera image at
    every step (256 x 256, rotated as the policy sees it) and "frame_commands": the command
    active at that step (src/sim/demo.py makes videos from them)."""
    task_index, state_index, condition, stream = job
    task = L.TASKS[task_index]
    env, instruction, initial_states = _env(task_index)
    client = _worker["client"]
    started = time.time()
    obs = L.reset(env, initial_states[state_index])
    plan: collections.deque = collections.deque()
    replan, done, held_ever = 0, False, False
    commands = []
    frames, frame_commands = [], []
    for step in range(L.MAX_STEPS):
        if not plan:
            s = L.state(env, obs, task)
            held_ever |= s["held"]
            command = controller.command(s) if condition == "steered" else None
            commands.append(None if command is None else list(command))
            request = L.policy_input(obs, instruction)
            request["noise_seed"] = injection.episode_seed(task_index, state_index, replan, stream)
            request["command"] = None if command is None else list(command)
            chunk = np.asarray(client.infer(request)["actions"])
            plan.extend(chunk[:REPLAN])
            replan += 1
        if record:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            frame_commands.append(commands[-1])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
    if record:
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        frame_commands.append(commands[-1])
    result = {
        "task": task_index, "state": state_index, "condition": condition, "stream": stream,
        "success": bool(done), "steps": step + 1, "replans": replan, "held_ever": bool(held_ever),
        "commands": commands, "seconds": round(time.time() - started, 1),
    }
    if record:
        result.update(frames=frames, frame_commands=frame_commands)
    return result


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[12, 60])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--states", default="0-49", help="initial states, e.g. 0-49 or 0,3,5")
    parser.add_argument("--stream", type=int, default=0, help="noise stream (a seed for the whole run)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["state"], r["condition"], r["stream"]))
    jobs = [(t, s, c, args.stream) for t in args.tasks for s in parse_range(args.states) for c in args.conditions]
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already in {out})", flush=True)

    started = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.host, args.port)) as pool:
        for i, result in enumerate(pool.imap_unordered(episode, jobs)):
            with out.open("a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[{i + 1}/{len(jobs)}, {(time.time() - started) / 60:.1f} min] task {result['task']} state "
                  f"{result['state']} {result['condition']}: {'success' if result['success'] else 'failure'} "
                  f"in {result['steps']} steps", flush=True)


if __name__ == "__main__":
    main()
