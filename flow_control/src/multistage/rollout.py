"""Multi-stage experiment: closed-loop episodes on LIBERO-90 tasks 3, 4, 63, 64. LIBERO env, CPU
workers, against the unchanged policy server (src/policy/server.py).

    sbatch --time=02:00:00 jobs/rollout_job.sh src/multistage/rollout.py \
        --out results/data/multistage_task63.jsonl --tasks 63 --workers 14

The loop is src/sim/rollout.py's (reset to one of 50 fixed initial states, at most 400 steps,
replan every 5 steps, paired noise via injection.episode_seed), with the subgoal oracle
(oracle.py) in place of the single-target controller. Conditions:

    unsteered   no command
    steered     plain Flow Control with the oracle's command
    adaptive    steered, plus Stage 3's adaptive push strength (threshold 0.8) while the gate
                is on (nothing held, more than 8 cm from the point being reached)

Per episode: success, steps, per-replan trace (phase, goal predicates, held flags, command,
gate, M used), and the first step each goal predicate held. One JSON line per episode,
resumable.
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
sys.path.insert(0, str(HERE.parents[0] / "sim"))
sys.path.insert(0, str(HERE.parents[0] / "policy"))
import injection  # noqa: E402
import libero_env as L  # noqa: E402
import oracle as O  # noqa: E402

REPLAN = 5
CONDITIONS = ("unsteered", "steered", "adaptive")
ADAPTIVE_THRESHOLD = 0.8  # as Stage 3 (src/sim/rollout.py)

_worker: dict = {}


def _init(host: str, port: int):
    from openpi_client import websocket_client_policy

    _worker["client"] = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    _worker["envs"] = {}


def episode(job: tuple) -> dict:
    task_index, state_index, condition, stream = job
    spec = O.SPECS[task_index]
    if task_index not in _worker["envs"]:
        _worker["envs"][task_index] = O.make_env(task_index)
    env, instruction, initial_states = _worker["envs"][task_index]
    started = time.time()
    obs = L.reset(env, initial_states[state_index])
    plan: collections.deque = collections.deque()
    trace, first_true, replan, done = [], {}, 0, False
    for step in range(L.MAX_STEPS):
        if not plan:
            s = O.observe(env, obs, spec)
            phase, cmd, gate = O.command(s, spec)
            for i, g in enumerate(s["goals"]):
                if g:
                    first_true.setdefault(str(i), step)
            request = L.policy_input(obs, instruction)
            request["noise_seed"] = injection.episode_seed(task_index, state_index, replan, stream)
            request["command"] = None if condition == "unsteered" else list(cmd)
            if condition == "adaptive" and gate:
                request["adaptive"] = ADAPTIVE_THRESHOLD
            reply = _worker["client"].infer(request)
            plan.extend(np.asarray(reply["actions"])[:REPLAN])
            trace.append({"step": step, "phase": phase, "goals": s["goals"], "held_first": bool(s["held_first"]),
                          "held_second": bool(s["held_second"]), "eef": s["eef"].round(4).tolist(),
                          "command": list(cmd), "gate": bool(gate), "m": int(reply.get("push_m", 0))})
            replan += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
    goals = O.goal_status(env)
    for i, g in enumerate(goals):
        if g:
            first_true.setdefault(str(i), step + 1)
    return {"task": task_index, "state": state_index, "condition": condition, "stream": stream,
            "success": bool(done), "steps": step + 1, "final_goals": goals, "first_true": first_true,
            "trace": trace, "seconds": round(time.time() - started, 1)}


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=sorted(O.SPECS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--states", default="0-49")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["task"], r["state"], r["condition"], r["stream"]) for r in map(json.loads, out.read_text().splitlines())}
    jobs = [(t, s, c, args.stream) for t in args.tasks for s in parse_range(args.states) for c in args.conditions]
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already in {out})", flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.host, args.port)) as pool:
        for i, r in enumerate(pool.imap_unordered(episode, jobs)):
            with out.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[{i + 1}/{len(jobs)}, {(time.time() - started) / 60:.1f} min] task {r['task']} state {r['state']} "
                  f"{r['condition']}: {'success' if r['success'] else 'failure'} in {r['steps']} steps, "
                  f"goals first true {r['first_true']}", flush=True)


if __name__ == "__main__":
    main()
