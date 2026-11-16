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

Stage 2 (--disturbances, disturbances.py): the same loop with a disturbance applied by the
simulator. Every episode then also records a trace, one entry per replan (step, end effector,
gripper opening, target and goal positions, held, any other bowl held), which
src/analysis/stage2.py turns into behavior labels. With --frames DIR, drop episodes also save the
policy request at an early replan and at the first two replans after the drop, for the
commitment test (src/policy/commitment.py).

One JSON line per episode is appended to --out; episodes already in the file are skipped, so a
killed run can be resumed. Workers run episodes in parallel; the server answers one request at
a time, so the GPU is shared.
"""

from __future__ import annotations

import argparse
import collections
import functools
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
import disturbances as D  # noqa: E402
import injection  # noqa: E402
import libero_env as L  # noqa: E402

REPLAN = 5  # actions executed per chunk, as in openpi and the paper
CONDITIONS = ("unsteered", "steered")
EARLY_REPLAN = 2  # commitment test: the "early in the episode" frame (step 10)
POST_DROP_FRAMES = 2  # ... and this many replans after the drop ends

_worker: dict = {}


def _init(host: str, port: int):
    from openpi_client import websocket_client_policy

    _worker["client"] = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    _worker["envs"] = {}


def _env(task_index: int):
    if task_index not in _worker["envs"]:
        _worker["envs"][task_index] = L.make_env(task_index)
    return _worker["envs"][task_index]


def _other_held(env, task) -> str | None:
    for name in L._inner(env).objects_dict:
        if "bowl" in name and name != task.target and L.is_held(env, name):
            return name
    return None


def episode(job: tuple, record: bool = False, frames_dir: str | None = None, grasp_frames: str | None = None,
            grasp_history: int = 2) -> dict:
    """One episode. job = (task, initial state, condition, stream[, disturbance]).

    With record=True the result also has "frames": the front camera image at every step (256 x
    256, rotated as the policy sees it) and "frame_commands": the command active at that step
    (src/sim/demo.py makes videos from them). With a disturbance (Stage 2, even "none") the result has
    "disturbance" (what happened, disturbances.py) and "trace" (one entry per replan)."""
    task_index, state_index, condition, stream = job[:4]
    kind = job[4] if len(job) > 4 else "none"
    stage2 = len(job) > 4  # Stage 2 episodes carry a trace, even with no disturbance
    task = L.TASKS[task_index]
    env, instruction, initial_states = _env(task_index)
    client = _worker["client"]
    started = time.time()
    disturbance = D.Disturbance(kind, task_index, state_index, stream)
    obs = disturbance.at_reset(env, L.reset(env, initial_states[state_index]))
    plan: collections.deque = collections.deque()
    replan, done, held_ever = 0, False, False
    commands, trace, saved = [], [], {}
    recent: collections.deque = collections.deque(maxlen=grasp_history)  # Stage 3: the last replans' requests
    frames, frame_commands = [], []
    for step in range(L.MAX_STEPS):
        if not plan:
            s = L.state(env, obs, task)
            held_ever |= s["held"]
            command = controller.command(s) if condition == "steered" else None
            commands.append(None if command is None else list(command))
            request = L.policy_input(obs, instruction)
            if stage2:
                trace.append({"step": step, "eef": s["eef"].round(4).tolist(), "grip": round(float(s["gripper"][0] - s["gripper"][1]), 4),
                              "target": s["target"].round(4).tolist(), "goal": s["goal"].round(4).tolist(),
                              "held": bool(s["held"]), "other_held": _other_held(env, task)})
            if frames_dir and kind == "drop":
                _maybe_save(saved, disturbance, replan, step, request, s)
            if grasp_frames:
                _grasp_frames(saved, recent, env, task, replan, step, request, s)
            request["noise_seed"] = injection.episode_seed(task_index, state_index, replan, stream)
            request["command"] = None if command is None else list(command)
            chunk = np.asarray(client.infer(request)["actions"])
            plan.extend(chunk[:REPLAN])
            replan += 1
        if record:
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            frame_commands.append(commands[-1])
        disturbance.before_step(env, obs, step)
        action = plan.popleft().tolist()
        forced = disturbance.gripper(step)
        if forced is not None:
            action[6] = forced
        obs, _, done, _ = env.step(action)
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
    if stage2:
        s = L.state(env, obs, task)
        trace.append({"step": step + 1, "eef": s["eef"].round(4).tolist(), "grip": round(float(s["gripper"][0] - s["gripper"][1]), 4),
                      "target": s["target"].round(4).tolist(), "goal": s["goal"].round(4).tolist(),
                      "held": bool(s["held"]), "other_held": _other_held(env, task)})
        result.update(disturbance=disturbance.info, trace=trace)
    if saved:
        folder = frames_dir or grasp_frames
        path = pathlib.Path(folder) / f"task{task_index}_state{state_index}_{condition}_{kind}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"{name}/{k}": v for name, r in saved.items() for k, v in r.items()})
        result["frames_saved"] = sorted(saved)
    if record:
        result.update(frames=frames, frame_commands=frame_commands)
    return result


def _maybe_save(saved: dict, disturbance, replan: int, step: int, request: dict, s: dict):
    """Commitment test frames: the request plus the command toward the object (the scripted
    controller's rule applied to the target, whatever the condition)."""
    open_until = disturbance.info.get("open_until")
    if replan == EARLY_REPLAN:
        name = "early"
    elif open_until is not None and step >= open_until and sum(k.startswith("post") for k in saved) < POST_DROP_FRAMES:
        name = f"post{sum(k.startswith('post') for k in saved)}"
    else:
        return
    axis, sign = controller.command({**s, "held": False})
    saved[name] = {**{k: np.asarray(v) for k, v in request.items()}, "axis": np.int64(axis), "sign": np.int64(sign),
                   "step": np.int64(step), "held": np.bool_(s["held"])}


def _grasp_frames(saved: dict, recent, env, task, replan: int, step: int, request: dict, s: dict):
    """Stage 3 diagnostic frames. Keeps the last --grasp-history replans; when another bowl is
    held for the first time, saves them as wrong1 (the replan just before) and wrong2; when the
    target is held for the first time, as right1 and right2. Also saves replan EARLY_REPLAN as
    early. Each frame: the request, the command toward the target, and the positions."""
    others = {n: L.body_position(env, n) for n in L._inner(env).objects_dict if "bowl" in n and n != task.target}
    other_held = _other_held(env, task)
    for event, now in (("wrong", other_held is not None), ("right", s["held"])):
        if now and f"{event}1" not in saved:
            for i, frame in enumerate(reversed(recent)):
                saved[f"{event}{i + 1}"] = frame
    axis, sign = controller.command({**s, "held": False})
    frame = {**{k: np.asarray(v) for k, v in request.items()}, "axis": np.int64(axis), "sign": np.int64(sign),
             "step": np.int64(step), "replan": np.int64(replan), "eef": s["eef"], "target": s["target"],
             "others": np.array(list(others.values())), "held": np.bool_(s["held"]),
             "other_held": np.bool_(other_held is not None)}
    if replan == EARLY_REPLAN:
        saved.setdefault("early", frame)
    recent.append(frame)


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
    parser.add_argument("--disturbances", nargs="+", choices=D.KINDS,
                        help="Stage 2: run every episode under each of these (default: Stage 1, no disturbance field)")
    parser.add_argument("--frames", help="Stage 2: directory for the drop episodes' commitment-test frames")
    parser.add_argument("--grasp-frames", help="Stage 3: directory for frames before the first wrong and correct grasps")
    parser.add_argument("--grasp-history", type=int, default=2, help="Stage 3: replans saved before each grasp (wrong1 ... wrongN)")
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
            done.add((r["task"], r["state"], r["condition"], r["stream"]) + ((r["disturbance"]["kind"],) if "disturbance" in r else ()))
    jobs = [(t, s, c, args.stream) for t in args.tasks for s in parse_range(args.states) for c in args.conditions]
    if args.disturbances:
        jobs = [j + (k,) for k in args.disturbances for j in jobs]
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already in {out})", flush=True)

    started = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.host, args.port)) as pool:
        run = functools.partial(episode, frames_dir=args.frames, grasp_frames=args.grasp_frames,
                                grasp_history=args.grasp_history)
        for i, result in enumerate(pool.imap_unordered(run, jobs)):
            with out.open("a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[{i + 1}/{len(jobs)}, {(time.time() - started) / 60:.1f} min] {result.get('disturbance', {}).get('kind', '')} task {result['task']} state "
                  f"{result['state']} {result['condition']}: {'success' if result['success'] else 'failure'} "
                  f"in {result['steps']} steps", flush=True)


if __name__ == "__main__":
    main()
