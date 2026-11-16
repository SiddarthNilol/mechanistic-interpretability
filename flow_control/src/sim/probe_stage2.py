"""Stage 2 module 1: scene facts the new disturbances depend on. No policy. LIBERO env, GPU node.

    sbatch jobs/sim_job.sh python src/sim/probe_stage2.py

Over all 50 initial states of both tasks:
  1. layout: where the target bowl sits among the other bowls along x and y (which axis is
     "left" in task 60 and "at the back" in task 12), and the distance from the gripper's start
     to each bowl;
  2. distractor placement: put the nearest other bowl at target + d * direction (d in DISTANCES,
     direction +-x, +-y), let the scene settle, and record whether it stays put (no collision,
     no tipping) and whether the target's rank among the bowls along the instruction's axis is
     kept;
  3. moved goal: on states 0-9, move the goal object (tray or plate) by 6 cm in 8 directions,
     settle, and record how far it actually moved.

Writes results/probe/stage2_probe.json and prints a short report.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import libero_env as L  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "probe"
DISTANCES = (0.08, 0.10, 0.12)  # meters, center to center
DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
SETTLE = 20
_envs: dict = {}


def _env(task_index):
    if task_index not in _envs:
        _envs[task_index] = L.make_env(task_index)
    return _envs[task_index]


def bowls(env, task):
    return [n for n in L._inner(env).objects_dict if "bowl" in n and n != task.target]


def settle(env):
    for _ in range(SETTLE):
        env.step(L.SETTLE_ACTION)


def layout(job):
    task_index, state_index = job
    task = L.TASKS[task_index]
    env, _, initial = _env(task_index)
    obs = L.reset(env, initial[state_index])
    eef = np.array(obs["robot0_eef_pos"])
    names = [task.target] + bowls(env, task)
    pos = {n: L.body_position(env, n) for n in names}
    out = {
        "task": task_index, "state": state_index, "eef": eef.round(3).tolist(),
        "pos": {n: p.round(3).tolist() for n, p in pos.items()},
        "goal": L.body_position(env, task.goal).round(3).tolist(),
        "rank_x": int(sorted(names, key=lambda n: pos[n][0]).index(task.target)),
        "rank_y": int(sorted(names, key=lambda n: pos[n][1]).index(task.target)),
        "eef_dist": {n: float(np.linalg.norm((pos[n] - eef)[:2]).round(3)) for n in names},
        "placements": [],
    }
    nearest = min(bowls(env, task), key=lambda n: np.linalg.norm((pos[n] - pos[task.target])[:2]))
    for d in DISTANCES:
        for name, (ux, uy) in DIRECTIONS.items():
            L.reset(env, initial[state_index])
            before = {n: L.body_position(env, n) for n in L._inner(env).objects_dict}
            spot = before[task.target] + np.array([d * ux, d * uy, 0.0])
            spot[2] = before[nearest][2]
            L.move_object(env, nearest, spot - before[nearest])
            settle(env)
            after = {n: L.body_position(env, n) for n in before}
            others = [n for n in before if n != nearest]
            moved_others = max(float(np.linalg.norm(after[n] - before[n])) for n in others)
            pos2 = {n: after[n] for n in names}
            out["placements"].append({
                "d": d, "dir": name, "distractor": nearest,
                "distractor_error": float(np.linalg.norm(after[nearest] - spot).round(4)),
                "others_moved": round(moved_others, 4),
                "rank_x": int(sorted(names, key=lambda n: pos2[n][0]).index(task.target)),
                "rank_y": int(sorted(names, key=lambda n: pos2[n][1]).index(task.target)),
                "eef_dist_target": float(np.linalg.norm((after[task.target] - eef)[:2]).round(3)),
                "eef_dist_distractor": float(np.linalg.norm((after[nearest] - eef)[:2]).round(3)),
            })
    if state_index < 10:
        out["goal_moves"] = []
        for k in range(8):
            angle = 2 * np.pi * k / 8
            offset = 0.06 * np.array([np.cos(angle), np.sin(angle), 0.0])
            L.reset(env, initial[state_index])
            g0 = L.body_position(env, task.goal)
            L.move_object(env, task.goal, offset)
            settle(env)
            moved = L.body_position(env, task.goal) - g0
            out["goal_moves"].append({"angle_deg": 45 * k, "moved_cm": (100 * moved).round(2).tolist()})
    return out


def main():
    jobs = [(t, s) for t in sorted(L.TASKS) for s in range(50)]
    with mp.get_context("spawn").Pool(8) as pool:
        rows = pool.map(layout, jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_probe.json").write_text(json.dumps(rows))
    for t in sorted(L.TASKS):
        r = [x for x in rows if x["task"] == t]
        n_bowls = len(r[0]["pos"])
        print(f"\n== task {t}: target {L.TASKS[t].target}, {n_bowls} bowls, goal {L.TASKS[t].goal}")
        print("target rank along x (0 = smallest):", np.bincount([x["rank_x"] for x in r], minlength=n_bowls).tolist())
        print("target rank along y (0 = smallest):", np.bincount([x["rank_y"] for x in r], minlength=n_bowls).tolist())
        print("state 0 positions:", r[0]["pos"], "eef", r[0]["eef"], "goal", r[0]["goal"])
        closer = sum(min(v for k, v in x["eef_dist"].items() if k != L.TASKS[t].target) < x["eef_dist"][L.TASKS[t].target] for x in r)
        print(f"a non-target bowl is nearer the gripper's start than the target in {closer}/50 states")
        for d in DISTANCES:
            for name in DIRECTIONS:
                p = [pl for x in r for pl in x["placements"] if pl["d"] == d and pl["dir"] == name]
                clean = [pl for pl in p if pl["distractor_error"] < 0.01 and pl["others_moved"] < 0.01]
                kept_x = sum(pl["rank_x"] == x["rank_x"] for x in r for pl in x["placements"] if pl["d"] == d and pl["dir"] == name)
                kept_y = sum(pl["rank_y"] == x["rank_y"] for x in r for pl in x["placements"] if pl["d"] == d and pl["dir"] == name)
                nearer = sum(pl["eef_dist_distractor"] < pl["eef_dist_target"] for pl in p)
                print(f"  distractor at d {d:.2f} {name}: clean {len(clean)}/50, rank kept x {kept_x}/50 y {kept_y}/50, "
                      f"distractor nearer the gripper {nearer}/50, worst error {max(pl['distractor_error'] for pl in p)*100:.1f} cm")
        moves = [m["moved_cm"] for x in r for m in x.get("goal_moves", [])]
        print(f"goal moved 6 cm: actual displacement {np.round(np.linalg.norm(np.array(moves)[:, :2], axis=1).min(), 2)}"
              f" to {np.round(np.linalg.norm(np.array(moves)[:, :2], axis=1).max(), 2)} cm (horizontal), "
              f"max |dz| {np.abs(np.array(moves)[:, 2]).max():.2f} cm over {len(moves)} moves")


if __name__ == "__main__":
    main()
