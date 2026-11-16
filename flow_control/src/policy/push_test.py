"""Stage 3: does a stronger push (M > 0) move translation and the gripper, and at what cost in
conformity? Offline, on saved frames. GPU, openpi env.

    sbatch --time=02:00:00 jobs/gpu_job.sh python src/policy/push_test.py \
        --frames /scratch/$USER/fc/stage3_push/frames --out results/data/stage3_push.json

Frames: rollout.py --grasp-frames with --grasp-history 6 (task 60, none and distractor): early
(step 10), wrong1 ... wrong6 and right1 ... right6 (the replans before the first wrong and correct
grasp; wrong1 is the last one). Used groups: GROUPS, at most PER_GROUP frames each.

Check first: our sampler with M = 0 against openpi's own Policy.infer on the same noise.

For each frame, seed in SEEDS and M in MS (starting-noise command always on, as in plain Flow
Control, plus the push after the first M Euler steps):
    translation, on the dominant axis toward the target:
        on_axis       (+command) - (-command), mean of the first 5 actions on that axis
        toward        command toward the target: signed motion toward the target
        fidelity      cosine between the command direction and the mean translation of the
                      first 5 actions (CLAUDE.md Stage 3)
        conformity    flow-matching loss of the toward chunk (lower = more like the policy's own)
    gripper (the gripper dim written into the noise and pushed):
        closes_open, closes_close   fraction of chunks whose first 5 actions close (> 0)
        grip_response               mean gripper value, close - open
        conformity_open             loss of the open chunk
and once per frame and seed, unsteered: closes_free, conformity_free.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402
import push as P  # noqa: E402
from server import CHECKPOINT, CONFIG_NAME  # noqa: E402

SEEDS = range(4)
MS = (0, 2, 4, 6, 8)
GROUPS = ("early", "wrong5", "wrong3", "wrong1", "right3", "right1")
PER_GROUP = 40
REQUEST_KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")
FIRST = injection.TAU


def load_frames(folder: pathlib.Path) -> list[dict]:
    frames = {g: [] for g in GROUPS}
    for path in sorted(folder.glob("task*.npz")):
        saved = np.load(path)
        task, state, condition, kind = path.stem.split("_", 3)
        for g in GROUPS:
            if f"{g}/axis" in saved.files and len(frames[g]) < PER_GROUP:
                f = {k: saved[f"{g}/{k}"] for k in REQUEST_KEYS + ("axis", "sign", "step")}
                f["prompt"] = str(f["prompt"])
                frames[g].append({"task": int(task[4:]), "state": int(state[5:]), "condition": condition,
                                  "kind": kind, "frame": g, **f})
    return [f for g in GROUPS for f in frames[g]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, help="test run: only this many frames")
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.training import config as _config

    base = policy_config.create_trained_policy(_config.get_config(CONFIG_NAME), CHECKPOINT)
    pushed = P.PushedPolicy(base)
    frames = load_frames(pathlib.Path(args.frames))[: args.limit]
    print(f"{len(frames)} frames: " + ", ".join(f"{g} {sum(f['frame'] == g for f in frames)}" for g in GROUPS), flush=True)

    obs0 = {k: frames[0][k] for k in REQUEST_KEYS}
    noise0 = injection.base_noise(0)
    ours = pushed.infer(obs0, noise0)[0]
    theirs = np.asarray(base.infer(obs0, noise=noise0)["actions"])
    print(f"check, M = 0 vs openpi Policy.infer: max |difference| {np.abs(ours - theirs).max():.2e} "
          f"(actions range {np.abs(theirs).max():.2f})", flush=True)

    rows, started = [], time.time()
    for n, fr in enumerate(frames):
        obs = {k: fr[k] for k in REQUEST_KEYS}
        axis, sign = int(fr["axis"]), int(fr["sign"])
        free = {"closes": [], "conf": []}
        per_m = {m: {k: [] for k in ("plus", "minus", "toward", "fidelity", "conf", "c_open", "c_close", "g_open",
                                     "g_close", "conf_open")} for m in MS}
        for seed in SEEDS:
            noise = injection.base_noise(seed)
            a, chunk = pushed.infer(obs, noise)
            free["closes"].append(float((a[:FIRST, 6] > 0).any()))
            free["conf"].append(pushed.conformity(obs, chunk))
            for m in MS:
                r = per_m[m]
                for s_ in (1, -1):
                    a, chunk = pushed.infer(obs, injection.inject(noise, axis, s_), P.push_array(axis, s_), m)
                    r["plus" if s_ > 0 else "minus"].append(a[:FIRST, axis].mean())
                    if s_ == sign:
                        move = a[:FIRST, :3].mean(axis=0)
                        r["toward"].append(sign * move[axis])
                        r["fidelity"].append(float(sign * move[axis] / (np.linalg.norm(move) + 1e-9)))
                        r["conf"].append(pushed.conformity(obs, chunk))
                for g in (-1, 1):
                    a, chunk = pushed.infer(obs, injection.inject_gripper(noise, g), P.push_array(injection.GRIPPER_DIM, g), m)
                    key = "open" if g < 0 else "close"
                    r[f"c_{key}"].append(float((a[:FIRST, 6] > 0).any()))
                    r[f"g_{key}"].append(a[:FIRST, 6].mean())
                    if g < 0:
                        r["conf_open"].append(pushed.conformity(obs, chunk))
        for m in MS:
            r = {k: np.array(v) for k, v in per_m[m].items()}
            rows.append({
                "task": fr["task"], "state": fr["state"], "condition": fr["condition"], "kind": fr["kind"],
                "frame": fr["frame"], "step": int(fr["step"]), "axis": axis, "sign": sign, "m": m,
                "on_axis": float((r["plus"] - r["minus"]).mean()), "toward": float(r["toward"].mean()),
                "fidelity": float(r["fidelity"].mean()), "conformity": float(r["conf"].mean()),
                "closes_open": float(r["c_open"].mean()), "closes_close": float(r["c_close"].mean()),
                "grip_response": float((r["g_close"] - r["g_open"]).mean()),
                "conformity_open": float(r["conf_open"].mean()),
                "closes_free": float(np.mean(free["closes"])), "conformity_free": float(np.mean(free["conf"])),
            })
        if n % 20 == 0:
            print(f"[{n + 1}/{len(frames)}, {(time.time() - started) / 60:.1f} min]", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
