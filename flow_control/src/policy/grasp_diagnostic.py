"""Stage 3 diagnostic: at the moment the policy closes on the wrong bowl, does it still follow
commands, and does its gripper output follow a gripper command? GPU, openpi env.

    sbatch --time=01:00:00 jobs/gpu_job.sh python src/policy/grasp_diagnostic.py \
        --frames /scratch/$USER/fc/stage3_diag/frames --out results/data/stage3_diagnostic.json

Frames (rollout.py --grasp-frames): the policy request at the 2 replans before the first wrong-bowl
grasp (wrong1 = the replan just before, wrong2), before the first correct grasp (right1, right2),
and an early approach frame (early, step 10). For each frame and each of SEEDS starting noises:

    translation, on the dominant axis from the end effector to the TARGET (the controller's rule):
        unsteered        mean of the first 5 actions on that axis, signed so + is toward the target
        on_axis          (+command) - (-command)
        toward           (command toward the target) - unsteered
    gripper (LIBERO: -1 open, +1 close), mean of the first 5 actions' gripper value:
        grip_free        unsteered
        grip_open        gripper dim of the noise written to open (-sigma) for the first tau steps
        grip_close       written to close (+sigma)
        closes_*         fraction of seeds whose first 5 actions contain a close (> 0)
        grip_response    grip_close - grip_open
    combined: toward-target command plus gripper open (the planned veto):
        veto_toward      its translation toward the target
        closes_veto      fraction of seeds that still close

Readout: if on_axis at wrong1 is far below early and right1, the policy is committed at the
wrong grasp and a stronger push (adaptive strength) is needed; if closes_open is near 0 at
wrong1 while closes_free is high, a gripper veto can stop the wrong grasp.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402
from server import CHECKPOINT, CONFIG_NAME  # noqa: E402

SEEDS = range(8)
REQUEST_KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")
FIRST = injection.TAU  # actions read per chunk, as executed between replans


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.training import config as _config

    policy = policy_config.create_trained_policy(_config.get_config(CONFIG_NAME), CHECKPOINT)
    rows = []
    files = sorted(pathlib.Path(args.frames).glob("task*.npz"))
    print(f"{len(files)} episode files", flush=True)
    for path in files:
        saved = np.load(path)
        task, state, condition, kind = path.stem.split("_", 3)
        for name in sorted({k.split("/")[0] for k in saved.files}):
            f = lambda key: saved[f"{name}/{key}"]  # noqa: E731
            obs = {k: (str(f(k)) if k == "prompt" else f(k)) for k in REQUEST_KEYS}
            axis, sign = int(f("axis")), int(f("sign"))

            def chunk(seed, command=None, grip=None):
                noise = injection.base_noise(seed)
                if command is not None:
                    noise = injection.inject(noise, *command)
                if grip is not None:
                    noise = injection.inject_gripper(noise, grip)
                return np.asarray(policy.infer(obs, noise=noise)["actions"])[:FIRST]

            out = {k: [] for k in ("free", "plus", "minus", "g_free", "g_open", "g_close", "c_free", "c_open",
                                   "c_close", "veto", "c_veto")}
            for seed in SEEDS:
                free, plus, minus = chunk(seed), chunk(seed, (axis, 1)), chunk(seed, (axis, -1))
                opened, closed = chunk(seed, grip=-1), chunk(seed, grip=1)
                veto = chunk(seed, (axis, sign), grip=-1)
                out["free"].append(free[:, axis].mean())
                out["plus"].append(plus[:, axis].mean())
                out["minus"].append(minus[:, axis].mean())
                for key, c in (("free", free), ("open", opened), ("close", closed), ("veto", veto)):
                    if key != "veto":
                        out[f"g_{key}"].append(c[:, 6].mean())
                    out[f"c_{key}"].append(float((c[:, 6] > 0).any()))
                out["veto"].append(veto[:, axis].mean())
            o = {k: np.array(v) for k, v in out.items()}
            toward = o["plus"] if sign > 0 else o["minus"]
            others = np.asarray(f("others"))
            eef, target = np.asarray(f("eef")), np.asarray(f("target"))
            rows.append({
                "task": int(task[4:]), "state": int(state[5:]), "condition": condition, "kind": kind, "frame": name,
                "step": int(f("step")), "axis": axis, "sign": sign,
                "eef_to_target_cm": float(100 * np.linalg.norm((target - eef)[:2])),
                "eef_to_nearest_other_cm": float(100 * np.min(np.linalg.norm((others - eef)[:, :2], axis=1))) if len(others) else None,
                "eef_height_cm": float(100 * (eef[2] - target[2])),
                "unsteered": float((sign * o["free"]).mean()),
                "on_axis": float((o["plus"] - o["minus"]).mean()),
                "toward": float((sign * (toward - o["free"])).mean()),
                "grip_free": float(o["g_free"].mean()), "grip_open": float(o["g_open"].mean()),
                "grip_close": float(o["g_close"].mean()), "grip_response": float((o["g_close"] - o["g_open"]).mean()),
                "closes_free": float(o["c_free"].mean()), "closes_open": float(o["c_open"].mean()),
                "closes_close": float(o["c_close"].mean()),
                "veto_toward": float((sign * o["veto"]).mean()), "closes_veto": float(o["c_veto"].mean()),
            })
        print(f"{path.name}: {len(rows)} frames so far", flush=True)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=0))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
