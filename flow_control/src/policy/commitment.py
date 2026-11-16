"""Stage 2 module 3: how committed is the policy after a drop? GPU, openpi env.

    sbatch --time=01:00:00 jobs/gpu_job.sh python src/policy/commitment.py \
        --frames /scratch/$USER/fc/stage2_frames --out results/data/stage2_commitment.json

The drop episodes of the Stage 2 rollouts (rollout.py --frames) saved the policy request at an
early replan ("early", step 10, the gripper approaching the object) and at the first two replans
after the forced opening ended ("post0", "post1", the object just fell). At each of these frames,
"toward the object" is the scripted controller's command with the target as the goal: the
dominant axis of the vector from the end effector to the object, and its sign.

For each frame and each of SEEDS starting noises, three chunks: unsteered, +command and
-command along that axis. Reported per frame (LIBERO action units, mean of the first 5 actions on
that axis, averaged over seeds):

    on_axis     (+command) - (-command)       the paper's on-axis response: how far a command
                                              moves the output at all
    toward      (toward command) - unsteered  how far steering toward the object moves it
    unsteered   the unsteered value, signed so that + is toward the object

A policy that is committed to carrying on after the drop shows a smaller on_axis response after
the drop than early in the episode, and an unsteered output that does not head to the object.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402
from server import CHECKPOINT, CONFIG_NAME, FlowControlPolicy  # noqa: E402

SEEDS = range(8)
REQUEST_KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.training import config as _config

    policy = FlowControlPolicy(policy_config.create_trained_policy(_config.get_config(CONFIG_NAME), CHECKPOINT))
    rows = []
    files = sorted(pathlib.Path(args.frames).glob("task*_state*_*.npz"))
    print(f"{len(files)} episode files", flush=True)
    for path in files:
        saved = np.load(path)
        task, state, condition = path.stem.split("_")
        for name in sorted({k.split("/")[0] for k in saved.files}):
            obs = {k: (str(saved[f"{name}/{k}"]) if k == "prompt" else saved[f"{name}/{k}"]) for k in REQUEST_KEYS}
            axis, sign = int(saved[f"{name}/axis"]), int(saved[f"{name}/sign"])

            def move(command, seed):
                actions = policy.infer({**obs, "noise_seed": seed, "command": command})["actions"]
                return float(np.asarray(actions)[: injection.TAU, axis].mean())

            free = np.array([move(None, s) for s in SEEDS])
            plus = np.array([move((axis, 1), s) for s in SEEDS])
            minus = np.array([move((axis, -1), s) for s in SEEDS])
            toward = plus if sign > 0 else minus
            rows.append({
                "task": int(task[4:]), "state": int(state[5:]), "condition": condition, "frame": name,
                "step": int(saved[f"{name}/step"]), "held": bool(saved[f"{name}/held"]), "axis": axis, "sign": sign,
                "on_axis": float((plus - minus).mean()), "on_axis_sd": float((plus - minus).std()),
                "toward": float((sign * (toward - free)).mean()), "unsteered": float((sign * free).mean()),
            })
        print(f"{path.name}: {len(rows)} frames so far", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
