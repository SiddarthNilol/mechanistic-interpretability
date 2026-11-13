"""Module 2 check: does the injected command move the predicted end effector? GPU, openpi env.

    source jobs/env.sh && python src/policy/check_server.py      # after src/sim/probe.py

For each task's saved observation (results/probe/task*_request.npz, written by the probe), and
for 8 noise seeds: one unsteered chunk, and one chunk per command (x, y, z, each sign). Prints
the mean of the first 5 actions' translation (dx, dy, dz, LIBERO units: scaled end-effector
deltas), averaged over seeds. Flow Control works if each commanded axis moves in the commanded
sign, clearly more than the unsteered chunk does. This is the paper's "on-axis response" on one
frame, not a success-rate result.

Calls the same FlowControlPolicy the server uses, without the network.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402
from server import CHECKPOINT, CONFIG_NAME, FlowControlPolicy  # noqa: E402

PROBE = pathlib.Path(__file__).resolve().parents[2] / "results" / "probe"
SEEDS = range(8)


def main():
    from openpi.policies import policy_config
    from openpi.training import config as _config

    policy = FlowControlPolicy(policy_config.create_trained_policy(_config.get_config(CONFIG_NAME), CHECKPOINT))
    for path in sorted(PROBE.glob("task*_request.npz")):
        saved = np.load(path)
        obs = {k: (str(saved[k]) if k == "prompt" else saved[k]) for k in saved.files}
        print(f"\n{path.name}: {obs['prompt']!r}")
        commands = [None] + [(axis, sign) for axis in range(3) for sign in (1, -1)]
        for command in commands:
            moves = []
            for seed in SEEDS:
                actions = policy.infer({**obs, "noise_seed": seed, "command": command})["actions"]
                moves.append(np.asarray(actions)[: injection.TAU, :3].mean(axis=0))
            mean, sd = np.mean(moves, axis=0), np.std(moves, axis=0)
            name = "unsteered" if command is None else f"{'+' if command[1] > 0 else '-'}{'xyz'[command[0]]}"
            print(f"  {name:>9}: dx {mean[0]:+.3f}  dy {mean[1]:+.3f}  dz {mean[2]:+.3f}   "
                  f"(s.d. over {len(SEEDS)} seeds {sd[0]:.3f} {sd[1]:.3f} {sd[2]:.3f})")


if __name__ == "__main__":
    main()
