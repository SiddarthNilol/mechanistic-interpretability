"""Module 2b: the pi05_libero policy server with Flow Control. GPU, openpi's environment.

    source jobs/env.sh && python src/policy/server.py --port 8000

openpi's own LIBERO server (scripts/serve_policy.py --env LIBERO) plus two extra request fields,
which the simulator side sends with every observation:

    noise_seed    int, required. The starting noise is base_noise(noise_seed), so every
                  condition gets the same noise at the same replan (pairing).
    command       None (unsteered) or [axis, sign]: Flow Control's injection (injection.py).

Stage 3 closed loop, optional (both use the stronger push, push.py):
    gripper       None, or -1 (open) / +1 (close): written into the gripper dim of the starting
                  noise and pushed for `gripper_m` Euler steps (default GRIPPER_M)
    adaptive      None, or a fidelity threshold: candidates with the translation push for
                  M in ADAPTIVE_MS, tried in order on the same noise; the first whose fidelity
                  (cosine between the command and the mean translation of the first 5 actions)
                  reaches the threshold is returned, else the one with the largest M. Trying them
                  in order returns the same chunk as sampling all in one batch and picking the
                  smallest passing M.

The reply is openpi's: "actions", the (10, 7) chunk in LIBERO's units, ready for env.step; with
a Stage 3 field also "push_m" (the M used for translation) and "fidelity".
The model, weights, transforms and 10 Euler steps are openpi's, unchanged (openpi's default
inference precision, bfloat16, as in the published LIBERO evaluation).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402
import push as P  # noqa: E402

CONFIG_NAME = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
ADAPTIVE_MS = (0, 2, 4, 6, 8)
GRIPPER_M = 6  # offline (Stage 3 push test): "open" stops all closings before a wrong grasp at M = 6


def fidelity(actions: np.ndarray, axis: int, sign: int) -> float:
    move = np.asarray(actions)[: injection.TAU, :3].mean(axis=0)
    return float(sign * move[axis] / (np.linalg.norm(move) + 1e-9))


class FlowControlPolicy:
    """Wraps an openpi Policy: picks the starting noise from the request, then calls it."""

    def __init__(self, policy):
        self._policy = policy
        self._pushed = None  # built on the first Stage 3 request (compiles the pushed sampler)

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        seed = obs.pop("noise_seed")
        command = obs.pop("command", None)
        gripper = obs.pop("gripper", None)
        gripper_m = int(obs.pop("gripper_m", GRIPPER_M))
        threshold = obs.pop("adaptive", None)
        noise = injection.base_noise(int(seed))
        if command is not None:
            axis, sign = (int(c) for c in command)
            noise = injection.inject(noise, axis, sign)
        if gripper is None and threshold is None:
            return self._policy.infer(obs, noise=noise)  # Stage 1 and 2 path, unchanged
        if self._pushed is None:
            self._pushed = P.PushedPolicy(self._policy)
        push = np.zeros_like(noise)
        m = np.zeros(injection.MODEL_ACTION_DIM, dtype=np.int32)
        if gripper is not None:
            noise = injection.inject_gripper(noise, int(gripper))
            push += P.push_array(injection.GRIPPER_DIM, int(gripper))
            m[injection.GRIPPER_DIM] = gripper_m
        candidates = ADAPTIVE_MS if (threshold is not None and command is not None) else (0,)
        for steps in candidates:
            p, mm = push.copy(), m.copy()
            if command is not None and steps:
                p += P.push_array(axis, sign)
                mm[axis] = steps
            actions, _ = self._pushed.infer(obs, noise, p, mm)
            fid = fidelity(actions, axis, sign) if command is not None else None
            if threshold is None or fid is None or fid >= float(threshold):
                break
        return {"actions": actions, "push_m": int(steps), "fidelity": fid}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    policy = policy_config.create_trained_policy(_config.get_config(CONFIG_NAME), CHECKPOINT)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=FlowControlPolicy(policy), host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logging.info("serving %s with Flow Control on port %d", CONFIG_NAME, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
