"""Module 2b: the pi05_libero policy server with Flow Control. GPU, openpi's environment.

    source jobs/env.sh && python src/policy/server.py --port 8000

openpi's own LIBERO server (scripts/serve_policy.py --env LIBERO) plus two extra request fields,
which the simulator side sends with every observation:

    noise_seed    int, required. The starting noise is base_noise(noise_seed), so every
                  condition gets the same noise at the same replan (pairing).
    command       None (unsteered) or [axis, sign]: Flow Control's injection (injection.py).

The reply is openpi's: "actions", the (10, 7) chunk in LIBERO's units, ready for env.step.
The model, weights, transforms and 10 Euler steps are openpi's, unchanged (openpi's default
inference precision, bfloat16, as in the published LIBERO evaluation).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import injection  # noqa: E402

CONFIG_NAME = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


class FlowControlPolicy:
    """Wraps an openpi Policy: picks the starting noise from the request, then calls it."""

    def __init__(self, policy):
        self._policy = policy

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        seed = obs.pop("noise_seed")
        command = obs.pop("command", None)
        noise = injection.base_noise(int(seed))
        if command is not None:
            axis, sign = (int(c) for c in command)
            noise = injection.inject(noise, axis, sign)
        return self._policy.infer(obs, noise=noise)


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
