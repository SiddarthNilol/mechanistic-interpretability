"""F8 (optional), part 1 of 3: a policy server for pi05_libero that can be steered. One GPU.

    python src/f8_steer_server.py --layer 9 --port 8000

NOT RUN YET: written from openpi's serving code and examples/libero, but never tested against
the LIBERO simulator. Do a short smoke test (one episode per condition) before the real run.

Serves pi05_libero like openpi's scripts/serve_policy.py, except that a request may ask for a
vector to be added to every prefix token's residual after --layer backbone layers:

    obs["steer/direction"]   "none", "progress", "progress_readable" or "random"
    obs["steer/alpha"]       strength, in units of the mean norm of one token's residual at
                             that layer (from F5), so alpha = 1 adds a vector as long as a
                             typical token
    obs["steer/episode"], obs["steer/step"]   seed the flow noise, so every condition gets the
                             same noise at the same step of the same episode

Directions, all scaled to the same norm:

    progress            the progress probe's weight vector at that layer (F5)
    progress_readable   its projection on the action-readable subspace (top k directions, F6)
    random              a Gaussian direction (seed 0)

Needs results/data/f5_directions.npz and f6_subspaces.npz, so --layer must be one of F6's.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.serving import websocket_policy_server
from openpi_client import base_policy

import pair as P
import states as S
from common import DATA, SEED


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def steering_vectors(layer: int) -> dict[str, np.ndarray]:
    directions = np.load(DATA / "f5_directions.npz")
    subspaces = np.load(DATA / "f6_subspaces.npz")
    k = json.loads((DATA / "f6.json").read_text())["layers"][str(layer)]["k90"]
    progress = directions["progress_ft"][layer]
    basis = subspaces[f"U_{layer}"][:, :k].astype(np.float64)
    scale = float(directions["token_norm_ft"][layer])
    vectors = {
        "none": np.zeros_like(progress),
        "progress": scale * unit(progress),
        "progress_readable": scale * unit(basis @ (basis.T @ progress)),
        "random": scale * unit(np.random.default_rng(SEED).standard_normal(len(progress))),
    }
    return {name: v.astype(np.float32) for name, v in vectors.items()}


class SteeredPolicy(base_policy.BasePolicy):
    def __init__(self, layer: int):
        self.pair = P.Pair.load(names=("ft",))
        self.input, self.output = S.transforms()
        self.vectors = steering_vectors(layer)
        config = P.model_config()
        self.action_shape = (1, config.action_horizon, config.action_dim)
        self.key = jax.random.key(SEED)

        def act(models, obs, noise, vector):
            model = models["ft"]
            obs = P.preprocess(obs)
            tokens, mask = P.embed(model, obs)
            kv, _ = P.run_prefix(model, tokens, mask, add_at=layer, add_vec=vector)
            return P.denoise(model, obs, mask, kv, noise)

        self.act = self.pair.jit(act)

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        direction = str(obs.pop("steer/direction", "none"))
        alpha = float(obs.pop("steer/alpha", 0.0))
        episode, step = int(obs.pop("steer/episode", 0)), int(obs.pop("steer/step", 0))
        inputs = self.input(obs)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        noise = jax.random.normal(jax.random.fold_in(jax.random.fold_in(self.key, episode), step), self.action_shape)
        actions = self.act(_model.Observation.from_dict(batch), noise, jnp.asarray(alpha * self.vectors[direction]))
        return self.output({"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0])})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=9, help="steer the residual after this many layers")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = SteeredPolicy(args.layer)
    print(f"serving steered pi05_libero (layer {args.layer}) on port {args.port}", flush=True)
    websocket_policy_server.WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
