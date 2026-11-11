"""Checks pair.py against openpi, and the patching sanity check of CLAUDE.md.

    python src/check_pair.py           # two small random models, on a CPU in about 2 minutes
    python src/check_pair.py --real    # the real pair on real states (GPU); every job runs this first

1. Our forward pieces equal openpi's: the KV cache of run_prefix equals the one from openpi's own
   backbone, and our action chunk equals model.sample_actions with the same noise.
2. Sanity check: patching a model's own activations into itself changes nothing. Tested for the
   residual at every layer and for the KV cache at every layer. Required: under 1e-5.
3. The patches do something: the other model's activations change the actions, except the
   residual after the last layer, which nothing reads.
4. The steering vector: adding zero changes nothing, adding a vector changes the actions.

--real appends the measured differences to results/run_log.md.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
from flax import traverse_util
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask

import pair as P
from common import REAL_DIMS, SEED, log_run

TOLERANCE = 1e-5  # CLAUDE.md: self-patching must reproduce the actions to this, in float32


def dummy_models(max_token_len: int = 20):
    """Two small random pi0.5 models in pi05_libero's format: (config, pair)."""
    config = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32",
        max_token_len=max_token_len, action_horizon=10, discrete_state_input=False,
    )

    def random_model(seed: int):
        # A freshly initialized pi0.5 ignores its prefix: the action expert's norm layers start at
        # zero, which closes the gate on every layer's output, and so does the vision encoder's
        # output layer ("head"). So perturb those weights.
        params = nnx.split(config.create(jax.random.key(seed)))[1].to_pure_dict()
        flat = traverse_util.flatten_dict(params)
        keys = jax.random.split(jax.random.key(seed + 100), len(flat))
        for key, (path, value) in zip(keys, flat.items()):
            if any("norm" in part or part == "head" for part in path):
                flat[path] = value + 0.3 * jax.random.normal(key, value.shape)
        return config.load(traverse_util.unflatten_dict(flat))

    return config, P.Pair({"base": random_model(1), "ft": random_model(2)})


def dummy_pair() -> tuple[P.Pair, _model.Observation, jax.Array]:
    """The small random pair, and random inputs with prompt padding and a masked third camera
    like the LIBERO inputs."""
    config, pair = dummy_models()
    keys = iter(jax.random.split(jax.random.key(3), 8))
    b, cameras = 2, ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = _model.Observation(
        images={c: jax.random.uniform(next(keys), (b, 224, 224, 3), minval=-1, maxval=1) for c in cameras},
        image_masks={c: jnp.full(b, c != "right_wrist_0_rgb") for c in cameras},
        state=jnp.zeros((b, config.action_dim)),
        tokenized_prompt=jax.random.randint(next(keys), (b, 20), 1, 1000),
        tokenized_prompt_mask=jnp.arange(20)[None] < jnp.array([12, 7])[:, None],
    )
    return pair, obs, jax.random.normal(next(keys), (b, config.action_horizon, config.action_dim))


def run_checks(pair: P.Pair, obs: _model.Observation, noise: jax.Array, agreement_tolerance: float) -> dict:
    depth = pair.depth

    def reference(models, obs, noise):
        """openpi's own computation, per model: its KV cache and its sampled actions."""
        out = {}
        for name, m in models.items():
            o = P.preprocess(obs)
            tokens, mask, ar = m.embed_prefix(o)
            _, kv = m.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
            ours, _ = P.run_prefix(m, tokens, mask)
            out[name] = (kv, ours, m.sample_actions(jax.random.key(0), obs, noise=noise), P.sample(m, obs, noise))
        return out

    def patched(models, obs, noise, r, source, layer_kv, vector):
        """Base model actions with one intervention. `source` picks whose activations go in."""
        obs = P.preprocess(obs)
        acts = {}
        for name, m in models.items():
            tokens, mask = P.embed(m, obs)
            acts[name] = (tokens, mask, *P.run_prefix(m, tokens, mask))
        base = models["base"]
        tokens, mask, kv, _ = acts["base"]
        other_kv, other_res = jax.tree.map(lambda a, b: jnp.where(source == 0, a, b), acts["base"][2:], acts["ft"][2:])
        res_kv, _ = P.run_prefix(base, tokens, mask, patch_at=r, patch_x=other_res[r])
        one_kv = jax.tree.map(lambda own, other: own.at[layer_kv].set(other[layer_kv]), kv, other_kv)
        add_kv, _ = P.run_prefix(base, tokens, mask, add_at=r, add_vec=vector)
        return tuple(P.denoise(base, obs, mask, c, noise)[..., :REAL_DIMS] for c in (kv, res_kv, one_kv, add_kv))

    report = {}
    agree_kv, agree_actions = 0.0, 0.0
    for name, (kv, ours, actions, ours_actions) in pair.jit(reference)(obs, noise).items():
        agree_kv = max(agree_kv, *(float(jnp.abs(a - b).max()) for a, b in zip(kv, ours)))
        agree_actions = max(agree_actions, float(jnp.abs(actions - ours_actions).max()))
    report["ours vs openpi, KV cache"] = agree_kv
    report["ours vs openpi, action chunk"] = agree_actions
    assert agree_actions < agreement_tolerance, f"our actions differ from openpi's by {agree_actions:.2e}"

    run = pair.jit(patched)
    width = int(pair.states["base"].to_pure_dict()["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].shape[-1])
    zero, vector = jnp.zeros(width), jax.random.normal(jax.random.key(SEED), (width,))
    self_res = self_kv = add_zero = 0.0
    effect_res, effect_kv, effect_add = [], [], []
    for r in range(depth + 1):
        layer = min(r, depth - 1)
        clean, res, one, add = run(obs, noise, r, 0, layer, zero)  # own activations, zero vector
        self_res = max(self_res, float(jnp.abs(res - clean).max()))
        self_kv = max(self_kv, float(jnp.abs(one - clean).max()))
        add_zero = max(add_zero, float(jnp.abs(add - clean).max()))
        clean, res, one, add = run(obs, noise, r, 1, layer, vector)  # the other model's, a real vector
        effect_res.append(float(jnp.abs(res - clean).max()))
        effect_kv.append(float(jnp.abs(one - clean).max()))
        effect_add.append(float(jnp.abs(add - clean).max()))
    report["self-patch, residual, worst layer"] = self_res
    report["self-patch, KV, worst layer"] = self_kv
    report["adding a zero vector, worst layer"] = add_zero
    assert max(self_res, self_kv, add_zero) < TOLERANCE, f"self-patching changed the actions: {report}"
    assert min(effect_res[:depth]) > 0 and min(effect_kv[:depth]) > 0, "a cross-model patch had no effect"
    assert min(effect_add[:depth]) > 0, "a steering vector had no effect"
    assert effect_res[depth] == 0 and effect_add[depth] == 0, "the residual after the last layer should be unread"
    report["cross-model patch, smallest effect over layers"] = min(effect_res[:depth] + effect_kv[:depth])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="check the real pair on real states (needs a GPU)")
    parser.add_argument("--states", type=int, default=4, help="with --real: number of states to check on")
    args = parser.parse_args()
    started = time.time()

    if args.real:
        import states as S

        pair = P.Pair.load()
        inputs = S.Inputs(S.load_states(patch_only=True)[: args.states])
        _, obs, noise = next(inputs.batches(args.states))
        # Two float32 routes to the same number (openpi's scan and loop, ours) agree to rounding.
        report = run_checks(pair, obs, noise, agreement_tolerance=1e-3)
    else:
        pair, obs, noise = dummy_pair()
        report = run_checks(pair, obs, noise, agreement_tolerance=1e-4)

    for name, value in report.items():
        print(f"{name}: {value:.2e}", flush=True)
    print(f"ok: self-patching reproduces the actions (tolerance {TOLERANCE:g}, float32)")
    if args.real:
        log_run("check_pair (sanity check)", started, seed=SEED, states=args.states,
                **{k: f"{v:.2e} (max abs difference, float32)" for k, v in report.items()})


if __name__ == "__main__":
    main()
