"""The model pair (pi05_base, pi05_libero) and the forward pieces every pilot is built from.

pi0.5 computes an action chunk in two passes (openpi, src/openpi/models/pi0.py, sample_actions):

    prefix pass   SigLIP turns each camera image into 256 tokens, the text prompt is embedded, and
                  the 18 backbone layers (PaliGemma's Gemma 2B) run over these 968 prefix tokens.
                  Each layer's attention keys and values are kept: the "KV cache".
    suffix pass   the action expert (Gemma 300M) runs over the noisy action tokens. In every
                  layer its attention also reads that layer's prefix keys and values. This is
                  the only way the backbone reaches the actions. It is repeated for each of the
                  10 flow steps, from noise (time 1) to the action chunk (time 0).

So a patch is a choice of what the action expert reads, and of whose weights do the reading:

    run_prefix    our own copy of the backbone's layer loop, so that one layer's residual can be
                  replaced (patching) or shifted (steering, Jacobians). openpi hides the loop in a
                  scan. check_pair.py tests the copy against openpi.
    velocity      one suffix pass through openpi's own modules, given any KV cache
    denoise       the 10 flow steps, as openpi's sample_actions does them

Layer numbering used everywhere: "residual after r layers", r = 0 (the token embeddings) to 18,
and "KV at layer l", l = 1 to 18. Layer l reads the residual after l - 1 layers, so the residual
after all 18 layers is read by nothing: the action expert only sees keys and values.

Everything runs in float32 (openpi's default is bfloat16), so patches can be checked exactly.
"""

from __future__ import annotations

import dataclasses

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import gemma
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

from common import NUM_STEPS

# Full float32 matrix products. On GPUs JAX otherwise uses TF32 (about 3 significant digits).
jax.config.update("jax_default_matmul_precision", "highest")

CONFIG_NAME = "pi05_libero"  # openpi train config: the input format and model shape of both models
CHECKPOINTS = {
    "base": "gs://openpi-assets/checkpoints/pi05_base",
    "ft": "gs://openpi-assets/checkpoints/pi05_libero",
}


def model_config():
    """pi05_libero's model config in float32. The base model is run in this format too (10 action
    tokens, state not in the prompt), so that both models get exactly the same input."""
    from openpi.training import config as _config

    return dataclasses.replace(_config.get_config(CONFIG_NAME).model, dtype="float32")


def load_params(name: str, *, on_cpu: bool = False) -> dict:
    """A checkpoint's weights as a nested dict of float32 arrays (downloaded on first use)."""
    import numpy as np
    from openpi.shared import download

    path = download.maybe_download(CHECKPOINTS[name] + "/params")
    return _model.restore_params(path, restore_type=np.ndarray if on_cpu else jax.Array, dtype=jnp.float32)


class Pair:
    """Both models side by side. `pair.jit(fn)` compiles fn(models, *args), where models is
    {"base": model, "ft": model}, into a function of *args only."""

    def __init__(self, models: dict):
        first = next(iter(models))
        self.graphdef = nnx.split(models[first])[0]  # the same for both: one architecture
        self.states = {name: nnx.split(model)[1] for name, model in models.items()}
        self.depth = self.states[first].to_pure_dict()["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].shape[0]

    @classmethod
    def load(cls, names=("base", "ft")) -> "Pair":
        config = model_config()
        models = {}
        for name in names:
            print(f"loading {name}: {CHECKPOINTS[name]}", flush=True)
            models[name] = config.load(load_params(name))
        return cls(models)

    def jit(self, fn):
        def run(states, *args):
            return fn({name: nnx.merge(self.graphdef, state) for name, state in states.items()}, *args)

        compiled = jax.jit(run)
        return lambda *args: compiled(self.states, *args)


# -- prefix pass ---------------------------------------------------------------------


def preprocess(obs: _model.Observation) -> _model.Observation:
    return _model.preprocess_observation(None, obs, train=False)


def embed(model, obs) -> tuple[jax.Array, jax.Array]:
    """Prefix tokens (b, 968, width) and their mask (b, 968). The first n_image_tokens(obs) are
    the cameras' SigLIP tokens, the rest the prompt's embeddings. False in the mask: prompt
    padding and the unused third camera."""
    tokens, mask, _ = model.embed_prefix(obs)
    return tokens, mask


def n_image_tokens(obs, tokens) -> int:
    return tokens.shape[1] - obs.tokenized_prompt.shape[1]


def backbone_layers(model) -> dict:
    """The backbone's per-layer weights, each with a leading layer axis. In openpi's parameter
    tree the action expert's have the suffix _1; these are the ones without."""
    layers = nnx.split(model)[1].to_pure_dict()["PaliGemma"]["llm"]["layers"]
    return {
        "q": layers["attn"]["q_einsum"]["w"],
        "kv": layers["attn"]["kv_einsum"]["w"],
        "out": layers["attn"]["attn_vec_einsum"]["w"],
        "attn_norm": layers["pre_attention_norm"]["scale"],
        "mlp_norm": layers["pre_ffw_norm"]["scale"],
        "mlp_gating": layers["mlp"]["gating_einsum"],
        "mlp_linear": layers["mlp"]["linear"],
    }


def _rms_norm(x, scale):
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jnp.reciprocal(jnp.sqrt(var + 1e-06)) * (1 + scale)).astype(x.dtype)


def _backbone_block(p: dict, x, positions, attn_mask):
    """One backbone layer on the prefix alone: openpi's gemma.Block with only the first expert.
    Returns the new residual and the layer's keys and values (b, tokens, 1, head_dim)."""
    h = _rms_norm(x, p["attn_norm"])
    q = jnp.einsum("BTD,NDH->BTNH", h, p["q"])
    k, v = jnp.einsum("BSD,2KDH->2BSKH", h, p["kv"])
    q = gemma._apply_rope(q, positions=positions) * p["q"].shape[-1] ** -0.5  # noqa: SLF001
    k = gemma._apply_rope(k, positions=positions)  # noqa: SLF001
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=1)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
    logits = jnp.where(attn_mask[:, None, None, :, :], logits, -2.3819763e38)
    probs = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
    heads = einops.rearrange(jnp.einsum("BKGTS,BSKH->BTKGH", probs, v), "B T K G H -> B T (K G) H")
    x = x + jnp.einsum("BTNH,NHD->BTD", heads, p["out"])

    h = _rms_norm(x, p["mlp_norm"])
    x = x + jnp.dot(nn.gelu(jnp.dot(h, p["mlp_gating"][0])) * jnp.dot(h, p["mlp_gating"][1]), p["mlp_linear"])
    return x, (k, v)


def run_prefix(model, tokens, mask, *, patch_at=-1, patch_x=None, add_at=-1, add_vec=None):
    """The backbone over the prefix tokens, with two optional interventions on the residual
    after r layers (r = 0 is the embeddings):

        patch_at=r, patch_x=x    replace it with x (b, tokens, width), e.g. the other model's
        add_at=r, add_vec=v      add v to every token: v is (width,) or (b, 1, width)

    r may be a traced integer, so one compiled function serves every layer. Returns the KV cache
    (keys and values, each (18, b, tokens, 1, head_dim)), in the form openpi's suffix pass takes,
    and the residuals (19, b, tokens, width): residual[r] is the one after r layers, with the
    intervention at r already applied."""
    layers = backbone_layers(model)
    depth = layers["q"].shape[0]
    positions = jnp.cumsum(mask, axis=1) - 1
    attn_mask = make_attn_mask(mask, jnp.zeros(mask.shape[1], dtype=bool))  # every valid token sees every other

    def layer(x, inputs):
        r, p = inputs
        if patch_x is not None:
            x = jnp.where(r == patch_at, patch_x, x)
        if add_vec is not None:
            x = x + (r == add_at) * add_vec
        y, kv = _backbone_block(p, x, positions, attn_mask)
        return y, (kv, x)

    final, (kv, residual) = jax.lax.scan(layer, tokens, (jnp.arange(depth), layers))
    return kv, jnp.concatenate([residual, final[None]])


def mean_pool(residual, mask):
    """Mean over the valid prefix tokens: (..., b, tokens, width) -> (..., b, width)."""
    weights = mask[..., None].astype(residual.dtype)
    return (residual * weights).sum(-2) / weights.sum(-2)


# -- suffix pass ---------------------------------------------------------------------


def velocity(model, obs, prefix_mask, kv, x_t, time):
    """The action expert's predicted velocity (b, horizon, 32) for noisy actions x_t at flow time
    `time` (1 = noise, 0 = actions), reading the prefix through `kv`. The body of the step function
    in openpi's sample_actions."""
    b = x_t.shape[0]
    suffix, suffix_mask, suffix_ar, adarms_cond = model.embed_suffix(obs, x_t, jnp.broadcast_to(time, (b,)))
    attn_mask = jnp.concatenate(
        [einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]), make_attn_mask(suffix_mask, suffix_ar)],
        axis=-1,
    )
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix], mask=attn_mask, positions=positions, kv_cache=kv, adarms_cond=[None, adarms_cond]
    )
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def trajectory(model, obs, prefix_mask, kv, noise, num_steps: int = NUM_STEPS):
    """Euler integration of the flow from `noise`, as in openpi's sample_actions. Returns the
    action chunk (b, horizon, 32), and the noisy actions (steps, b, horizon, 32) and flow time
    (steps,) at the start of every step."""
    dt = -1.0 / num_steps

    def step(carry, _):
        x_t, time = carry
        return (x_t + dt * velocity(model, obs, prefix_mask, kv, x_t, time), time + dt), (x_t, time)

    (actions, _), (xs, times) = jax.lax.scan(step, (noise, jnp.float32(1.0)), None, length=num_steps)
    return actions, xs, times


def denoise(model, obs, prefix_mask, kv, noise, num_steps: int = NUM_STEPS):
    """The action chunk (b, horizon, 32) from `noise`, reading the prefix through `kv`."""
    return trajectory(model, obs, prefix_mask, kv, noise, num_steps)[0]


def sample(model, obs, noise):
    """A model's unpatched action chunk, through the pieces above."""
    obs = preprocess(obs)
    tokens, mask = embed(model, obs)
    kv, _ = run_prefix(model, tokens, mask)
    return denoise(model, obs, mask, kv, noise)
