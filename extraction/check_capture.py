"""Checks that extract_heads.py records the same head outputs openpi computes.

    python extraction/check_capture.py

Builds a small randomly initialized pi0.5 (openpi's "dummy" Gemma size, so it runs on a
CPU in about a minute) and runs extract_heads.make_forward on random inputs. Alongside, it
records the input of every layer's output projection (attn_vec_einsum): openpi's own
per-head outputs, softmax(q k^T / sqrt(d)) v. Both must agree at the recorded tokens, in
every layer of both passes.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models import pi0_config

from extract_heads import make_denoise, make_forward

BATCH, PROMPT_LEN = 2, 20
REAL_TOKENS = [12, 7]  # prompt tokens per sample, the rest is padding


def main():
    # Full float32 matrix products. On GPUs, JAX by default multiplies float32 matrices with
    # TF32 (~3 significant digits), which alone makes the two computations differ by ~1e-4.
    jax.config.update("jax_default_matmul_precision", "highest")
    config = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32", max_token_len=PROMPT_LEN
    )
    model = config.create(jax.random.key(0))
    keys = iter(jax.random.split(jax.random.key(1), 8))
    obs = _model.Observation(
        images={
            slot: jax.random.uniform(next(keys), (BATCH, 224, 224, 3), minval=-1, maxval=1)
            for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        image_masks={slot: jnp.ones(BATCH, dtype=bool) for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        state=jnp.zeros((BATCH, config.action_dim)),
        tokenized_prompt=jax.random.randint(next(keys), (BATCH, PROMPT_LEN), 1, 1000),
        tokenized_prompt_mask=jnp.arange(PROMPT_LEN)[None] < jnp.array(REAL_TOKENS)[:, None],
    )
    noise = jax.random.normal(next(keys), (BATCH, config.action_horizon, config.action_dim))

    # openpi's per-head outputs: the input of the output projection, for all tokens
    reference = []

    def interceptor(next_fun, args, kwargs, context):
        if context.method_name == "__call__" and (context.module.name or "").startswith("attn_vec_einsum"):
            jax.debug.callback(lambda x: reference.append(np.asarray(x)), args[1], ordered=True)
        return next_fun(*args, **kwargs)

    sink = []
    forward = make_forward(model, sink)
    actions = make_denoise(model)(obs, noise)
    last_prompt_token = 3 * 256 + np.array(REAL_TOKENS) - 1  # 256 tokens per image
    worst = 0.0
    action_heads = {}
    # Both inputs of the suffix pass: noise at flow time 1, and --denoised's clean actions at 0.
    for name, suffix_input, flow_time in (("noise", noise, 1.0), ("denoised", actions, 0.0)):
        sink.clear()
        reference.clear()
        with nn.intercept_methods(interceptor):  # in effect while forward is traced
            forward(obs, suffix_input, flow_time)
        jax.effects_barrier()

        depth = len(sink) // 2
        assert len(sink) == len(reference) == 2 * depth, (len(sink), len(reference))
        for layer in range(2 * depth):
            query = last_prompt_token if layer < depth else np.zeros(BATCH, dtype=int)
            expected = reference[layer][np.arange(BATCH), query]  # (b, heads, head_dim)
            heads, weights = sink[layer]
            np.testing.assert_allclose(heads, expected, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(weights.sum(-1), 1.0, rtol=1e-5)
            if layer < depth:  # the prompt token attends to no padding
                for b, real in enumerate(REAL_TOKENS):
                    assert np.all(weights[b, :, 3 * 256 + real :] == 0)
            worst = max(worst, float(np.abs(heads - expected).max()))
        action_heads[name] = np.stack([h for h, _ in sink[depth:]])
    assert not np.allclose(action_heads["noise"], action_heads["denoised"]), "the suffix input had no effect"
    print(f"ok: {2 * depth} layers match openpi with both suffix inputs (largest difference {worst:.1e})")


if __name__ == "__main__":
    main()
