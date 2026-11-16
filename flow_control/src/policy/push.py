"""Stage 3: Flow Control's stronger push (paper Appendix D) for pi05_libero. GPU, openpi env.

Plain Flow Control (M = 0) only writes the command into the starting noise. The stronger push
also shifts the sample after each of the first M of the K = 10 Euler steps by sigma / sqrt(K)
along the command, so the command keeps acting while the flow integrates. openpi integrates
from t = 1 (noise) to t = 0 (action), so "the first M steps" are the ones nearest the noise.

openpi's `Pi0.sample_actions` has no hook for this, and openpi must not be edited, so
`_sample_pushed` below is a copy of its loop (openpi aec5d4d, src/openpi/models/pi0.py,
without the train-time RTC prefix, which pi05_libero does not use) with one added line: after
each Euler step i < M, x_t += push. `push` is a (10, 32) array: sigma / sqrt(K) * sign on the
commanded dim for the first tau chunk steps (the same steps the starting-noise command uses),
zero elsewhere. M is passed as data, so every M shares one compiled function. M can also be set
per action dim (a length-32 array), so the gripper and a translation axis can be pushed for
different numbers of Euler steps in one sample (Stage 3 closed loop).

`PushedPolicy` wraps an openpi Policy: `infer(obs, noise, push, m)` returns the LIBERO actions
(10, 7) and the model-space chunk (10, 32); `conformity(obs, chunk)` is the frozen policy's
flow-matching loss on a model-space chunk, averaged over a fixed set of DRAWS noise and time
draws (paper Eq. 10): low means the chunk looks like the policy's own actions.
"""

from __future__ import annotations

import types

import numpy as np

import injection

K = 10  # Euler steps (openpi's default for pi05_libero)
DRAWS = 32  # conformity: noise and time draws, fixed by CONFORMITY_SEED
CONFORMITY_SEED = 0


def push_array(axis: int, sign: int, *, tau: int = injection.TAU, sigma: float = injection.SIGMA) -> np.ndarray:
    """The per-step shift for a command on `axis` (0-2 translation, 6 gripper) with `sign`."""
    p = np.zeros((injection.HORIZON, injection.MODEL_ACTION_DIM), dtype=np.float32)
    p[:tau, axis] = sign * sigma / np.sqrt(K)
    return p


def _sample_pushed(self, rng, observation, noise, push, m):
    """openpi's Pi0.sample_actions (no RTC prefix) plus `x_t += push` after Euler steps i < m."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    del rng
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / K
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time, i = carry
        timestep = jnp.broadcast_to(time, batch_size)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        x_next = x_t + dt * v_t
        x_next = x_next + jnp.where(i < m, 1.0, 0.0) * push  # the only change from openpi (m: per dim)
        return x_next, time + dt, i + 1

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, 0))
    return x_0


def _conformity(self, rng, observation, actions):
    return self.compute_loss(rng, observation, actions, train=False)


class PushedPolicy:
    def __init__(self, policy):
        import jax
        from openpi.shared import nnx_utils

        self._policy = policy
        model = policy._model
        self._sample = nnx_utils.module_jit(types.MethodType(_sample_pushed, model))
        self._loss = nnx_utils.module_jit(types.MethodType(_conformity, model))
        self._key = jax.random.key(CONFORMITY_SEED)

    def _observation(self, obs: dict, batch: int = 1):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[np.newaxis, ...], batch, axis=0), inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def infer(self, obs: dict, noise: np.ndarray, push: np.ndarray | None = None, m=0):
        """m: Euler steps with the push, one int for all dims or one per dim (length 32)."""
        import jax
        import jax.numpy as jnp

        inputs, observation = self._observation(obs)
        push = np.zeros_like(noise) if push is None else push
        m = np.broadcast_to(np.asarray(m, dtype=np.int32), (injection.MODEL_ACTION_DIM,))
        chunk = self._sample(self._key, observation, jnp.asarray(noise)[None], jnp.asarray(push)[None], jnp.asarray(m))
        outputs = {"state": np.asarray(inputs["state"][0]), "actions": np.asarray(chunk[0])}
        actions = self._policy._output_transform(jax.tree.map(lambda x: x, outputs))["actions"]
        return np.asarray(actions), np.asarray(chunk[0])

    def conformity(self, obs: dict, chunk: np.ndarray) -> float:
        import jax.numpy as jnp

        _, observation = self._observation(obs, DRAWS)
        actions = jnp.repeat(jnp.asarray(chunk)[None], DRAWS, axis=0)
        return float(np.asarray(self._loss(self._key, observation, actions)).mean())
