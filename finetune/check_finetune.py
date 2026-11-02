"""Checks that finetune_heads.py trains exactly what it says, with no weights or data needed.

    python finetune/check_finetune.py

1. On the full-size model (shapes only): the number of trainable parameters per group, which the
   paper's README lists as about 1.5 M, 15.9 M, 2.1 M and 0.07 M.
2. On a tiny random model with LoRA, a few training steps with 3 chosen heads: the parts that should
   train changed, and every other parameter is bit-for-bit the same, including the other heads'
   slices of the same arrays. PaliGemma's last layer is also unchanged, whatever is chosen: the
   action tokens read each layer's keys and values, which come from the layer before, so that
   layer's outputs never reach the loss.
3. The checkpoint, loaded the way the inference server loads it: a plain model without LoRA gives
   the same actions as the trained model with LoRA. Without merging they would differ.
4. Steps 2 and 3 again with train-time RTC: the committed actions come back unchanged at the start
   of the chunk, and the plain model still gives the trained model's actions.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from openpi.models import gemma
from openpi.models import lora
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import nnx_utils

import finetune_heads as ft

BATCH = 2
TINY_HORIZON = 4
CHOSEN = [(0, 2), (0, 6), (2, 1)]  # the tiny model has 4 layers; layer 3 is the last
LAST_LAYER = 3


def check_full_size_counts():
    config = ft.make_config()
    state = jax.eval_shape(lambda: nnx.state(config.create(jax.random.key(0))))
    trainable = state.filter(ft.trainable_filter()).to_pure_dict()

    for path in traverse_util.flatten_dict(trainable, sep="/"):
        assert "img" not in path and "_1/" not in path, f"{path} should be frozen"

    # head_mask needs real shapes only for np.zeros, so 20 heads costs little memory
    heads = [(layer, 0) for layer in range(18)] + [(0, 1), (1, 1)]
    counts = ft.count_trainable(trainable, ft.head_mask(trainable, heads))
    print(f"full size, 20 heads: {counts}, total {sum(counts.values()):,}")
    # Q and O LoRA per head: (2048 x 16 + 16 x 256) x 2, as the rank is 16
    assert counts["selected heads"] == 20 * 2 * (2048 * 16 + 16 * 256), counts
    expected = {"FFN LoRA": 15.9e6, "time MLP": 2.1e6, "action projections": 0.07e6}
    for name, value in expected.items():
        assert abs(counts[name] - value) / value < 0.05, (name, counts[name], value)


def tiny_config(**rtc):
    """The openpi dummy size, with LoRA in the variants that have it."""
    original = gemma.get_config

    def get_config(variant):
        config = original("dummy")
        if "lora" in variant:
            config.lora_configs = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
        return config

    gemma.get_config = get_config
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=TINY_HORIZON,
        loss_action_dim=ft.ACTION_DIM,
        max_token_len=20,
        **rtc,
    )


def open_gates(params, key):
    """A random model's action expert ignores attention: its adaRMS layers start at zero, which
    closes the gates on each layer's update. The real checkpoint has them open, so do the same."""
    flat = traverse_util.flatten_dict(params.to_pure_dict(), sep="/")
    for path, value in flat.items():
        if "_1/Dense_0/kernel" in path:
            key, sub = jax.random.split(key)
            flat[path] = 0.1 * jax.random.normal(sub, value.shape, value.dtype)
    params.replace_by_pure_dict(traverse_util.unflatten_dict(flat, sep="/"))
    return params


def check_training_is_exact(config):
    model = config.create(jax.random.key(0))
    graphdef, params = nnx.split(model)
    params = ft.to_training_dtypes(open_gates(params, jax.random.key(2)))
    before = traverse_util.flatten_dict(params.to_pure_dict(), sep="/")
    before = {k: np.asarray(v) for k, v in before.items()}

    trainable = params.filter(ft.trainable_filter()).to_pure_dict()
    mask = ft.head_mask(trainable, CHOSEN)
    tx = ft.make_optimizer(steps=10, peak_lr=1e-2, warmup=1)
    opt_state = tx.init(trainable)
    step_fn = jax.jit(lambda *a: ft.train_step(graphdef, tx, mask, *a))

    keys = iter(jax.random.split(jax.random.key(1), 8))
    slots = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = _model.Observation(
        images={s: jax.random.uniform(next(keys), (BATCH, 224, 224, 3), minval=-1, maxval=1) for s in slots},
        image_masks={s: jnp.ones(BATCH, dtype=bool) for s in slots},
        state=jnp.zeros((BATCH, config.action_dim)),
        tokenized_prompt=jax.random.randint(next(keys), (BATCH, 20), 1, 1000),
        tokenized_prompt_mask=jnp.ones((BATCH, 20), dtype=bool),
    )
    actions = jax.random.normal(next(keys), (BATCH, TINY_HORIZON, config.action_dim))

    for step in range(3):
        params, opt_state, info = step_fn(params, opt_state, jax.random.key(step), obs, actions)
        assert np.isfinite(float(info["loss"]))
    after = {k: np.asarray(v) for k, v in traverse_util.flatten_dict(params.to_pure_dict(), sep="/").items()}
    assert before.keys() == after.keys()

    n_changed = n_same = 0
    for path, old in before.items():
        new = after[path]
        assert old.dtype == new.dtype, path
        same = old.astype(np.float32) == new.astype(np.float32)
        is_attn_lora = "lora" in path and "/attn/" in path and "_1/" not in path
        if is_attn_lora and ("/q_einsum/" in path or "/attn_vec_einsum/" in path):
            for layer in range(old.shape[0]):
                for head in range(old.shape[1]):
                    if (layer, head) in CHOSEN and layer != LAST_LAYER:
                        assert not same[layer, head].all(), f"{path} [{layer},{head}] should train"
                    else:
                        assert same[layer, head].all(), f"{path} [{layer},{head}] should be frozen"
            n_changed += 1
        elif path in traverse_util.flatten_dict(trainable, sep="/") and "/attn/" not in path:
            assert not same.all(), f"{path} should train"
            if "/mlp/" in path:
                assert same[LAST_LAYER].all(), f"{path}: the last layer never reaches the loss"
            n_changed += 1
        else:
            assert same.all(), f"{path} should be frozen"
            n_same += 1
    print(f"tiny model: {n_changed} arrays trained as expected, {n_same} are bit-for-bit unchanged")

    return config, graphdef, params, obs


def check_server_loading(config, graphdef, params, obs):
    """Everything in float32, so that only a wrong merge can make the actions differ."""
    params = nnx_utils.state_map(params, nnx.Param, lambda p: p.replace(p.value.astype(jnp.float32)))
    noise = jax.random.normal(jax.random.key(3), (BATCH, TINY_HORIZON, config.action_dim))
    plain = dataclasses.replace(config, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    float32_model = dataclasses.replace(config, dtype="float32")
    float32_plain = dataclasses.replace(plain, dtype="float32")

    prefix = {}
    if config.rtc_train_enabled:
        committed = jax.random.normal(jax.random.key(5), noise.shape)
        prefix = {"action_prefix": committed, "prefix_len": 1}

    def sample(cfg, state):
        graph = nnx.graphdef(cfg.create(jax.random.key(0)))
        return np.asarray(nnx.merge(graph, state).sample_actions(jax.random.key(4), obs, noise=noise, **prefix))

    trained = sample(float32_model, params)
    if prefix:
        np.testing.assert_allclose(trained[:, 0], committed[:, 0], atol=1e-5)

    with tempfile.TemporaryDirectory() as tmp:
        stats = {"state": ft.norm_stats(np.random.rand(10, 25)), "actions": ft.norm_stats(np.random.rand(10, 25))}
        ft.save_checkpoint(Path(tmp), 3, params, stats, "task")
        assert (Path(tmp) / "3" / "assets" / "task" / "norm_stats.json").exists()

        # What the server reads to find the model type: paths end in kernel/value, no LoRA.
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = dict(checkpointer.metadata((Path(tmp) / "3" / "params").resolve()))
        paths = traverse_util.flatten_dict(dict(metadata["params"]), sep="/")
        assert all(path.endswith("/value") for path in paths)
        assert not any("lora" in path for path in paths)
        assert "action_in_proj/kernel/value" in paths

        restored = _model.restore_params(Path(tmp) / "3" / "params", restore_type=np.ndarray)
        served = float32_plain.load(restored)
        served_actions = np.asarray(served.sample_actions(jax.random.key(4), obs, noise=noise, **prefix))

    difference = float(np.abs(trained - served_actions).max())
    assert difference < 2e-4, difference

    # Dropping the LoRA, which is what the server would do to an unmerged checkpoint, is far worse.
    no_lora = traverse_util.flatten_dict(params.to_pure_dict(), sep="/")
    no_lora = traverse_util.unflatten_dict({k: v for k, v in no_lora.items() if "lora" not in k}, sep="/")
    dropped = np.asarray(float32_plain.load(no_lora).sample_actions(jax.random.key(4), obs, noise=noise, **prefix))
    dropped_difference = float(np.abs(trained - dropped).max())
    assert dropped_difference > 100 * difference, (dropped_difference, difference)
    print(f"served checkpoint: largest action difference {difference:.1e} (LoRA dropped: {dropped_difference:.1e})")


def check_actions():
    action = np.arange(3 * 25, dtype=np.float64).reshape(3, 25)
    chunks = ft.action_chunks(action)
    assert chunks.shape == (3, ft.HORIZON, 25)
    assert np.array_equal(chunks[0, 1], action[1])  # absolute, and the next frame's
    assert np.array_equal(chunks[2, 5], action[2])  # after the end, the last action repeats
    print("actions: absolute, the end repeats")


if __name__ == "__main__":
    check_actions()
    check_full_size_counts()
    check_server_loading(*check_training_is_exact(tiny_config()))
    print("with train-time RTC:")
    check_server_loading(*check_training_is_exact(tiny_config(**ft.rtc_settings(True, 0, 2))))
    print("ok")
