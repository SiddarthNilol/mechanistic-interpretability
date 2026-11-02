"""Step 2: every attention head's output on every frame of frames.csv, from the base pi0.5.

    python extraction/extract_heads.py --dataset /path/to/dataset \
        --frames prepared/hook_unhook/frames.csv --out $SCRATCH/mi/pi05_base

The model sees each frame the way it would during training: the three cameras resized to
224x224, and the prompt "Task: <task>, State: <state as 25 numbers in 0-255>;\nAction: ".
It then runs once:

    prefix pass   PaliGemma (Gemma 2B) over the image and text tokens
    suffix pass   the action expert (Gemma 300M) over 50 action tokens, at flow time 1
                  (pure noise, the first denoising step), attending to the prefix

Each of the 18 layers has 8 heads that share one key/value head. For one query token
per pass, this records each head's attention weights over the keys and its output,
h = softmax(q k^T / sqrt(d)) v (eq. 1 of the paper), a 256-dim vector:

    vlm      PaliGemma's heads at the last prompt token ("Action: "), which sees the
             images, task and state but not the actions
    action   the action expert's heads at the first action token, the one the paper uses

The noise is the same for every frame, so the suffix pass differs between frames only
through what it reads from the prefix.

--denoised does what the paper's released code does instead: for each frame, denoise from
fresh noise (10 steps, nothing recorded) to the model's predicted actions, then run the suffix
pass once more on those actions at flow time 0, and record there. The first action token then
carries the model's own predicted action.

Both the task and the state are text in the prompt, so heads can tell tasks and phases
apart by reading it. Two options take that away, leaving only the images to differ:

    --prompt ring    the same task text for every frame
    --mean-state     the dataset's mean state for every frame

Writes to --out, row i of every array = row i of frames.csv:

    heads_vlm.npy      (frames, 18, 8, 256) float32   head outputs (layer, head, dim)
    heads_action.npy   (frames, 18, 8, 256) float32
    attn_vlm.npy       (frames, 18, 8, 968) float16   attention weights over the keys
    attn_action.npy    (frames, 18, 8, 1018) float16
    tokens.npy         (frames, 200) int32            prompt token ids (0 = padding)
    frames.csv         copy of the input
    info.json          key layout, normalization, settings

Key layout (attn_* last axis): head camera 0-255, left wrist 256-511, right wrist 512-767,
prompt tokens 768-967 (padding has weight 0), and for attn_action the 50 action tokens
968-1017.

Needs the lair-openpi environment and a GPU. The base weights are downloaded to
$OPENPI_DATA_HOME on first use (or pass --params with a local copy).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from flax import nnx
from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import download
from openpi_client.image_tools import resize_with_pad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import iter_video_frames, read_episode_frames, read_episodes, read_excluded  # noqa: E402

BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
# openpi camera slot -> dataset video feature (as in the lair-train configs)
CAMERAS = {
    "base_0_rgb": "observation.images.head",
    "left_wrist_0_rgb": "observation.images.left_wrist",
    "right_wrist_0_rgb": "observation.images.right_wrist",
}
STATE = "observation.state"
FLOW_TIME = 1.0  # 1 = pure noise, the first of the denoising steps
DENOISED_FLOW_TIME = 0.0  # 0 = finished, the actions are clean


# -- recording the heads ---------------------------------------------------------


def record_heads(query: jax.Array, sink: list):
    """A flax method interceptor that records, in every layer, each head's attention
    weights and output for one query token per sample (`query`: its index, shape (b,)).

    openpi's Attention module computes q (in its q_einsum) and k, v for all tokens, and
    returns k, v (including the prefix's, in the suffix pass). From those, this recomputes
    the query token's attention like the module does:

        weights = softmax(q k^T / sqrt(head_dim)), masked     (b, heads, keys)
        heads   = weights v                                  (b, heads, head_dim)

    and sends both to `sink`, one (heads, weights) entry per layer in order.
    check_capture.py tests that `heads` equals what openpi computes internally."""
    captured = {}

    def interceptor(next_fun, args, kwargs, context):
        module = context.module
        if context.method_name != "__call__":
            return next_fun(*args, **kwargs)
        if (module.name or "").startswith("q_einsum"):  # q_einsum (PaliGemma) or q_einsum_1 (action expert)
            captured["q"] = next_fun(*args, **kwargs)  # (b, tokens, heads, head_dim), before RoPE
            return captured["q"]
        if isinstance(module, gemma.Attention):
            _, positions, attn_mask, _ = args
            out, (k, v) = next_fun(*args, **kwargs)  # k, v: (b, keys, 1 kv head, head_dim)
            rows = jnp.arange(query.shape[0])
            q = captured["q"][rows, query]  # (b, heads, head_dim)
            q = gemma._apply_rope(q[:, None], positions=positions[rows, query][:, None])[:, 0]
            q = q * module.configs[0].head_dim ** -0.5
            logits = jnp.einsum("bnd,bsd->bns", q, k[:, :, 0], preferred_element_type=jnp.float32)
            logits = jnp.where(attn_mask[rows, 0, query][:, None, :], logits, -jnp.inf)
            weights = jax.nn.softmax(logits, axis=-1)
            heads = jnp.einsum("bns,bsd->bnd", weights, v[:, :, 0].astype(jnp.float32))
            jax.debug.callback(lambda h, w: sink.append((np.asarray(h), np.asarray(w))), heads, weights, ordered=True)
            return out, (k, v)
        return next_fun(*args, **kwargs)

    return interceptor


def make_forward(model, sink: list):
    """A function (observation, actions, flow_time) -> None that runs the prefix pass and one
    suffix pass, as pi0.5's sample_actions does for one denoising step, and appends one
    (heads, weights) entry per layer to `sink`: the 18 PaliGemma layers at the last
    prompt token, then the 18 action-expert layers at the first action token. `actions` are
    the suffix's input: noise at flow time 1, clean actions at flow time 0."""
    graphdef, state = nnx.split(model)

    def forward(state, obs, actions, flow_time):
        m = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)
        b = obs.state.shape[0]

        # Prefix: image tokens, then prompt tokens, then padding. Image tokens are never
        # masked, so the last real prompt token is at (number of real tokens - 1).
        prefix, prefix_mask, prefix_ar = m.embed_prefix(obs)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        last_prompt_token = jnp.sum(prefix_mask, axis=1) - 1
        with nn.intercept_methods(record_heads(last_prompt_token, sink)):
            _, kv_cache = m.PaliGemma.llm(
                [prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar), positions=positions
            )

        # Suffix: the 50 action tokens attend to the whole prefix and to each other.
        suffix, suffix_mask, suffix_ar, adarms_cond = m.embed_suffix(obs, actions, jnp.full(b, flow_time))
        mask = jnp.concatenate(
            [jnp.repeat(prefix_mask[:, None], suffix.shape[1], axis=1), make_attn_mask(suffix_mask, suffix_ar)],
            axis=-1,
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        first_action_token = jnp.zeros(b, dtype=jnp.int32)
        with nn.intercept_methods(record_heads(first_action_token, sink)):
            m.PaliGemma.llm(
                [None, suffix], mask=mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
            )

    jitted = jax.jit(forward)
    return lambda obs, actions, flow_time=FLOW_TIME: jitted(state, obs, actions, flow_time)


def make_denoise(model):
    """A function (observation, noise) -> the model's predicted actions (10 denoising steps)."""
    graphdef, state = nnx.split(model)

    def denoise(state, obs, noise):
        return nnx.merge(graphdef, state).sample_actions(jax.random.key(0), obs, noise=noise)

    jitted = jax.jit(denoise)
    return lambda obs, noise: jitted(state, obs, noise)


# -- model inputs ------------------------------------------------------------------


def state_stats(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1st percentile, 99th percentile and mean of every state dim over all frames of the
    non-excluded episodes. openpi's quantile normalization for pi0.5 maps the percentiles
    to -1 and 1."""
    excluded = set(read_excluded(dataset))
    states = []
    for path in sorted((dataset / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", STATE]).to_pandas()
        table = table[~table["episode_index"].isin(excluded)]
        states.append(np.stack(table[STATE].to_numpy()))
    states = np.concatenate(states).astype(np.float64)
    return np.quantile(states, 0.01, axis=0), np.quantile(states, 0.99, axis=0), states.mean(axis=0)


def normalize_state(state: np.ndarray, q01: np.ndarray, q99: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """openpi's quantile normalization. By default clipped to [-1, 1]: the prompt bins [-1, 1]
    into 256 values, and values outside would become bin -1 or 256. openpi's trainer and the
    inference server don't clip, so fine-tuning uses clip=False."""
    x = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return np.clip(x, -1.0, 1.0) if clip else x


def episode_inputs(dataset, episodes, ep, frame_indices, task, q01, q99, tokenizer, fixed_state=None, clip_state=True) -> dict:
    """Model inputs for some frames of one episode: images per camera slot (n, 224, 224, 3)
    uint8, the normalized state (n, 25), and the prompt tokens and mask (n, 200). With
    `fixed_state` (25,), every frame gets that state instead of its own."""
    wanted = set(int(f) for f in frame_indices)
    images = {}
    for slot, feature in CAMERAS.items():
        decoded = {i: np.asarray(img) for i, img in iter_video_frames(dataset, episodes, ep, feature) if i in wanted}
        images[slot] = resize_with_pad(np.stack([decoded[f] for f in frame_indices]), 224, 224)
    state = np.stack(read_episode_frames(dataset, episodes, ep)[STATE].to_numpy())[frame_indices]
    if fixed_state is not None:
        state = np.broadcast_to(fixed_state, state.shape)
    state = normalize_state(state.astype(np.float64), q01, q99, clip=clip_state)
    tokens, token_mask = zip(*(tokenizer.tokenize(task, s) for s in state))
    return {"images": images, "state": state, "tokens": np.stack(tokens), "token_mask": np.stack(token_mask)}


def observation(inputs: dict, rows: np.ndarray, action_dim: int) -> _model.Observation:
    """A batched model input from the given rows of episode_inputs()."""
    state = inputs["state"][rows]
    return _model.Observation(
        images={slot: jnp.asarray(img[rows], jnp.float32) / 255.0 * 2.0 - 1.0 for slot, img in inputs["images"].items()},
        image_masks={slot: jnp.ones(len(rows), dtype=bool) for slot in inputs["images"]},
        # pi0.5 reads the state from the prompt; this zero-padded copy only fills the field
        state=jnp.asarray(np.pad(state, ((0, 0), (0, action_dim - state.shape[1]))), jnp.float32),
        tokenized_prompt=jnp.asarray(inputs["tokens"][rows], jnp.int32),
        tokenized_prompt_mask=jnp.asarray(inputs["token_mask"][rows], bool),
    )


# -- main --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--frames", type=Path, required=True, help="frames.csv from build_labels.py")
    parser.add_argument("--out", type=Path, required=True, help="output directory (gets ~5 GB)")
    parser.add_argument("--params", default=BASE_PARAMS, help="model weights (default: the released pi05_base)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="seed of the action noise")
    parser.add_argument("--denoised", action="store_true", help="feed the action tokens as the paper's code does")
    parser.add_argument("--prompt", help="task text for every frame (default: each frame's task)")
    parser.add_argument("--mean-state", action="store_true", help="give every frame the dataset's mean state")
    args = parser.parse_args()

    frames = pd.read_csv(args.frames)
    episodes = read_episodes(args.dataset)
    config = pi0_config.Pi0Config(pi05=True)
    print(f"loading {args.params}", flush=True)
    params = _model.restore_params(download.maybe_download(args.params), dtype=jnp.bfloat16)
    model = config.load(params)
    tokenizer = PaligemmaTokenizer(config.max_token_len)
    q01, q99, mean_state = state_stats(args.dataset)
    noise_key = jax.random.key(args.seed)
    noise = jax.random.normal(noise_key, (1, config.action_horizon, config.action_dim))
    sink = []
    forward = make_forward(model, sink)
    denoise = make_denoise(model)
    n_batches = 0

    gemma_config = gemma.get_config(config.paligemma_variant)  # the action expert has the same layout
    n, layers, heads, dim = len(frames), gemma_config.depth, gemma_config.num_heads, gemma_config.head_dim
    n_prefix = len(CAMERAS) * 256 + config.max_token_len  # 256 tokens per image
    args.out.mkdir(parents=True, exist_ok=True)

    def array(name, shape, dtype):
        return np.lib.format.open_memmap(args.out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    out = {
        "heads_vlm": array("heads_vlm", (n, layers, heads, dim), np.float32),
        "heads_action": array("heads_action", (n, layers, heads, dim), np.float32),
        "attn_vlm": array("attn_vlm", (n, layers, heads, n_prefix), np.float16),
        "attn_action": array("attn_action", (n, layers, heads, n_prefix + config.action_horizon), np.float16),
        "tokens": array("tokens", (n, config.max_token_len), np.int32),
    }

    for e, (ep, ep_frames) in enumerate(frames.groupby("episode_index", sort=False)):
        task = ep_frames["task"].iloc[0]
        inputs = episode_inputs(
            args.dataset, episodes, ep, ep_frames["frame_index"].to_numpy(), args.prompt or task, q01, q99,
            tokenizer, mean_state if args.mean_state else None,
        )
        out["tokens"][ep_frames.index] = inputs["tokens"]
        for start in range(0, len(ep_frames), args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, len(ep_frames)))
            padded = np.pad(rows, (0, args.batch_size - len(rows)), mode="edge")  # one batch shape, one compile
            sink.clear()
            obs = observation(inputs, padded, config.action_dim)
            if args.denoised:
                n_batches += 1
                fresh = jax.random.normal(
                    jax.random.fold_in(noise_key, n_batches), (args.batch_size, config.action_horizon, config.action_dim)
                )
                forward(obs, denoise(obs, fresh), DENOISED_FLOW_TIME)
            else:
                forward(obs, jnp.repeat(noise, args.batch_size, axis=0), FLOW_TIME)
            jax.effects_barrier()  # wait until every callback has run
            if len(sink) != 2 * layers:
                raise RuntimeError(f"recorded {len(sink)} layers, expected {2 * layers}")
            target = ep_frames.index[rows]
            for name, entries in (("vlm", sink[:layers]), ("action", sink[layers:])):
                out[f"heads_{name}"][target] = np.stack([h for h, _ in entries], axis=1)[: len(rows)]
                out[f"attn_{name}"][target] = np.stack([w for _, w in entries], axis=1)[: len(rows)]
        print(f"[{e + 1}/{frames['episode_index'].nunique()}] episode {ep} ({task}): {len(ep_frames)} frames", flush=True)

    for a in out.values():
        a.flush()
    shutil.copy(args.frames, args.out / "frames.csv")
    info = {
        "params": args.params,
        "action_input": "denoised actions" if args.denoised else "noise",
        "flow_time": DENOISED_FLOW_TIME if args.denoised else FLOW_TIME,
        "noise_seed": args.seed,
        "prompt": args.prompt or "each frame's task",
        "state": "dataset mean" if args.mean_state else "each frame's",
        "cameras": CAMERAS,
        "layers": layers,
        "heads_per_layer": heads,
        "head_dim": dim,
        "keys": {"head": [0, 256], "left_wrist": [256, 512], "right_wrist": [512, 768],
                 "prompt": [768, n_prefix], "actions": [n_prefix, n_prefix + config.action_horizon]},
        "state_q01": q01.tolist(),
        "state_q99": q99.tolist(),
    }
    (args.out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {n} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
