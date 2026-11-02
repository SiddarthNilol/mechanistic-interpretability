"""Fine-tune pi0.5 on one task, training only the chosen heads, as the paper's released code does.

    python finetune/finetune_heads.py --dataset $DATA --task "unhook ring" \
        --selection $H/selected_heads.json --out $SCRATCH/mechanistic-interpretability/finetune/unhook_selected

The model is the base pi0.5 with LoRA adapters added: small extra matrices beside the large weights
of Gemma. The large weights never change. Only these parts train:

    Q and O LoRA of the chosen heads    in PaliGemma (not the action expert)      about 8%
    FFN LoRA of every PaliGemma layer                                             about 81%
    time MLP                                                                      about 11%
    action projections                                                            under 1%

Everything else stays frozen: SigLIP, the whole action expert, the large Gemma weights, and all key
and value (KV) LoRA. Each layer stores its heads' Q and O LoRA in one array, so a head's freezing is
done with a mask: the gradient of every other head is multiplied by 0.

Note what follows from the paper's code: heads are scored in the action expert, but the same
(layer, head) numbers are trained in PaliGemma. The frozen LoRA keep their small random start. And
PaliGemma's last layer (17) never trains, though it is counted above: the action tokens read each
layer's keys and values, which come from the layer before, so its outputs never reach the loss.

--heads chooses the heads, so a run can be compared with a control:

    selected   the 20 heads for this task in selected_heads.json (select_heads.py)
    random     20 random heads
    all        all 144 heads

From the paper's code: 3000 steps, batch 32, AdamW, learning rate 2.5e-5 falling to 2.5e-6 with 200
warmup steps, image augmentation, no EMA.

What we had to adapt, because the paper's robot differs from ours:

    Checkpoint   pi05_base, as in the head selection, not the paper's pi05_droid or pi05_libero.
    Horizon      50 action tokens (1 s at 50 fps), as in our lab's runs.
    Actions      All 25 dims, absolute joint positions, because the inference server returns the model's
                 output to the robot as absolute positions. The selection scored motion instead.
    Loss         Averaged over our 25 dims, not the 32 padded ones, as in our lab's runs.

--rtc-train adds train-time real-time chunking (lair-openpi, arXiv 2512.05964), which the paper doesn't
use. While the next chunk is computed the robot keeps acting, so the new chunk's first d actions are
already committed. Each training sample draws a delay d from [--rtc-min-delay, --rtc-max-delay),
gets its first d actions clean instead of noisy, and is scored only on the rest. The loss then covers
different actions, so compare held-out losses only between runs that both use RTC or both don't.

The checkpoints are made for the inference server (policy-inference-spec, server/server_openpi.py).
It builds a plain pi0.5 with no LoRA and silently ignores any LoRA arrays in a checkpoint, so each
LoRA is merged into its large weight before saving, which gives the same layout as a lair-train
checkpoint. Serve one with:

    python -m server.server_openpi --checkpoint-path <out>/3000 --task "unhook ring"

and add --rtc-train for a run trained with it. The checkpoint doesn't record this; only run.json does.

Merged weights are rounded to half precision when served, so the end of the run loads the last
checkpoint the way the server does and writes both held-out losses to served_check.json.

A note on the O LoRA: openpi computes it as (sum over heads of x_n A_n) (sum over heads of B_n), so a
chosen head's B also changes what the other heads of its layer send out. Masking is exact for the
parameters, but in this one matrix it isn't exact for the effect. The paper's code does the same.

The script trains on the task's 20 episodes of episodes.csv, all frames. All of them fit in memory
(about 6 to 9 GB), so there is no data-loader. The held-out loss uses other episodes of the same
task, always the same ones with the same noise, so it can be compared between runs. Step 0 is the
model before training.

Logs to wandb like lair-train: loss, grad_norm, param_norm and learning_rate every 100 steps, eval/loss
at step 0 and at each checkpoint, and camera_views at step 0. The run is named after --out, and
--no-wandb turns it off.

Writes to --out:

    run.json                         settings, heads, training and held-out episodes
    metrics.csv                      step, loss, grad_norm, learning_rate, heldout_loss
    served_check.json                held-out loss of the trained model and of the served checkpoint
    wandb_id.txt                     the wandb run
    <step>/params/                   merged weights, laid out like a lair-train checkpoint
    <step>/assets/<task>/norm_stats.json
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import sys
import time
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import wandb
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "extraction"), str(ROOT / "data_prep")]
from common import column, read_episode_frames, read_episodes, read_excluded  # noqa: E402
from extract_heads import BASE_PARAMS, STATE, episode_inputs, observation  # noqa: E402

ACTION_DIM = 25  # of our robot; the model pads to 32
HORIZON = 50
LAYERS, HEADS = 18, 8
LOG_EVERY = 100
SAVE_EVERY = 1000
EVAL_EVERY_NTH_FRAME = 5
HELDOUT_SEED = 0  # not --seed: every run must be compared on the same episodes


# -- which parameters train ------------------------------------------------------------


def freeze_filter():
    """The parameters that are never trained (from the paper's pi0_config.py)."""
    expert = nnx_utils.PathRegex(".*llm.*_1.*")
    siglip = nnx_utils.PathRegex(".*img.*")
    main_llm = nnx_utils.PathRegex(".*llm.*")
    lora = nnx_utils.PathRegex(".*lora.*")
    return nnx.Any(expert, siglip, nnx.All(main_llm, nnx.Not(lora)))


def trainable_filter():
    return nnx.All(nnx.Param, nnx.Not(freeze_filter()))


def head_mask(trainable: dict, heads: list[tuple[int, int]]) -> dict:
    """For a nested dict of trainable parameters, what to multiply each one's gradient by: for
    attention LoRA 0 everywhere except at the chosen (layer, head), and for the rest 1. Q and O LoRA
    arrays start with (layer, head, ...)."""
    masks = {}
    for path, leaf in traverse_util.flatten_dict(trainable, sep="/").items():
        if "/attn/" not in path:
            masks[path] = 1.0
        elif "/q_einsum/" in path or "/attn_vec_einsum/" in path:
            mask = np.zeros(leaf.shape, np.float32)
            for layer, head in heads:
                mask[layer, head] = 1.0
            masks[path] = jnp.asarray(mask)
        elif "/kv_einsum/" in path:
            masks[path] = 0.0
        else:
            raise ValueError(f"unexpected attention parameter {path}")
    return traverse_util.unflatten_dict(masks, sep="/")


def count_trainable(trainable: dict, mask: dict) -> dict[str, int]:
    """Number of parameters that really train, per group."""
    counts = {"selected heads": 0, "FFN LoRA": 0, "time MLP": 0, "action projections": 0}
    flat_mask = traverse_util.flatten_dict(mask, sep="/")
    for path, leaf in traverse_util.flatten_dict(trainable, sep="/").items():
        if "/attn/" in path:
            counts["selected heads"] += int(np.sum(np.broadcast_to(flat_mask[path], leaf.shape)))
        elif "/mlp/" in path:
            counts["FFN LoRA"] += int(np.prod(leaf.shape))
        elif "time_mlp" in path:
            counts["time MLP"] += int(np.prod(leaf.shape))
        elif "action_" in path:
            counts["action projections"] += int(np.prod(leaf.shape))
        else:
            raise ValueError(f"unexpected trainable parameter {path}")
    return counts


def choose_heads(mode: str, selection: Path | None, task: str, seed: int) -> list[tuple[int, int]]:
    if mode == "all":
        return [(layer, head) for layer in range(LAYERS) for head in range(HEADS)]
    if mode == "random":
        chosen = np.random.default_rng(seed).choice(LAYERS * HEADS, size=20, replace=False)
        return sorted((int(h) // HEADS, int(h) % HEADS) for h in chosen)
    if selection is None:
        raise ValueError("--heads selected needs --selection")
    rows = [s for s in json.loads(selection.read_text()) if s["stream"] == "action" and s["task"] == task]
    if len(rows) != 1:
        raise ValueError(f"{selection} has {len(rows)} action-stream rows for task {task!r}")
    return [tuple(h) for h in rows[0]["heads"]]


# -- data ------------------------------------------------------------------------------


def action_chunks(action: np.ndarray) -> np.ndarray:
    """For every frame t the next HORIZON actions (frames, HORIZON, 25). After the episode's end
    the last action repeats."""
    t = np.arange(len(action))
    return action[np.minimum(t[:, None] + np.arange(HORIZON), len(action) - 1)]


def norm_stats(x: np.ndarray) -> normalize.NormStats:
    """One row per sample, one column per dim."""
    return normalize.NormStats(
        mean=x.mean(axis=0), std=x.std(axis=0), q01=np.quantile(x, 0.01, axis=0), q99=np.quantile(x, 0.99, axis=0)
    )


def normalize_actions(chunks: np.ndarray, stats: normalize.NormStats) -> np.ndarray:
    """openpi's quantile normalization, zero-padded to the model's 32 dims."""
    x = (chunks - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
    return np.pad(x, ((0, 0), (0, 0), (0, 32 - ACTION_DIM))).astype(np.float32)


def heldout_episodes(dataset: Path, task: str, training: list[int], n: int) -> list[int]:
    episodes = read_episodes(dataset)
    tasks = pd.Series([str(t[0]) for t in episodes["tasks"]], index=episodes.index)
    candidates = tasks.drop(read_excluded(dataset))
    candidates = candidates[candidates == task].index.difference(training).tolist()
    picked = np.random.default_rng(HELDOUT_SEED).choice(candidates, size=n, replace=False)
    return sorted(int(e) for e in picked)


def load_frames(dataset, episode_list, task, stats, tokenizer, every=1) -> dict:
    """Model inputs and normalized actions of the given episodes, held in memory."""
    episodes = read_episodes(dataset)
    parts = []
    for ep in episode_list:
        data = read_episode_frames(dataset, episodes, ep)
        rows = np.arange(0, len(data), every)
        inputs = episode_inputs(
            dataset, episodes, ep, rows, task, stats["state"].q01, stats["state"].q99, tokenizer, clip_state=False
        )
        chunks = action_chunks(column(data, "action"))[rows]
        inputs["actions"] = normalize_actions(chunks, stats["actions"])
        parts.append(inputs)
        print(f"  episode {ep}: {len(rows)} frames", flush=True)
    return {
        "images": {slot: np.concatenate([p["images"][slot] for p in parts]) for slot in parts[0]["images"]},
        **{key: np.concatenate([p[key] for p in parts]) for key in ("state", "tokens", "token_mask", "actions")},
    }


def compute_stats(dataset, episode_list) -> dict:
    """Normalization statistics from the training episodes only."""
    episodes = read_episodes(dataset)
    states, chunks = [], []
    for ep in episode_list:
        data = read_episode_frames(dataset, episodes, ep)
        states.append(column(data, STATE))
        chunks.append(action_chunks(column(data, "action")).reshape(-1, ACTION_DIM))
    return {"state": norm_stats(np.concatenate(states)), "actions": norm_stats(np.concatenate(chunks))}


def batch_of(data: dict, rows: np.ndarray, action_dim: int):
    return observation(data, rows, action_dim), jnp.asarray(data["actions"][rows])


def shuffled_batches(n_frames: int, batch_size: int, seed: int):
    """Endless batches of frame rows; every frame once per pass over the data."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n_frames)
        for start in range(0, n_frames - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


# -- training --------------------------------------------------------------------------


def rtc_settings(enabled: bool, min_delay: int, max_delay: int) -> dict:
    """Pi0Config's train-time RTC fields."""
    return {"rtc_train_enabled": enabled, "rtc_train_min_delay": min_delay, "rtc_train_max_delay": max_delay}


def make_config(**rtc) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=HORIZON,
        loss_action_dim=ACTION_DIM,
        **rtc,
    )


def make_served_config(**rtc) -> pi0_config.Pi0Config:
    """The model the inference server builds: the same, without LoRA."""
    return pi0_config.Pi0Config(pi05=True, action_horizon=HORIZON, loss_action_dim=ACTION_DIM, **rtc)


def load_model(config, params_path: str, seed: int):
    """The model with the checkpoint's weights; the LoRA, which it lacks, keep their random start."""
    graphdef, state = nnx.split(config.create(jax.random.key(seed)))
    state.replace_by_pure_dict(weight_loaders.CheckpointWeightLoader(params_path).load(state.to_pure_dict()))
    return graphdef, state


def to_training_dtypes(state):
    """Frozen weights in half precision, as in openpi's trainer, to save memory."""
    return nnx_utils.state_map(state, freeze_filter(), lambda p: p.replace(p.value.astype(jnp.bfloat16)))


def make_optimizer(steps: int, peak_lr: float, warmup: int) -> optax.GradientTransformation:
    schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=warmup, peak_lr=peak_lr, decay_steps=steps, decay_lr=peak_lr / 10
    )
    return _optimizer.create_optimizer(_optimizer.AdamW(), schedule)


def train_step(graphdef, tx, mask, params, opt_state, rng, obs, actions):
    """One optimizer step (from the paper's train.py). Multiplying the gradients by the mask first
    means gradient clipping and Adam only see the parameters that train."""
    model = nnx.merge(graphdef, params)
    model.train()

    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, obs, actions, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter()))(model)
    grads = jax.tree.map(lambda g, m: g * m, grads.to_pure_dict(), mask)
    trainable = params.filter(trainable_filter()).to_pure_dict()
    updates, opt_state = tx.update(grads, opt_state, trainable)
    nnx.update(model, nnx.State(optax.apply_updates(trainable, updates)))
    kernels = nnx.state(  # the weights openpi's trainer reports as param_norm
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernels)}
    return nnx.state(model), opt_state, info


def make_eval(graphdef):
    @jax.jit
    def eval_fn(params, rng, obs, actions):
        model = nnx.merge(graphdef, params)
        model.eval()
        return jnp.mean(model.compute_loss(rng, obs, actions, train=False))

    return eval_fn


def heldout_loss(eval_fn, params, data: dict, batch_size: int, action_dim: int) -> float:
    """Mean loss over the held-out frames, with the same noise every time. Leftover frames that
    don't fill a batch are skipped."""
    losses = []
    for i in range(len(data["state"]) // batch_size):
        obs, actions = batch_of(data, np.arange(i * batch_size, (i + 1) * batch_size), action_dim)
        losses.append(float(eval_fn(params, jax.random.key(i), obs, actions)))
    return float(np.mean(losses))


def merge_lora(params: dict) -> dict:
    """The same model without LoRA: every LoRA pair (A, B) is added to its large weight w. The
    LoRA of our two Gemma variants has alpha = rank, so its scale is 1.

        most weights       w + A B                       per head where they have heads
        attn_vec_einsum    w[n] + A[n] (sum over m of B[m])   see the module docstring
    """
    flat = traverse_util.flatten_dict(params, sep="/")

    def as_float32(path):
        return np.asarray(flat[path]).astype(np.float32)

    # One weight at a time, keeping no float32 copy of the old weights: the model has 3.4 billion
    # parameters, so holding both costs about 10 GB more memory.
    merged = {}
    for path in flat:
        if "lora" in path:
            continue
        # attention: .../q_einsum/lora_a beside .../q_einsum/w; FFN: .../linear_lora_a beside .../linear
        lora_a = path.removesuffix("w") + "lora_a" if path.endswith("/w") else path + "_lora_a"
        merged[path] = as_float32(path)
        if lora_a in flat:
            b = as_float32(lora_a.removesuffix("a") + "b")
            if "/attn_vec_einsum" in path:
                b = b.sum(axis=1, keepdims=True)  # (layers, heads, rank, width): over the heads
            merged[path] += as_float32(lora_a) @ b
    return traverse_util.unflatten_dict(merged, sep="/")


def save_checkpoint(out: Path, step: int, params, stats: dict, asset: str) -> None:
    """Merged weights, each under a "value" key: openpi's trainer saves them that way, and the
    server's check of the model type expects it."""
    merged = traverse_util.flatten_dict(merge_lora(params.to_pure_dict()))
    tree = traverse_util.unflatten_dict({(*k, "value"): v for k, v in merged.items()})
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save((out / str(step) / "params").resolve(), {"params": tree})
    normalize.save(out / str(step) / "assets" / asset, stats)


def load_as_served(out: Path, step: int, **rtc):
    """A checkpoint loaded the way the inference server loads it."""
    params = _model.restore_params(out / str(step) / "params", dtype=jnp.bfloat16)
    return nnx.split(make_served_config(**rtc).load(params))


def camera_views(data: dict, n: int = 5) -> list:
    """The first frames as the model gets them, cameras side by side, to check them by eye."""
    return [wandb.Image(np.concatenate([images[i] for images in data["images"].values()], axis=1)) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--task", required=True, choices=["hook ring", "unhook ring"])
    parser.add_argument("--out", type=Path, required=True, help="output directory (about 12 GB per checkpoint)")
    parser.add_argument("--heads", default="selected", choices=["selected", "random", "all"])
    parser.add_argument("--selection", type=Path, help="selected_heads.json, for --heads selected")
    parser.add_argument("--episodes", type=Path, default=ROOT / "prepared/hook_unhook/episodes.csv")
    parser.add_argument("--params", default=BASE_PARAMS, help="starting weights (default: the released pi05_base)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--peak-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--heldout-episodes", type=int, default=10)
    parser.add_argument("--rtc-train", action="store_true", help="train-time real-time chunking")
    parser.add_argument("--rtc-min-delay", type=int, default=0, help="smallest simulated delay, in control steps")
    parser.add_argument("--rtc-max-delay", type=int, default=15, help="delays stay below this (the server's default)")
    parser.add_argument("--seed", type=int, default=0, help="LoRA start, batches, noise and --heads random")
    parser.add_argument("--wandb-project", default="mechanistic-interpretability")
    parser.add_argument("--no-wandb", action="store_true", help="don't log to wandb")
    args = parser.parse_args()

    if (args.out / "run.json").exists():
        raise FileExistsError(f"{args.out} already has a run; choose another --out")
    args.out.mkdir(parents=True, exist_ok=True)
    # Before the frames load, so that a missing wandb login fails at once.
    wandb.init(
        project=args.wandb_project,
        name=args.out.name,
        mode="disabled" if args.no_wandb else None,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    if not args.no_wandb:
        (args.out / "wandb_id.txt").write_text(wandb.run.id)

    table = pd.read_csv(args.episodes)
    training = table.loc[table["task"] == args.task, "episode_index"].tolist()
    heldout = heldout_episodes(args.dataset, args.task, training, args.heldout_episodes)
    heads = choose_heads(args.heads, args.selection, args.task, args.seed)
    asset = re.sub(r"\W+", "_", args.task)
    print(f"{args.task}: {args.heads} heads {heads}\ntraining on {training}\nheld out {heldout}", flush=True)

    rtc = rtc_settings(args.rtc_train, args.rtc_min_delay, args.rtc_max_delay)
    config = make_config(**rtc)
    stats = compute_stats(args.dataset, training)
    tokenizer = PaligemmaTokenizer(config.max_token_len)
    print("loading the training frames", flush=True)
    train_data = load_frames(args.dataset, training, args.task, stats, tokenizer)
    print("loading the held-out frames", flush=True)
    eval_data = load_frames(args.dataset, heldout, args.task, stats, tokenizer, every=EVAL_EVERY_NTH_FRAME)

    print(f"loading {args.params}", flush=True)
    graphdef, params = load_model(config, args.params, args.seed)
    params = to_training_dtypes(params)
    trainable = params.filter(trainable_filter()).to_pure_dict()
    mask = head_mask(trainable, heads)
    counts = count_trainable(trainable, mask)
    print(f"trainable parameters: {sum(counts.values()):,} {counts}", flush=True)
    tx = make_optimizer(args.steps, args.peak_lr, args.warmup_steps)
    opt_state = tx.init(trainable)
    learning_rate = _optimizer.CosineDecaySchedule(
        warmup_steps=args.warmup_steps, peak_lr=args.peak_lr, decay_steps=args.steps, decay_lr=args.peak_lr / 10
    ).create()

    step_fn = jax.jit(functools.partial(train_step, graphdef, tx, mask))
    eval_fn = make_eval(graphdef)

    resolved = {
        "selected_heads": heads,
        "training_episodes": training,
        "heldout_episode_list": heldout,
        "trainable": counts,
        "n_training_frames": len(train_data["state"]),
        "n_heldout_frames": len(eval_data["state"]),
        "horizon": HORIZON,
    }
    wandb.config.update(resolved)
    (args.out / "run.json").write_text(
        json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, **resolved}, indent=2)
    )

    key = jax.random.key(args.seed + 1)
    batches = shuffled_batches(len(train_data["state"]), args.batch_size, args.seed)
    with (args.out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "grad_norm", "learning_rate", "heldout_loss"])

        def evaluate():
            return heldout_loss(eval_fn, params, eval_data, args.batch_size, config.action_dim)

        before = evaluate()
        writer.writerow([0, "", "", "", f"{before:.5f}"])
        f.flush()
        print(f"step 0: held-out loss {before:.4f}", flush=True)
        wandb.log({"eval/loss": before, "camera_views": camera_views(train_data)}, step=0)

        start, recent = time.time(), []
        for step in range(1, args.steps + 1):
            obs, actions = batch_of(train_data, next(batches), config.action_dim)
            rng = jax.random.fold_in(key, step)
            params, opt_state, info = step_fn(params, opt_state, rng, obs, actions)
            recent.append(info)
            if step % LOG_EVERY == 0 or step == args.steps:
                mean = {k: float(np.mean([r[k] for r in recent])) for k in recent[0]}
                recent = []
                is_save = step % SAVE_EVERY == 0 or step == args.steps
                heldout_value = evaluate() if is_save else None
                lr = float(learning_rate(step - 1))
                writer.writerow(
                    [step, f"{mean['loss']:.5f}", f"{mean['grad_norm']:.5f}", f"{lr:.3e}",
                     "" if heldout_value is None else f"{heldout_value:.5f}"]
                )
                f.flush()
                text = f"step {step}: loss {mean['loss']:.4f}, grad norm {mean['grad_norm']:.3f}, {time.time() - start:.0f} s"
                print(text if heldout_value is None else f"{text}, held-out loss {heldout_value:.4f}", flush=True)
                wandb.log(
                    {**mean, "learning_rate": lr, **({} if heldout_value is None else {"eval/loss": heldout_value})},
                    step=step,
                )
                if is_save:
                    save_checkpoint(args.out, step, params, stats, asset)

    trained = heldout_loss(eval_fn, params, eval_data, args.batch_size, config.action_dim)
    served_graphdef, served_params = load_as_served(args.out, args.steps, **rtc)
    served = heldout_loss(make_eval(served_graphdef), served_params, eval_data, args.batch_size, config.action_dim)
    (args.out / "served_check.json").write_text(json.dumps({"trained": trained, "served": served}, indent=2))
    print(f"held-out loss: trained model {trained:.4f}, as served {served:.4f}", flush=True)
    wandb.summary.update({"served_check/trained": trained, "served_check/served": served})
    wandb.finish()
    print(f"done: {args.out}", flush=True)


if __name__ == "__main__":
    main()
