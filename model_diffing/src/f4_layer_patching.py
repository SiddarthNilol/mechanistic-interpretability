"""F4: layer-level activation patching, sufficiency direction. One GPU, about 10 to 20 minutes.

    python src/f4_layer_patching.py

Base run, with one fine-tuned activation swapped in at a time (same observation, prompt, noise
and flow steps for both models):

    (a) residual after r layers, r = 0 (the token embeddings) to 17: all prefix tokens' residual
        stream is replaced by the fine-tuned model's. The base backbone's remaining layers then
        run on it, so the KV cache of layers r + 1 to 18 changes.
    (b) KV at layer l, l = 1 to 18: only that layer's prefix keys and values are replaced, the
        other 17 layers keep the base model's.

R = (L_base - L_patched) / L_base, L = squared error to the fine-tuned model's action chunk
(7 real dims), as a ratio of means over states with 95% bootstrap bands over states.

The residual after layer 18 is not shown: the action expert reads only keys and values, so
nothing reads it (check_pair.py tests that patching it changes nothing).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np

from common import BASE_COLOR, DATA, FT_COLOR, REAL_DIMS, SEED, bootstrap_ci, log_run, save_figure, style


def compute(args) -> dict:
    import pair as P
    import states as S

    pair = P.Pair.load()
    depth = pair.depth
    inputs = S.Inputs(S.load_states(patch_only=True)[: args.limit])

    def activations(models, obs, noise):
        """Both models' unpatched prefix activations and action chunks."""
        obs = P.preprocess(obs)
        acts, chunks = {}, {}
        for name, model in models.items():
            tokens, mask = P.embed(model, obs)
            kv, residual = P.run_prefix(model, tokens, mask)
            acts[name] = {"tokens": tokens, "mask": mask, "kv": kv, "residual": residual}
            chunks[name] = P.denoise(model, obs, mask, kv, noise)[..., :REAL_DIMS]
        return acts, chunks

    def patched(models, obs, noise, acts, r):
        """Base model chunks with the fine-tuned (a) residual after r layers, (b) KV at layer r + 1."""
        obs = P.preprocess(obs)
        base, own, other = models["base"], acts["base"], acts["ft"]
        kv_a, _ = P.run_prefix(base, own["tokens"], own["mask"], patch_at=r, patch_x=other["residual"][r])
        kv_b = jax.tree.map(lambda mine, theirs: mine.at[r].set(theirs[r]), own["kv"], other["kv"])
        return tuple(P.denoise(base, obs, own["mask"], kv, noise)[..., :REAL_DIMS] for kv in (kv_a, kv_b))

    run_activations, run_patched = pair.jit(activations), pair.jit(patched)
    l_base, l_residual, l_kv = [], [], []
    for i, (rows, obs, noise) in enumerate(inputs.batches(args.batch_size)):
        acts, chunks = run_activations(obs, noise)
        target = np.asarray(chunks["ft"])
        loss = lambda chunk: np.square(np.asarray(chunk) - target).mean(axis=(1, 2))[: len(rows)]  # noqa: E731
        l_base.append(loss(chunks["base"]))
        per_layer = [[loss(c) for c in run_patched(obs, noise, acts, r)] for r in range(depth)]
        l_residual.append(np.stack([a for a, _ in per_layer], axis=1))
        l_kv.append(np.stack([b for _, b in per_layer], axis=1))
        print(f"batch {i + 1}: {rows[-1] + 1}/{len(inputs)} states", flush=True)
    losses = {"base": np.concatenate(l_base), "residual": np.concatenate(l_residual), "kv": np.concatenate(l_kv)}
    np.savez(DATA / "f4.npz", ids=[s["id"] for s in inputs.states], **losses)  # residual, kv: (states, 18)
    return losses


def summarize(losses: dict) -> dict:
    l_base = losses["base"]
    n, depth = losses["residual"].shape
    result = {"states": n, "L_base_mean": float(l_base.mean())}
    for kind in ("residual", "kv"):
        values, intervals = [], []
        for j in range(depth):
            patched = losses[kind][:, j]
            value = lambda rows, p=patched: float(1.0 - p[rows].mean() / l_base[rows].mean())  # noqa: E731
            values.append(value(np.arange(n)))
            intervals.append(bootstrap_ci(value, n, n_boot=1000))
        result[kind] = {"R": values, "ci": intervals}
    result["residual"]["x"] = list(range(depth))  # after r layers, r = 0..17
    result["kv"]["x"] = list(range(1, depth + 1))  # at layer l = 1..18
    return result


def plot(result: dict):
    plt = style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for kind, color, label in (
        ("residual", BASE_COLOR, "(a) residual stream after layer x (0 = token embeddings)"),
        ("kv", FT_COLOR, "(b) prefix keys and values of layer x only"),
    ):
        x, values, ci = result[kind]["x"], np.array(result[kind]["R"]), np.array(result[kind]["ci"])
        ax.plot(x, values, marker="o", color=color, label=label)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(0, len(result["kv"]["x"]) + 1))
    ax.set_xlabel("backbone layer x")
    ax.set_ylabel("recovery share R (fraction of the\nbase-to-fine-tuned action change)")
    ax.legend(loc="best", fontsize=11)
    kv = np.array(result["kv"]["R"])
    top = np.argsort(kv)[::-1][:3]
    residual = np.array(result["residual"]["R"])
    ax.set_title(
        f"No single layer's KV recovers more than {kv.max():.0%} (best: layers "
        f"{', '.join(str(result['kv']['x'][i]) for i in sorted(top))}); residual patch peaks at {residual.max():.0%}"
    )
    save_figure(fig, "F4", f"Layer-level activation patching, sufficiency direction: base run with one fine-tuned "
                f"activation swapped in, on {result['states']} LIBERO training states; bands are 95% bootstrap "
                "intervals over states. A residual patch changes the KV cache of all later layers (computed by "
                "base weights); a KV patch changes one layer only.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="use only the first N patch states (for a quick test)")
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/data/f4.npz")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        saved = np.load(DATA / "f4.npz")
        losses = {k: saved[k] for k in ("base", "residual", "kv")}
    else:
        losses = compute(args)
    result = summarize(losses)
    (DATA / "f4.json").write_text(json.dumps(result, indent=1))
    for kind in ("residual", "kv"):
        values = np.array(result[kind]["R"])
        print(f"{kind}: R per layer = {np.round(values, 3).tolist()}")
        if values.min() < -0.05 or values.max() > 1.05:
            print("  SURPRISING: R outside [0, 1]. Check the pairing rule and check_pair.py before reporting.")
    plot(result)
    if not args.plot_only:
        log_run("F4 (layer-level patching)", started, seed=SEED, states=result["states"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
