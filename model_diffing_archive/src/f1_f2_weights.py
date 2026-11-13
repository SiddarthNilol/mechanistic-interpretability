"""F1 (model map) and F2 (where weights moved). No GPU; needs about 40 GB of memory.

    python src/f1_f2_weights.py

Loads both checkpoints as numpy arrays and reads everything from the weights themselves.

F1: parameter counts of the vision encoder (SigLIP), the language backbone (Gemma 2B, with its
token embedding table), the action expert (Gemma 300M) and the rest (action and flow-time
projections). Also writes the architecture table for summary.md to results/data/f1_table.md.

F2: for every layer of each module, the relative weight change of fine-tuning,

    sqrt(sum_W ||W_ft - W_base||^2) / sqrt(sum_W ||W_base||^2)     W: the layer's weight matrices

(Frobenius norms, pooled over the layer's matrices; biases and norm scales are left out because
several start at zero in the base model). Weights that belong to no layer are shown next to it.
This is where weights moved, not which changes matter.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from model_diffing.archive.src.common import DATA, GRAY, SEED, log_run, save_figure, style

MODULES = ("vision encoder", "language backbone", "action expert", "other")


def module_of(path: tuple[str, ...]) -> str:
    if path[:2] == ("PaliGemma", "img"):
        return "vision encoder"
    if path[:2] == ("PaliGemma", "llm"):
        # openpi names the action expert's weights with the suffix _1
        return "action expert" if any(part.endswith("_1") for part in path) else "language backbone"
    return "other"


def is_layered(path: tuple[str, ...]) -> bool:
    """Weights stacked over layers (leading axis): SigLIP's encoder blocks and Gemma's layers."""
    return "encoderblock" in path or "layers" in path


def analyse(base: dict, ft: dict) -> dict:
    counts = dict.fromkeys(MODULES, 0)
    embedding_table = 0
    squares: dict[str, list] = {}  # module -> [sum of squared change per layer, sum of squared base per layer]
    single = {}  # weights outside the layers: name -> relative change
    for path, w in base.items():
        module = module_of(path)
        counts[module] += w.size
        if path[-1] == "input_embedding":
            embedding_table = w.size
        delta = ft[path].astype(np.float64) - w.astype(np.float64)
        if is_layered(path):
            if w.ndim < 3:  # per layer a vector: a bias or a norm scale
                continue
            axes = tuple(range(1, w.ndim))
            entry = squares.setdefault(module, [0.0, 0.0])
            entry[0] = entry[0] + np.square(delta).sum(axes)
            entry[1] = entry[1] + np.square(w.astype(np.float64)).sum(axes)
        elif w.ndim >= 2:
            name = "/".join(part for part in path if part not in ("PaliGemma", "kernel", "w"))
            single[name] = float(
                np.linalg.norm(delta) / np.linalg.norm(w.astype(np.float64))
            )
    change = {module: np.sqrt(num / den).tolist() for module, (num, den) in squares.items()}
    llm = {k: v for k, v in base.items() if k[:2] == ("PaliGemma", "llm")}
    q = llm[("PaliGemma", "llm", "layers", "attn", "q_einsum", "w")]
    q1 = llm[("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w")]
    vq = base[("PaliGemma", "img", "Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "query", "kernel")]
    return {
        "counts": counts,
        "embedding_table": embedding_table,
        "layer_change": change,
        "single_change": single,
        "architecture": {
            "language backbone": {"layers": q.shape[0], "heads": q.shape[1], "width": q.shape[2], "head_dim": q.shape[3]},
            "action expert": {"layers": q1.shape[0], "heads": q1.shape[1], "width": q1.shape[2], "head_dim": q1.shape[3]},
            "vision encoder": {"layers": vq.shape[0], "heads": vq.shape[2], "width": vq.shape[1], "head_dim": vq.shape[3]},
            "image_tokens_per_camera": base[("PaliGemma", "img", "pos_embedding")].shape[1],
            "action_dim": base[("action_in_proj", "kernel")].shape[0],
        },
    }


def write_table(result: dict):
    """The table for summary.md: layers, heads, width, prefix tokens per observation."""
    from model_diffing.archive.src.pair import model_config

    config = model_config()
    arch = result["architecture"]
    per_camera = arch["image_tokens_per_camera"]
    lines = ["| module | parameters | layers | heads per layer | width |", "|---|---|---|---|---|"]
    for module in MODULES[:3]:
        a = arch[module]
        lines.append(f"| {module} | {result['counts'][module] / 1e6:.0f} M | {a['layers']} | {a['heads']} | {a['width']} |")
    lines.append(f"| other (action and time projections) | {result['counts']['other'] / 1e6:.2f} M | | | |")
    lines += [
        "",
        f"The backbone's count includes its token embedding table ({result['embedding_table'] / 1e6:.0f} M).",
        f"Prefix tokens per observation: 3 camera slots x {per_camera} image tokens + {config.max_token_len} prompt "
        f"tokens = {3 * per_camera + config.max_token_len}. In LIBERO two cameras are real (the third slot is masked) "
        f"and the prompt is padded, so about {2 * per_camera} image tokens plus the prompt's own tokens are attended to.",
        f"Suffix: {config.action_horizon} action tokens, {arch['action_dim']} padded action dims, 7 real for LIBERO.",
    ]
    (DATA / "f1_table.md").write_text("\n".join(lines) + "\n")


def plot(result: dict):
    plt = style()
    counts = result["counts"]
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = [counts[m] / 1e6 for m in MODULES]
    bars = ax.bar(MODULES, values, color=["#55A868", "#4C72B0", "#C44E52", GRAY])
    for bar, module, value in zip(bars, MODULES, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.1f} M\n({counts[module] / total:.1%})",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("parameters (millions)")
    ax.set_ylim(0, max(values) * 1.2)
    share = counts["action expert"] / total
    ax.set_title(f"The action expert is {share:.0%} of pi0.5's {total / 1e9:.2f} B parameters")
    save_figure(fig, "F1", f"Parameter counts by module, from the pi05_base checkpoint. The language backbone "
                f"includes its token embedding table ({result['embedding_table'] / 1e6:.0f} M). Other: the action "
                f"input and output projections and the flow-time MLP.")
    plt.close(fig)

    change = result["layer_change"]
    rows = MODULES[:3]
    width = max(len(change[m]) for m in rows)
    grid = np.full((len(rows), width), np.nan)
    for i, module in enumerate(rows):
        grid[i, : len(change[module])] = change[module]
    single = dict(sorted(result["single_change"].items(), key=lambda kv: kv[1]))

    fig, (ax, side) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1.1]})
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#EEEEEE")
    image = ax.imshow(grid, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(rows)), [f"{m}\n({len(change[m])} layers)" for m in rows])
    ax.set_xticks(range(0, width, 2), range(1, width + 1, 2))
    ax.set_xlabel("layer")
    fig.colorbar(image, ax=ax, label="relative weight change\n||W_ft - W_base|| / ||W_base||")
    side.barh(list(single), list(single.values()), color=GRAY)
    side.set_xlabel("relative weight change")
    side.set_title("weights outside the layers", fontsize=11, fontweight="normal")
    side.tick_params(axis="y", labelsize=9)
    means = {m: float(np.mean(change[m])) for m in rows}
    top = max(means, key=means.get)
    low = min(means, key=means.get)
    ax.set_title(f"Fine-tuning moved the {top} most ({means[top]:.1%} per layer on average; {low}: {means[low]:.1%})")
    save_figure(fig, "F2", "Relative weight change of pi05_libero from pi05_base, per layer (Frobenius norms pooled "
                "over the layer's weight matrices; gray: the module has no such layer). This shows where weights "
                "moved, not which changes matter: it is the static view of Häon et al. and RETAIN that the causal "
                "pilots (F3, F4) go beyond.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/data/f1_f2.json")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "f1_f2.json"
    if not args.plot_only:
        from flax import traverse_util

        from model_diffing.archive.src.pair import load_params

        base, ft = (traverse_util.flatten_dict(load_params(name, on_cpu=True)) for name in ("base", "ft"))
        if base.keys() != ft.keys() or any(base[k].shape != ft[k].shape for k in base):
            raise RuntimeError("the two checkpoints do not have the same weights and shapes")
        out.write_text(json.dumps(analyse(base, ft), indent=1))
    result = json.loads(out.read_text())
    write_table(result)
    plot(result)
    if not args.plot_only:
        log_run("F1 + F2 (weights)", started, seed=f"{SEED} (not used: no sampling)", states="0 (weights only)")


if __name__ == "__main__":
    main()
