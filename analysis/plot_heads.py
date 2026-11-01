"""Step 4: head maps from score_heads.py.

    python analysis/plot_heads.py --heads $SCRATCH/mechanistic-interpretability/pi05_base

Writes to <heads>/figures/:

    heatmaps.png        per label (rows) and stream (columns), an 18 x 8 map (layer x head)
                        of score above control; the top --top heads are outlined
    phases_hook.png     per phase of the task and stream, each head's recall of that phase
    phases_unhook.png   (the share of the phase's frames its neighbours vote for correctly)
                        above the control's recall of it
    overlap.png         how many of the top heads two labels share
    top_heads.csv       the top heads per label and stream, best first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STREAMS = {"vlm": "PaliGemma heads, last prompt token", "action": "action expert heads, first action token"}
# one hue, light (near zero) to dark (high)
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f1f1e", "#6b6b66"


def head_map(table: pd.DataFrame, value: str) -> np.ndarray:
    """(layers, heads) array of one column of a table with one row per head."""
    return table.pivot(index="layer", columns="head", values=value).sort_index().to_numpy()


def draw_map(ax, values: np.ndarray, vmin: float, vmax: float, outline=(), title: str = "", axis_labels=True):
    """An 18 x 8 head map: layer 0 at the top, head 0 at the left."""
    image = ax.imshow(values, cmap=BLUES, vmin=vmin, vmax=vmax, aspect="auto")
    for layer, head in outline:
        ax.add_patch(Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
    ax.set_xticks(range(values.shape[1]))
    ax.set_yticks(range(0, values.shape[0], 2))
    ax.tick_params(length=0, labelsize=7, colors=MUTED)
    if axis_labels:
        ax.set_xlabel("head", fontsize=8, color=MUTED)
        ax.set_ylabel("layer", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for side in ax.spines.values():
        side.set_visible(False)
    return image


def top_heads(table: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    """The n heads with the highest score above control, best first."""
    best = table.assign(above=table["score"] - table["control"]).nlargest(n, "above")
    return list(zip(best["layer"], best["head"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="directory with scores.csv and recalls.csv")
    parser.add_argument("--top", type=int, default=20, help="heads per label to outline and compare (paper: 20)")
    args = parser.parse_args()

    scores = pd.read_csv(args.heads / "scores.csv")
    recalls = pd.read_csv(args.heads / "recalls.csv")
    out = args.heads / "figures"
    out.mkdir(exist_ok=True)
    labels = list(dict.fromkeys(scores["label"]))
    top = {(s, l): top_heads(t, args.top) for (s, l), t in scores.groupby(["stream", "label"])}

    # 1. score above control, per label and stream
    above = scores["score"] - scores["control"]
    vmax = max(float(above.max()), 0.05)
    fig, axes = plt.subplots(len(labels), 2, figsize=(7, 4.2 * len(labels)), squeeze=False, layout="constrained")
    for row, label in enumerate(labels):
        for col, stream in enumerate(STREAMS):
            t = scores[(scores["label"] == label) & (scores["stream"] == stream)]
            best = t.loc[t["score"].idxmax()]
            title = f"{label} · {stream}\nbest L{best['layer']}H{best['head']}: {best['score']:.2f} (control {best['control']:.2f})"
            image = draw_map(axes[row, col], head_map(t.assign(a=t["score"] - t["control"]), "a"), 0, vmax,
                             top[stream, label], title)
    fig.colorbar(image, ax=axes, shrink=0.3, label="balanced accuracy above control")
    fig.suptitle(f"Head maps (outlined: top {args.top})\n" + "; ".join(f"{s}: {d}" for s, d in STREAMS.items()),
                 fontsize=9, color=INK)
    fig.savefig(out / "heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. per-phase recall, per task
    for label in [l for l in labels if l.startswith("phase_")]:
        r = recalls[recalls["label"] == label]
        phases = list(dict.fromkeys(r["class"]))
        r = r.assign(above=r["recall"] - r["control"])
        vmax = max(float(r["above"].max()), 0.05)
        fig, axes = plt.subplots(2, len(phases), figsize=(2.2 * len(phases), 8), squeeze=False, layout="constrained")
        for row, stream in enumerate(STREAMS):
            for col, phase in enumerate(phases):
                t = r[(r["stream"] == stream) & (r["class"] == phase)]
                image = draw_map(axes[row, col], head_map(t, "above"), 0, vmax, title=f"{phase} · {stream}",
                                 axis_labels=col == 0)
        fig.colorbar(image, ax=axes, shrink=0.3, label="recall of the phase above control")
        fig.suptitle(f"{label}: which heads recognize each phase", fontsize=10, color=INK)
        fig.savefig(out / f"{label.replace('phase_', 'phases_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. overlap of the top heads between labels (within a stream; the streams are different heads)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, stream in zip(axes, STREAMS):
        shared = np.array([[len(set(top[stream, a]) & set(top[stream, b])) for b in labels] for a in labels])
        ax.imshow(shared, cmap=BLUES, vmin=0, vmax=args.top)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, shared[i, j], ha="center", va="center", fontsize=8,
                        color="white" if shared[i, j] > args.top / 2 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_title(f"{stream}: top-{args.top} heads in common", fontsize=9, color=INK, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
    fig.savefig(out / "overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = [
        {"stream": s, "label": l, "rank": i + 1, "layer": layer, "head": head}
        for (s, l), heads in top.items()
        for i, (layer, head) in enumerate(heads)
    ]
    table = pd.DataFrame(rows).merge(scores, on=["stream", "label", "layer", "head"])
    table.to_csv(out / "top_heads.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
