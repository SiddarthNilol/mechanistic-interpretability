"""Figures for the paper's head selection, from the files of select_heads.py.

    python analysis/plot_selection.py --heads pi05_base_denoised

All errors are the action error of select_heads.py, at each task's chosen k. Lower is better.

Writes to <heads>/figures/:

    selection_heatmaps.png   one 18 x 8 map (layer x head) per task and model part. Darker = lower
                             error. The 20 lowest-error heads are outlined.
    selected_heads.png       which action-expert heads were selected for hook ring, unhook ring or both
    selection_layers.png     every head's error against its layer, with each layer's mean
    selection_k.png          the error of the 20 best heads joined, for each k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_heads import BLUES, INK, MUTED, draw_map, head_map  # noqa: E402

TASKS = ("hook ring", "unhook ring")
STREAMS = {"action": "action expert, first action token", "vlm": "VLM, last prompt token"}
LAYERS, HEADS = 18, 8
TOP = 20
# Clips the color scale: a few VLM layer 0 heads reach 0.47 and would wash out everything else.
MAP_MIN, MAP_MAX = 0.08, 0.14
HOOK, UNHOOK, BOTH = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#e4e3df"


def clean(ax):
    """Recessive axes: a light grid, no box."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)


def chosen_scores(scores: pd.DataFrame, selection: dict, stream: str, task: str) -> pd.DataFrame:
    """One row per head, at the k chosen for this stream and task."""
    k = selection[stream, task]["best_k"]
    return scores[(scores["stream"] == stream) & (scores["task"] == task) & (scores["k"] == k)]


def top_heads(table: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(table.nsmallest(TOP, "score")["layer"], table.nsmallest(TOP, "score")["head"]))


def plot_heatmaps(scores, selection, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.4), layout="constrained")
    for row, task in enumerate(TASKS):
        for col, stream in enumerate(STREAMS):
            t = chosen_scores(scores, selection, stream, task)
            best = t.loc[t["score"].idxmin()]
            title = (f"{task} · {STREAMS[stream]}\nbest L{best['layer']}H{best['head']}: {best['score']:.3f}, "
                     f"median {t['score'].median():.3f}, k={selection[stream, task]['best_k']}")
            image = draw_map(axes[row, col], -head_map(t, "score"), -MAP_MAX, -MAP_MIN, top_heads(t), title)
    bar = fig.colorbar(image, ax=axes, shrink=0.35, ticks=[-0.14, -0.12, -0.10, -0.08])
    bar.ax.set_yticklabels(["0.14 or more", "0.12", "0.10", "0.08"], fontsize=8, color=MUTED)
    bar.set_label("action error (darker = lower, better)", fontsize=8, color=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(f"How well each head predicts the action (outlined: the {TOP} lowest errors)", fontsize=10, color=INK)
    fig.savefig(out / "selection_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selected(selection, out: Path):
    """The selected heads themselves. The paper selects from the action expert only."""
    hook = {tuple(h) for h in selection["action", "hook ring"]["heads"]}
    unhook = {tuple(h) for h in selection["action", "unhook ring"]["heads"]}
    grid = np.zeros((LAYERS, HEADS), dtype=int)
    for layer, head in hook:
        grid[layer, head] = 1
    for layer, head in unhook:
        grid[layer, head] = 3 if (layer, head) in hook else 2

    fig, ax = plt.subplots(figsize=(4.6, 7), layout="constrained")
    ax.imshow(grid, cmap=ListedColormap(["#f1f0ec", HOOK, UNHOOK, BOTH]), vmin=0, vmax=3, aspect="auto")
    # The letters mean identity doesn't depend on color alone; each has the ink that reads on its fill.
    letters = {1: ("H", "white"), 2: ("U", INK), 3: ("B", INK)}
    for layer, head in np.argwhere(grid > 0):
        letter, ink = letters[grid[layer, head]]
        ax.text(head, layer, letter, ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(np.arange(-0.5, HEADS), minor=True)
    ax.set_yticks(np.arange(-0.5, LAYERS), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0, labelsize=8, colors=MUTED)
    ax.set_xticks(range(HEADS))
    ax.set_yticks(range(LAYERS))
    ax.set_xlabel("head", fontsize=8, color=MUTED)
    ax.set_ylabel("layer", fontsize=8, color=MUTED)
    for side in ax.spines.values():
        side.set_visible(False)
    shared = len(hook & unhook)
    ax.set_title(f"Selected action-expert heads: {shared} of {TOP} shared", fontsize=10, color=INK, loc="left")
    fig.legend(
        handles=[
            Patch(color=HOOK, label=f"H: hook ring only ({len(hook - unhook)})"),
            Patch(color=UNHOOK, label=f"U: unhook ring only ({len(unhook - hook)})"),
            Patch(color=BOTH, label=f"B: both ({shared})"),
        ],
        loc="outside lower center", ncol=3, frameon=False, fontsize=8, handlelength=1, columnspacing=1,
    )
    fig.savefig(out / "selected_heads.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layers(scores, selection, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, sharey="col", layout="constrained")
    for row, task in enumerate(TASKS):
        for col, stream in enumerate(STREAMS):
            ax = axes[row, col]
            t = chosen_scores(scores, selection, stream, task)
            mean = t.groupby("layer")["score"].mean()
            ax.scatter(t["layer"], t["score"], s=14, color="#86b6ef", linewidths=0, label="one head")
            ax.plot(mean.index, mean.to_numpy(), color="#0d366b", linewidth=2, label="layer mean")
            ax.set_title(f"{task} · {STREAMS[stream]}", fontsize=9, color=INK, loc="left")
            ax.set_xticks(range(0, LAYERS, 2))
            clean(ax)
            if row == 1:
                ax.set_xlabel("layer", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel("action error (lower is better)", fontsize=8, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Action error by layer", fontsize=10, color=INK)
    fig.savefig(out / "selection_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_k(selection, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, layout="constrained")
    for ax, stream in zip(axes, STREAMS):
        for task, color in zip(TASKS, (HOOK, UNHOOK)):
            row = selection[stream, task]
            ks = sorted(int(k) for k in row["joint_score"])
            values = [row["joint_score"][str(k)] for k in ks]
            ax.plot(ks, values, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor="white", markeredgewidth=2, label=task)
            ax.annotate(task, (ks[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 2, ks[-1] + 13)
        ax.set_xlabel("k (number of similar frames)", fontsize=8, color=MUTED)
        ax.set_title(STREAMS[stream], fontsize=9, color=INK, loc="left")
        clean(ax)
    axes[0].set_ylabel("error of the 20 best heads, joined", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Effect of k on the joined heads", fontsize=10, color=INK)
    fig.savefig(out / "selection_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="directory with selection_scores.csv and selected_heads.json")
    args = parser.parse_args()

    scores = pd.read_csv(args.heads / "selection_scores.csv")
    selection = {(r["stream"], r["task"]): r for r in json.loads((args.heads / "selected_heads.json").read_text())}
    out = args.heads / "figures"
    out.mkdir(exist_ok=True)

    plot_heatmaps(scores, selection, out)
    plot_selected(selection, out)
    plot_layers(scores, selection, out)
    plot_k(selection, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
