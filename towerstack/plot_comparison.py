"""Compare the tower stack selection with hook and unhook ring.

    python towerstack/plot_comparison.py --heads towerstack_denoised --reference pi05_base_denoised

Writes to <heads>/figures/:

    comparison_selected.png   the selected action-expert heads: tower stack's, the ring tasks', or both
    comparison_errors.png     every head's error on tower stack against its error on a ring task
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from plot_selection import BOTH, GRID, HEADS, HOOK, INK, LAYERS, MUTED, STREAMS, UNHOOK, chosen_scores, clean  # noqa: E402

TASK = "stack_rings"
RING_TASKS = ("hook ring", "unhook ring")


def load(heads: Path) -> tuple[pd.DataFrame, dict]:
    scores = pd.read_csv(heads / "selection_scores.csv")
    return scores, {(r["stream"], r["task"]): r for r in json.loads((heads / "selected_heads.json").read_text())}


def plot_selected(selection, reference, out: Path):
    tower = {tuple(h) for h in selection["action", TASK]["heads"]}
    ring = {tuple(h) for task in RING_TASKS for h in reference["action", task]["heads"]}
    grid = np.zeros((LAYERS, HEADS), dtype=int)
    for layer, head in tower:
        grid[layer, head] = 1
    for layer, head in ring:
        grid[layer, head] = 3 if (layer, head) in tower else 2

    fig, ax = plt.subplots(figsize=(4.6, 7), layout="constrained")
    ax.imshow(grid, cmap=ListedColormap(["#f1f0ec", HOOK, UNHOOK, BOTH]), vmin=0, vmax=3, aspect="auto")
    letters = {1: ("T", "white"), 2: ("R", INK), 3: ("B", INK)}  # identity doesn't depend on color alone
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
    ax.set_title("Selected action-expert heads across tasks", fontsize=10, color=INK, loc="left")
    fig.legend(
        handles=[
            Patch(color=HOOK, label=f"T: tower stack only ({len(tower - ring)})"),
            Patch(color=UNHOOK, label=f"R: ring tasks only ({len(ring - tower)})"),
            Patch(color=BOTH, label=f"B: both ({len(tower & ring)})"),
        ],
        loc="outside lower center", ncol=3, frameon=False, fontsize=8, handlelength=1, columnspacing=1,
    )
    fig.savefig(out / "comparison_selected.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_errors(scores, selection, ref_scores, reference, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), squeeze=False, layout="constrained")
    for row, stream in enumerate(STREAMS):
        tower = chosen_scores(scores, selection, stream, TASK).set_index(["layer", "head"])["score"]
        for col, task in enumerate(RING_TASKS):
            ring = chosen_scores(ref_scores, reference, stream, task).set_index(["layer", "head"])["score"].reindex(tower.index)
            ax = axes[row, col]
            ax.scatter(ring, tower, s=16, color="#3987e5", linewidths=0)
            correlation = tower.rank().corr(ring.rank())
            ax.set_title(f"{STREAMS[stream]}\nrank correlation with {task}: {correlation:.2f}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel(f"error on {task}", fontsize=8, color=MUTED)
            ax.set_ylabel("error on tower stack", fontsize=8, color=MUTED)
            clean(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("Do the same heads do well on both tasks? (one dot per head, lower is better)", fontsize=10, color=INK)
    fig.savefig(out / "comparison_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="the tower stack results")
    parser.add_argument("--reference", type=Path, required=True, help="the hook and unhook ring results")
    args = parser.parse_args()

    scores, selection = load(args.heads)
    ref_scores, reference = load(args.reference)
    out = args.heads / "figures"
    out.mkdir(exist_ok=True)
    plot_selected(selection, reference, out)
    plot_errors(scores, selection, ref_scores, reference, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
