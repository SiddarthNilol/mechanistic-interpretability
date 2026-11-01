"""Step 3: how well each head's output tells apart the task, the phases and the predicates.

    python analysis/score_heads.py --heads $SCRATCH/mechanistic-interpretability/pi05_base

For every head (2 streams x 18 layers x 8 heads) and every label, a k-nearest-neighbour
classifier as in the paper: each frame is predicted from the k frames whose head output
is most similar to its own (cosine similarity), by majority vote of their labels.
Neighbours never come from the frame's own cycle (the unhook/hook episode pair, which
share a scene), so a head can't score by recognizing the episode.

    task           hook ring vs unhook ring, over all frames
    phase_hook     the 6 phases, among hook ring frames only (neighbours too)
    phase_unhook   the 6 phases, among unhook ring frames only
    grasped        ring held or not, over all frames
    ring_on_pipe   ring on the pipe or not, over all frames

The score is balanced accuracy (the mean of the per-class recalls), so a head that always
predicts the most common class scores 1 / number of classes.

Control: the same neighbours, with the labels of each episode replaced by those of another
episode (of the same task, for the phase labels), matched by progress through the episode.
The labels keep their proportions and their order in time but no longer belong to the
frames, so a head that only tracks time since the episode start, or that scores by chance,
does as well on the control. Averaged over --controls shuffles.

Writes to --heads:

    scores.csv     stream, layer, head, label, score, control
    recalls.csv    stream, layer, head, label, class, recall, control (for the per-phase maps)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import TASKS  # noqa: E402

# label name -> (frames.csv column, task whose frames it is scored on or None for all)
LABELS = {
    "task": ("task", None),
    "phase_hook": ("phase", "hook ring"),
    "phase_unhook": ("phase", "unhook ring"),
    "grasped": ("grasped", None),
    "ring_on_pipe": ("ring_on_pipe", None),
}
STREAMS = ["vlm", "action"]


def neighbours(x: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """For each row of x (frames, dim), the indices of the k rows with the highest cosine
    similarity to it, among the rows of other groups. Returns (frames, k)."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    similarity[groups[:, None] == groups[None, :]] = -np.inf  # also excludes the frame itself
    return np.argpartition(-similarity, k, axis=1)[:, :k]


def vote(nbrs: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """The most common label among each frame's neighbours (ties go to the lower class)."""
    counts = np.zeros((len(nbrs), n_classes))
    for j in range(nbrs.shape[1]):
        counts[np.arange(len(nbrs)), y[nbrs[:, j]]] += 1
    return counts.argmax(axis=1)


def recalls(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Per class, the share of its frames predicted as that class."""
    return np.array([np.mean(pred[y == c] == c) for c in range(n_classes)])


def shuffled_labels(y: np.ndarray, episode: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """y with every episode's labels replaced by those of another episode, taken at the
    same progress (0 at its first kept frame, 1 at its last). Frames must be in order
    within each episode."""
    episodes = rng.permutation(np.unique(episode))
    donor = dict(zip(episodes, np.roll(episodes, 1)))  # a cycle: no episode keeps its own labels
    out = np.empty_like(y)
    for ep, d in donor.items():
        mine, theirs = np.flatnonzero(episode == ep), np.flatnonzero(episode == d)
        progress = np.linspace(0, 1, len(mine))
        out[mine] = y[theirs][np.round(progress * (len(theirs) - 1)).astype(int)]
    return out


def encode(values: pd.Series, column: str, task: str | None) -> tuple[np.ndarray, list[str]]:
    """Labels as integers 0..C-1, and the class names (phases in their task's order)."""
    classes = TASKS[task]["phases"] if column == "phase" else sorted(values.astype(str).unique())
    lookup = {c: i for i, c in enumerate(classes)}
    return values.astype(str).map(lookup).to_numpy(), classes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="output directory of extract_heads.py")
    parser.add_argument("--k", type=int, default=20, help="neighbours per frame (the paper tries 10-40)")
    parser.add_argument("--controls", type=int, default=3, help="label shuffles to average the control over")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = pd.read_csv(args.heads / "frames.csv")
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable")  # order within episodes
    order = frames.index.to_numpy()  # rows of the head arrays
    frames = frames.reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    # Per label: the frames it is scored on, the true labels, and the shuffled controls.
    labels = {}
    for name, (column, task) in LABELS.items():
        rows = np.flatnonzero(frames["task"] == task) if task else np.arange(len(frames))
        y, classes = encode(frames[column].iloc[rows], column, task)
        episode = frames["episode_index"].to_numpy()[rows]
        controls = [shuffled_labels(y, episode, rng) for _ in range(args.controls)]
        labels[name] = {"rows": rows, "task": task, "y": y, "classes": classes, "controls": controls}

    scores, class_recalls = [], []
    for stream in STREAMS:
        heads = np.load(args.heads / f"heads_{stream}.npy", mmap_mode="r")
        n_layers, n_heads = heads.shape[1:3]
        for layer in range(n_layers):
            x_layer = np.asarray(heads[:, layer], dtype=np.float32)[order]  # (frames, heads, dim)
            for head in range(n_heads):
                x = x_layer[:, head]
                # one neighbour search per frame subset (all frames, each task's frames)
                found = {}
                for name, label in labels.items():
                    if label["task"] not in found:
                        rows = label["rows"]
                        found[label["task"]] = neighbours(x[rows], frames["cycle"].to_numpy()[rows], args.k)
                    nbrs, y, n_classes = found[label["task"]], label["y"], len(label["classes"])
                    r = recalls(y, vote(nbrs, y, n_classes), n_classes)
                    control = np.mean([recalls(c, vote(nbrs, c, n_classes), n_classes) for c in label["controls"]], axis=0)
                    scores.append((stream, layer, head, name, r.mean(), control.mean()))
                    class_recalls += [
                        (stream, layer, head, name, c, v, cv) for c, v, cv in zip(label["classes"], r, control)
                    ]
            print(f"{stream} layer {layer}: done", flush=True)

    scores = pd.DataFrame(scores, columns=["stream", "layer", "head", "label", "score", "control"])
    scores.round(4).to_csv(args.heads / "scores.csv", index=False)
    pd.DataFrame(class_recalls, columns=["stream", "layer", "head", "label", "class", "recall", "control"]).round(4).to_csv(
        args.heads / "recalls.csv", index=False
    )

    print(f"\nwrote {args.heads / 'scores.csv'}\nbest head per label (score / control):")
    for (stream, name), s in scores.groupby(["stream", "label"], sort=False):
        best = s.loc[s["score"].idxmax()]
        print(f"  {stream:6s} {name:13s} L{best['layer']:02d}H{best['head']}  {best['score']:.3f} / {best['control']:.3f}")


if __name__ == "__main__":
    main()
