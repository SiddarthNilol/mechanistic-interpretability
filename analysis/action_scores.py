"""The paper's head score: how well a head's output retrieves the action (eq. 2-4).

    python analysis/action_scores.py --dataset $DATA --heads $SCRATCH/mechanistic-interpretability/pi05_base

As in the paper, heads are scored separately per task, on that task's 20 episodes. For every
frame, the k frames with the most similar head output (cosine similarity) are found in the
task's other episodes, and the mean of their actions is the prediction. The score of a head
is the squared error of that prediction, summed over the action dimensions and averaged
over frames. LOWER IS BETTER. All k of {10, 20, 30, 40} are scored; action_report.py picks
one the way the paper does.

The paper's actions are joint velocities; ours are joint position commands. So two targets,
each scaled like openpi does (1st / 99th percentile of each dimension over the dataset to
-1 / 1) so that no joint dominates the error:

    position   the action: the joint positions commanded at this frame
    motion     how far the command moves the joints over the next 0.1 s, one step at our 10 Hz
               (action[t + 5] - state[t]): the closest we have to the paper's velocities

Control: the same neighbours, predicting the actions of another episode at the same progress
(as in score_heads.py). A head that only tracks time since the episode start does as well on
the control. The borrowed actions are rescaled to the spread of the task's real actions:
otherwise the control's error differs from the score even for a head of pure noise, because
long and short episodes swap frames. Averaged over --controls shuffles.

Splits, to test whether the ranking of heads can be repeated: --splits times, the task's
episodes are randomly divided into two halves with no episode in common, and each half is
scored on its own (neighbours within the half, k = --split-k).

Scores are also broken down by phase (the same predictions, averaged over the phase's frames).

Writes to --heads (phase "all" = all the task's frames):

    action_scores.csv   stream, layer, head, task, target, k, phase, score, control
    action_splits.csv   stream, layer, head, task, target, split, half, phase, score
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import read_excluded  # noqa: E402
from score_heads import STREAMS, shuffled_labels  # noqa: E402

K_VALUES = (10, 20, 30, 40)  # the paper's search
MOTION_STEP = 5  # frames at 50 fps: 0.1 s, the step between our kept frames
TARGETS = ("position", "motion")


def scale(x: np.ndarray) -> np.ndarray:
    """1st / 99th percentile of each dimension to -1 / 1, as openpi normalizes actions."""
    q01, q99 = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def read_targets(dataset: Path, frames: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per target, the scaled (frames, 25) array for the rows of frames. The scaling uses
    all frames of the non-excluded episodes."""
    excluded = set(read_excluded(dataset))
    columns = ["episode_index", "frame_index", "action", "observation.state"]
    data = pd.concat(pq.read_table(p, columns=columns).to_pandas() for p in sorted((dataset / "data").glob("*/*.parquet")))
    data = data[~data["episode_index"].isin(excluded)].sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    action = np.stack(data["action"].to_numpy()).astype(np.float64)
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    episode = data["episode_index"].to_numpy()
    last_row = np.searchsorted(episode, episode, side="right") - 1  # of each row's episode
    later = np.minimum(np.arange(len(data)) + MOTION_STEP, last_row)
    targets = {"position": scale(action), "motion": scale(action[later] - state)}

    rows = pd.MultiIndex.from_frame(data[["episode_index", "frame_index"]]).get_indexer(
        pd.MultiIndex.from_frame(frames[["episode_index", "frame_index"]])
    )
    if (rows < 0).any():
        raise ValueError("frames.csv has frames that are not in the dataset (or are in excluded episodes)")
    return {name: t[rows].astype(np.float32) for name, t in targets.items()}


def borrowed_actions(target: np.ndarray, episode: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Actions of another episode at the same progress, with the same total variance as
    target. Errors are quadratic in the target, so this scale also makes errors comparable."""
    borrowed = shuffled_labels(target, episode, rng)
    return borrowed * np.sqrt(target.var(axis=0).sum() / borrowed.var(axis=0).sum())


def similarity_to_others(x: np.ndarray, same_episode: np.ndarray) -> np.ndarray:
    """Cosine similarity of all rows of x (frames, dim), with frames of the same episode
    excluded by setting theirs to -inf (the paper's "all other trajectories")."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    similarity = x @ x.T
    similarity[same_episode] = -np.inf
    return similarity


def nearest(similarity: np.ndarray, k: int) -> np.ndarray:
    """(frames, k) indices of each frame's k most similar frames, most similar first."""
    top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def frame_errors(neighbours: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Eq. 3-4 per frame: squared error of the mean action of the k nearest neighbours."""
    prediction = target[neighbours[:, :k]].mean(axis=1)
    return ((prediction - target) ** 2).sum(axis=1)


def by_phase(errors: np.ndarray, phase_rows: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean error over each phase's frames (phase "all" = every frame)."""
    return {name: float(errors[rows].mean()) for name, rows in phase_rows.items() if len(rows)}


def phase_rows_of(phases: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.arange(len(phases)), **{p: np.flatnonzero(phases == p) for p in np.unique(phases)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--heads", type=Path, required=True, help="output directory of extract_heads.py")
    parser.add_argument("--controls", type=int, default=3, help="shuffles to average the control over")
    parser.add_argument("--splits", type=int, default=10, help="random divisions of each task's episodes in two")
    parser.add_argument("--split-k", type=int, default=20, help="neighbours per frame within a half")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = pd.read_csv(args.heads / "frames.csv")
    targets = read_targets(args.dataset, frames)
    rng = np.random.default_rng(args.seed)

    # What stays the same for all heads, per task: its frames, the controls, the splits.
    tasks = {}
    for task, table in frames.groupby("task"):
        rows = table.index.to_numpy()
        episode = table["episode_index"].to_numpy()
        phases = table["phase"].to_numpy()
        splits = []
        for _ in range(args.splits):
            shuffled = rng.permutation(np.unique(episode))
            half_a, half_b = np.array_split(shuffled, 2)
            for half, members in (("a", half_a), ("b", half_b)):
                inside = np.flatnonzero(np.isin(episode, members))
                splits.append((len(splits) // 2, half, inside, phase_rows_of(phases[inside])))
        tasks[task] = {
            "rows": rows,
            "same_episode": episode[:, None] == episode[None, :],
            "phase_rows": phase_rows_of(phases),
            "splits": splits,
            "targets": {name: t[rows] for name, t in targets.items()},
            "controls": {
                name: [borrowed_actions(t[rows], episode, rng) for _ in range(args.controls)]
                for name, t in targets.items()
            },
        }

    scores, split_scores = [], []
    for stream in STREAMS:
        heads = np.load(args.heads / f"heads_{stream}.npy", mmap_mode="r")
        n_layers, n_heads = heads.shape[1:3]
        for layer in range(n_layers):
            x_layer = np.asarray(heads[:, layer], dtype=np.float32)  # (frames, heads, dim)
            for head in range(n_heads):
                for task, t in tasks.items():
                    similarity = similarity_to_others(x_layer[t["rows"], head], t["same_episode"])
                    neighbours = nearest(similarity, max(K_VALUES))
                    for name, target in t["targets"].items():
                        for k in K_VALUES:
                            score = by_phase(frame_errors(neighbours, target, k), t["phase_rows"])
                            control = [
                                by_phase(frame_errors(neighbours, c, k), t["phase_rows"]) for c in t["controls"][name]
                            ]
                            scores += [
                                (stream, layer, head, task, name, k, phase, value, np.mean([c[phase] for c in control]))
                                for phase, value in score.items()
                            ]
                    for split, half, inside, phase_rows in t["splits"]:
                        nearest_in_half = nearest(similarity[np.ix_(inside, inside)], args.split_k)
                        for name, target in t["targets"].items():
                            score = by_phase(frame_errors(nearest_in_half, target[inside], args.split_k), phase_rows)
                            split_scores += [
                                (stream, layer, head, task, name, split, half, phase, value)
                                for phase, value in score.items()
                            ]
            print(f"{stream} layer {layer}: done", flush=True)

    columns = ["stream", "layer", "head", "task", "target", "k", "phase", "score", "control"]
    pd.DataFrame(scores, columns=columns).round(5).to_csv(args.heads / "action_scores.csv", index=False)
    columns = ["stream", "layer", "head", "task", "target", "split", "half", "phase", "score"]
    pd.DataFrame(split_scores, columns=columns).round(5).to_csv(args.heads / "action_splits.csv", index=False)
    print(f"wrote {args.heads / 'action_scores.csv'} and {args.heads / 'action_splits.csv'}")


if __name__ == "__main__":
    main()
