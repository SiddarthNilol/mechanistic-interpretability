"""The paper's head selection, following its released code (robotic_steering, head_tuning/knn).

    python analysis/select_heads.py --dataset $DATA --heads $SCRATCH/mechanistic-interpretability/pi05_base_denoised

Use it on heads extracted with extract_heads.py --denoised, which feeds the first action token
the way the paper's code does. Per task:

    1. For every head and every frame, find the 40 frames with the most similar head output
       (cosine distance), from the task's other episodes.
    2. For k in 10, 20, 30, 40: predict the frame's action as the average of its k neighbours'
       actions, weighted by 1 / distance. A head's score is the squared error, averaged over
       action dimensions and frames. Lower is better. (The paper's text says a plain mean; its
       code uses this weighted one.)
    3. For each k, take the 20 heads with the lowest scores, join their outputs into one long
       vector, and score that vector the same way. The k with the lowest joint score wins, and
       its 20 heads are the selected heads.

What we had to adapt, because the paper's robot has one arm and ours has two, a chest and a neck:

    Action      The paper's: its 7 commanded joint velocities + its gripper command. Ours: the same for
                the whole robot, 25 numbers. For the 23 joints (action[t + 0.1 s] - state[t]) / 0.1 s,
                and for the 2 grippers the command. Not scaled.

We use all frames of frames.csv (10 Hz). The paper's code by default keeps only keyframes (gripper
flips, arm almost still), and the paper says this doesn't change the outcome. On our data that
rule would leave almost no frames in the contact phases, so we don't apply it.

The paper selects from the action stream (first action token). The vlm stream (last prompt token)
is included only for the score spread, which the paper compares between the two (its table 4).

Writes to --heads:

    selection_scores.csv   stream, task, k, layer, head, score
    selected_heads.json    per stream and task: best k, joint score per k, the 20 heads
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import GRIPPER_DIM, column, read_episode_frames, read_episodes  # noqa: E402

K_VALUES = (10, 20, 30, 40)
TOP = 20
STEP_SECONDS = 0.1  # between our kept frames (every 5th at 50 fps)
STEP_FRAMES = 5
ACTION_DIM = 25
EPS = 1e-8  # the paper's code: guards divisions by zero
STREAMS = ("action", "vlm")


def read_actions(dataset: Path, frames: pd.DataFrame) -> np.ndarray:
    """For the rows of frames: the (rows, 25) action of the whole robot."""
    episodes = read_episodes(dataset)
    grippers = list(GRIPPER_DIM.values())
    actions = np.zeros((len(frames), ACTION_DIM), dtype=np.float32)
    for ep, kept in frames.groupby("episode_index", sort=False):
        data = read_episode_frames(dataset, episodes, ep)
        state, action = column(data, "observation.state"), column(data, "action")
        t = kept["frame_index"].to_numpy()
        later = np.minimum(t + STEP_FRAMES, len(data) - 1)
        target = (action[later] - state[t]) / STEP_SECONDS
        target[:, grippers] = action[t][:, grippers]
        actions[kept.index] = target
    return actions


def neighbours(x: np.ndarray, same_episode: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's k nearest frames from other episodes, nearest first, and their distances."""
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)
    distance = 1.0 - x @ x.T
    distance[same_episode] = np.inf
    nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(distance, nearest, axis=1), axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    return nearest, np.take_along_axis(distance, nearest, axis=1)


def score(nearest: np.ndarray, distance: np.ndarray, y: np.ndarray, k: int) -> float:
    """Squared error of the distance-weighted prediction from the k nearest frames, averaged
    over action dimensions and frames. A frame with an exact match copies that match's action."""
    weights = 1.0 / (distance[:, :k] + EPS)
    weights /= weights.sum(axis=1, keepdims=True) + EPS
    prediction = (weights[:, :, None] * y[nearest[:, :k]]).sum(axis=1)
    exact = distance[:, 0] < EPS
    prediction[exact] = y[nearest[exact, 0]]
    return float(((prediction - y) ** 2).mean())


def select(x: np.ndarray, y: np.ndarray, episode: np.ndarray) -> dict:
    """The selection for one task. x: (frames, heads, dim) head outputs, y: (frames, 25) actions."""
    same_episode = episode[:, None] == episode[None, :]
    k_max = max(K_VALUES)
    per_head = np.zeros((len(K_VALUES), x.shape[1]))
    for head in range(x.shape[1]):
        nearest, distance = neighbours(x[:, head], same_episode, k_max)
        per_head[:, head] = [score(nearest, distance, y, k) for k in K_VALUES]

    joint = {}
    for i, k in enumerate(K_VALUES):
        top = np.argsort(per_head[i], kind="stable")[:TOP]
        nearest, distance = neighbours(x[:, top].reshape(len(x), -1), same_episode, k)
        joint[k] = score(nearest, distance, y, k)
    best_k = min(K_VALUES, key=lambda k: joint[k])  # the smallest k wins a tie
    best = K_VALUES.index(best_k)
    top = np.argsort(per_head[best], kind="stable")[:TOP]
    return {"per_head": per_head, "joint": joint, "best_k": best_k, "top": top, "cv": per_head[best].std() / per_head[best].mean()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--heads", type=Path, required=True, help="output directory of extract_heads.py")
    args = parser.parse_args()

    frames = pd.read_csv(args.heads / "frames.csv")
    actions = read_actions(args.dataset, frames)

    rows, selected = [], []
    for stream in STREAMS:
        heads = np.load(args.heads / f"heads_{stream}.npy", mmap_mode="r")
        n_layers, n_heads = heads.shape[1:3]
        for task, table in frames.groupby("task"):
            rows_of_task = table.index.to_numpy()
            x = np.asarray(heads[rows_of_task], dtype=np.float32).reshape(len(table), n_layers * n_heads, -1)
            result = select(x, actions[rows_of_task], table["episode_index"].to_numpy())
            for i, k in enumerate(K_VALUES):
                rows += [
                    (stream, task, k, h // n_heads, h % n_heads, result["per_head"][i, h])
                    for h in range(n_layers * n_heads)
                ]
            selected.append(
                {
                    "stream": stream,
                    "task": task,
                    "n_frames": len(table),
                    "best_k": result["best_k"],
                    "joint_score": result["joint"],
                    "cv": round(float(result["cv"]), 3),
                    "heads": [[int(h) // n_heads, int(h) % n_heads] for h in result["top"]],
                }
            )
            print(f"done: {stream} {task}", flush=True)

    columns = ["stream", "task", "k", "layer", "head", "score"]
    pd.DataFrame(rows, columns=columns).round(6).to_csv(args.heads / "selection_scores.csv", index=False)
    (args.heads / "selected_heads.json").write_text(json.dumps(selected, indent=2))

    print("\nselection (the paper's uses the action stream; its score spread: 0.47 action, 0.05 vlm):")
    for s in selected:
        heads = " ".join(f"L{layer}H{head}" for layer, head in s["heads"])
        print(f"  {s['stream']:6} {s['task']:11} frames={s['n_frames']:4} k={s['best_k']} spread={s['cv']:.3f}\n    {heads}")

    if frames["task"].nunique() == 2:
        print("\ntop 20 shared between the tasks (%):")
        for stream in STREAMS:
            first, second = ({tuple(h) for h in s["heads"]} for s in selected if s["stream"] == stream)
            print(f"  {stream}: {100 * len(first & second) / TOP:.0f}")


if __name__ == "__main__":
    main()
