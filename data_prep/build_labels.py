"""Step 4: per-frame labels for the chosen episodes, from the (hand-checked) phase boundaries.

    python data_prep/build_labels.py --dataset /path/to/dataset --out prepared/hook_unhook

annotate.py runs this automatically once every episode is reviewed; run it by hand only to
rebuild frames.csv (e.g. with another --stride).

Reads <out>/episodes.csv and <out>/phase_boundaries.csv (every boundary, including t_pipe,
must be filled in) and writes <out>/frames.csv with one row for every --stride-th frame:

    episode_index  episode in the source dataset
    frame_index    frame within the episode
    index          global frame index in the source dataset
    timestamp      seconds since the episode start
    task           task string (the task-identity label)
    phase          the task's phase (see common.TASKS): approach / align / grasp /
                   unhook / carry / retract, or ... carry / hook / retract for hook ring
    phase_id       position of the phase in its task's list (0-5)
    grasped        1 while the ring is held (t_closed <= frame < t_release)
    ring_on_pipe   1 while the ring is on the pipe (before t_pipe for unhook, from t_pipe on for hook)
    active_arm     arm that grasps the ring (left / right)
    gripper        active arm's gripper state (1 open, 0 closed), for reference
    + the per-episode columns of episodes.csv (e.g. open_end, distractors)

The extraction step runs the model on exactly these frames, and every score is computed
against these columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, check_boundaries, gripper, phase_ids, read_episode_frames, read_episodes


def build_labels(dataset: Path, out: Path, stride: int = 5) -> pd.DataFrame:
    """Write <out>/frames.csv and return it."""
    chosen = pd.read_csv(out / "episodes.csv").set_index("episode_index")
    boundaries = pd.read_csv(out / "phase_boundaries.csv").set_index("episode_index")
    if set(chosen.index) != set(boundaries.index):
        raise ValueError("episodes.csv and phase_boundaries.csv list different episodes")
    episodes = read_episodes(dataset)
    per_episode = [c for c in chosen.columns if c not in ("length", "n_grasps")]

    tables = []
    for ep in chosen.index:
        b = boundaries.loc[ep]
        length = int(chosen.loc[ep, "length"])
        task = TASKS[chosen.loc[ep, "task"]]
        check_boundaries(ep, length, b)
        frames = read_episode_frames(dataset, episodes, ep)
        phase = phase_ids(length, b)
        keep = np.arange(0, length, stride)

        table = pd.DataFrame(
            {
                "episode_index": ep,
                "frame_index": keep,
                "index": frames["index"].to_numpy()[keep],
                "timestamp": frames["timestamp"].to_numpy()[keep],
                "phase": [task["phases"][i] for i in phase[keep]],
                "phase_id": phase[keep],
                "grasped": ((keep >= b["t_closed"]) & (keep < b["t_release"])).astype(int),
                "ring_on_pipe": ((keep < b["t_pipe"]) == task["starts_on_pipe"]).astype(int),
                "gripper": gripper(frames, chosen.loc[ep, "active_arm"])[keep].round(4),
            }
        )
        for c in per_episode:
            table[c] = chosen.loc[ep, c]
        tables.append(table)

    frames_table = pd.concat(tables, ignore_index=True)
    first = ["episode_index", "frame_index", "index", "timestamp", "task", "phase", "phase_id", "grasped", "ring_on_pipe"]
    frames_table = frames_table[first + [c for c in frames_table.columns if c not in first]]
    frames_table.to_csv(out / "frames.csv", index=False)
    return frames_table


def summary(frames_table: pd.DataFrame) -> str:
    """Frame counts per phase and per label value."""
    lines = [f"{len(frames_table)} frames from {frames_table['episode_index'].nunique()} episodes"]
    for task, frames in frames_table.groupby("task"):
        counts = frames["phase"].value_counts().reindex(TASKS[task]["phases"])
        lines.append(f"{task} ({frames['episode_index'].nunique()} episodes): " + ", ".join(f"{p} {n}" for p, n in counts.items()))
    skip = {"episode_index", "frame_index", "index", "timestamp", "phase", "phase_id", "gripper", "cycle"}
    for c in [c for c in frames_table.columns if c not in skip]:
        lines.append(f"{c}: {frames_table[c].value_counts(dropna=False).to_dict()}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--out", type=Path, required=True, help="directory holding episodes.csv")
    parser.add_argument("--stride", type=int, default=5, help="keep every stride-th frame (5 = 10 Hz at 50 fps)")
    args = parser.parse_args()

    frames_table = build_labels(args.dataset, args.out, args.stride)
    print(summary(frames_table))
    print(f"wrote {args.out / 'frames.csv'}")


if __name__ == "__main__":
    main()
