"""Step 1: pick a seeded random subset of each task's episodes and record per-episode facts.

    python data_prep/select_episodes.py \
        --dataset /path/to/unhook-hook-data \
        --labels prepared/hook_unhook/labels.csv \
        --out prepared/hook_unhook

Picks --num episodes of every task (common.TASKS) from those not in
meta/excluded_episodes.json, and, when --labels is given, only from episodes it has a row
for. Writes <out>/episodes.csv, one row per chosen episode:

    episode_index  episode in the source dataset
    length         number of frames (50 fps)
    task           task string
    active_arm     arm that grasps the ring (left / right)
    n_grasps       open -> closed transitions of its gripper (>1 means a regrasp: check by hand)
    + every column of --labels (e.g. cycle, open_end, distractors), if given
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, active_arm, count_grasps, gripper, read_episode_frames, read_episodes, read_excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--labels", type=Path, help="optional per-episode CSV with an episode_index column")
    parser.add_argument("--num", type=int, default=20, help="episodes to pick per task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = read_episodes(args.dataset)
    tasks = pd.Series([str(t[0]) for t in episodes["tasks"]], index=episodes.index)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"tasks {sorted(unknown)} have no phase definition in common.TASKS")
    usable = tasks.drop(read_excluded(args.dataset))
    labels = None
    if args.labels is not None:
        labels = pd.read_csv(args.labels)
        usable = usable[usable.index.isin(labels["episode_index"])]

    rng = np.random.default_rng(args.seed)
    chosen = []
    for task in sorted(set(usable)):  # sorted, so the picks don't depend on dict order
        candidates = usable.index[usable == task].tolist()
        picked = sorted(rng.choice(candidates, size=args.num, replace=False).tolist())
        print(f"{task}: {(tasks == task).sum()} episodes, {len(candidates)} usable, picked {picked}")
        chosen += picked

    rows = []
    for ep in sorted(chosen):
        frames = read_episode_frames(args.dataset, episodes, ep)
        arm = active_arm(frames)
        rows.append(
            {
                "episode_index": ep,
                "length": int(episodes.loc[ep, "length"]),
                "task": tasks[ep],
                "active_arm": arm,
                "n_grasps": count_grasps(gripper(frames, arm)),
            }
        )
    table = pd.DataFrame(rows)
    if labels is not None:
        table = table.merge(labels, on="episode_index", how="left")

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "episodes.csv", index=False)
    print(table.to_string(index=False))
    print(f"wrote {args.out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
