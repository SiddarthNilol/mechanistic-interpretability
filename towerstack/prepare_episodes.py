"""Pick random episodes of one task, for the paper's head selection on a new task.

    python towerstack/prepare_episodes.py --dataset /path/to/towerstack-data

No phases or other labels are needed, so unlike data_prep/ nothing is annotated by hand. Picks --num
episodes from those not in meta/excluded_episodes.json and keeps every 5th frame, as for hook and
unhook ring. If the dataset has several task strings, --task says which one to use.

select_heads.py predicts one arm's action per episode, and only one arm grasps in each episode. That
arm is taken to be the one whose commanded joints move furthest over the episode. common.active_arm
(the gripper that closes furthest) labels this dataset wrongly, so it isn't used here. Check the
printed arms against the videos: a travel ratio near 1 means the two arms moved about equally, and
such a label is the least certain.

Writes to --out:

    episodes.csv   episode_index, length, task, active_arm, travel_left, travel_right (radians)
    frames.csv     episode_index, frame_index, task, active_arm: the input of extract_heads.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import GRIPPER_DIM, column, read_episode_frames, read_episodes, read_excluded, read_info  # noqa: E402

STRIDE = 5
FPS = 50  # select_heads.py and every earlier result assume 5 frames = 0.1 s


def choose_task(tasks: pd.Series, task: str | None) -> str:
    """The requested task, or the only one. Stops if the choice isn't clear, listing the options."""
    counts = tasks.value_counts()
    if task is None and len(counts) == 1:
        return counts.index[0]
    if task in counts.index:
        return task
    raise ValueError(f"choose one with --task. Usable episodes per task:\n{counts.to_string()}")


def arm_travel(frames: pd.DataFrame) -> dict[str, float]:
    """Per arm, how far its 7 commanded joints move over the episode, added up."""
    action = column(frames, "action")
    return {
        arm: float(np.abs(np.diff(action[:, gripper - 7 : gripper], axis=0)).sum())
        for arm, gripper in GRIPPER_DIM.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "prepared")
    parser.add_argument("--task", help="task string to use, if the dataset has several")
    parser.add_argument("--num", type=int, default=20, help="episodes to pick")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fps = read_info(args.dataset)["fps"]
    if fps != FPS:
        raise ValueError(f"{args.dataset} has {fps} fps; the 0.1 s step in select_heads.py assumes {FPS}")

    episodes = read_episodes(args.dataset)
    tasks = pd.Series([str(t[0]) for t in episodes["tasks"]], index=episodes.index)
    usable = tasks.drop(read_excluded(args.dataset))
    task = choose_task(usable, args.task)
    candidates = usable.index[usable == task].tolist()
    picked = sorted(np.random.default_rng(args.seed).choice(candidates, size=args.num, replace=False).tolist())
    print(f"{task!r}: {(tasks == task).sum()} episodes, {len(candidates)} usable, picked {picked}")

    episode_rows, frame_rows = [], []
    for ep in picked:
        frames = read_episode_frames(args.dataset, episodes, ep)
        travel = arm_travel(frames)
        arm = max(travel, key=travel.get)
        episode_rows.append(
            {"episode_index": ep, "length": len(frames), "task": task, "active_arm": arm,
             "travel_left": round(travel["left"], 2), "travel_right": round(travel["right"], 2)}
        )
        kept = np.arange(0, len(frames), STRIDE)
        frame_rows.append(pd.DataFrame({"episode_index": ep, "frame_index": kept, "task": task, "active_arm": arm}))

    table = pd.DataFrame(episode_rows)
    frames_table = pd.concat(frame_rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "episodes.csv", index=False)
    frames_table.to_csv(args.out / "frames.csv", index=False)

    ratio = table[["travel_left", "travel_right"]].max(axis=1) / table[["travel_left", "travel_right"]].min(axis=1)
    print(table.assign(travel_ratio=ratio.round(1)).to_string(index=False))
    print(f"{len(frames_table)} frames; arms: {table['active_arm'].value_counts().to_dict()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
