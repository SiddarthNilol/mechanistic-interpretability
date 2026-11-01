"""Step 0: per-episode labels for the hook/unhook dataset, from the unhook-only labels.csv.

    python data_prep/cycle_labels.py \
        --labels /path/to/lair-train-pipeline/labels.csv \
        --dataset /path/to/unhook-hook-data \
        --out prepared/hook_unhook/labels.csv

The hook/unhook dataset records cycles: episode 2k unhooks the ring, episode 2k+1 hooks it
back, starting exactly where 2k ended (same pipe orientation, same distractors). Its unhook
episodes are the unhook-ring-sept25-data episodes, in order: sept25 episode k is episode 2k
here, with identical data. So a label of sept25 episode k holds for both 2k and 2k+1.

labels.csv is numbered like sept25. This writes one row per labelled episode of the
hook/unhook dataset, with a `cycle` column (= k) for grouping the two halves of a cycle.
Columns that are empty in labels.csv are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="labels.csv numbered like unhook-ring-sept25-data")
    parser.add_argument("--dataset", type=Path, required=True, help="the hook/unhook LeRobot v3.0 dataset")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    args = parser.parse_args()

    tasks = [str(t[0]) for t in read_episodes(args.dataset)["tasks"]]
    expected = ["unhook ring", "hook ring"] * (len(tasks) // 2)
    if tasks != expected:
        raise ValueError(f"{args.dataset} doesn't alternate unhook ring / hook ring episodes")

    labels = pd.read_csv(args.labels).dropna(axis="columns", how="all")
    labels = labels.rename(columns={"episode_index": "cycle"})
    unhook = labels.assign(episode_index=2 * labels["cycle"])
    hook = labels.assign(episode_index=2 * labels["cycle"] + 1)
    out = pd.concat([unhook, hook]).sort_values("episode_index")
    out = out[["episode_index", *labels.columns]]
    out = out[out["episode_index"] < len(tasks)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(labels)} cycles -> {len(out)} episodes; wrote {args.out}")


if __name__ == "__main__":
    main()
