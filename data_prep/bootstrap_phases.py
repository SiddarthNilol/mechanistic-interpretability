"""Step 2: bootstrap phase boundaries for the chosen episodes.

    python data_prep/bootstrap_phases.py --dataset /path/to/dataset --out prepared/hook_unhook

Reads <out>/episodes.csv and writes two files with one row per episode:

    phase_boundaries_auto.csv  what the rules found (kept as a record; don't edit)
    phase_boundaries.csv       the same, to correct by hand and to fill in t_pipe
                               (annotate.py is the GUI for this)

Columns: episode_index, length, t_align, t_grasp, t_closed, t_pipe (empty), t_release, note,
reviewed (0 until the episode is saved in annotate.py).
Boundaries are frame indices within the episode; see common.py for the phase definitions.
The note column flags episodes worth a closer look (e.g. regrasps).

phase_boundaries.csv is never overwritten unless --overwrite is given, so hand edits are safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import BOUNDARIES, bootstrap_boundaries, ee_position, gripper, read_episode_frames, read_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--out", type=Path, required=True, help="directory holding episodes.csv")
    parser.add_argument("--overwrite", action="store_true", help="replace a hand-edited phase_boundaries.csv")
    args = parser.parse_args()

    target = args.out / "phase_boundaries.csv"
    if target.exists() and not args.overwrite:
        raise SystemExit(f"{target} exists (it may hold hand edits); pass --overwrite to replace it")

    chosen = pd.read_csv(args.out / "episodes.csv")
    episodes = read_episodes(args.dataset)

    rows = []
    for ep, arm, n_grasps in chosen[["episode_index", "active_arm", "n_grasps"]].itertuples(index=False):
        frames = read_episode_frames(args.dataset, episodes, ep)
        found = bootstrap_boundaries(gripper(frames, arm), ee_position(frames, arm))
        notes = []
        if n_grasps > 1:
            notes.append(f"{n_grasps} grasps: boundaries use the longest closure, check by hand")
        if found["t_release"] == len(frames):
            notes.append("gripper never reopens")
        rows.append(
            {"episode_index": ep, "length": len(frames), **found, "t_pipe": pd.NA, "note": "; ".join(notes), "reviewed": 0}
        )

    table = pd.DataFrame(rows)[["episode_index", "length", *BOUNDARIES, "note", "reviewed"]]
    table.to_csv(args.out / "phase_boundaries_auto.csv", index=False)
    table.to_csv(target, index=False)
    print(table.to_string(index=False))
    print(f"wrote {target} (fill in t_pipe, correct the rest if needed) and phase_boundaries_auto.csv")


if __name__ == "__main__":
    main()
