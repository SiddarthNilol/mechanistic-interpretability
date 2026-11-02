"""Do heads prefer a task or a phase? A closer look at test 5 of action_report.py.

    python analysis/action_specificity.py --heads $SCRATCH/mechanistic-interpretability/pi05_base

Runs on the files of action_scores.py and action_report.py (no head outputs needed). These
measures were added after seeing the results of action_report.py, so they explain its tests
rather than replace them. Read them next to the same measures on heads of pure noise.

A head can be good at everything, or better at one task or phase than at the others. Only the
second is specialization, so the "preference" columns remove what a head does in general:
a head's preference for a task is its standardized score there minus its standardized score
on the other task. If preferences are real, they repeat between two halves of the episodes.

Columns of action_specificity.csv (one row per stream, target and task; rank correlations
over the 144 heads, between two halves with no episode in common, mean over splits):

    other_task, same_task      ranking of heads in a half against the other task's ranking in
                               the other half, and against the same task's
    task_preference            how well a head's preference for this task repeats. The same
                               for both tasks, as the preference of one is minus the other's
    other_phase, same_phase    the same between phases of this task
    phase_preference           how well a head's preference for a phase repeats
    single_phase_heads         heads in the top 20 of both halves in at least 70% of splits for
                               one phase and in at most 20% for every other phase
    layer_share                share of the differences between heads explained by their layer
    same_task_within_layer     same_task after removing each layer's mean score
    beat_control               heads whose score is at least 1% below their control, of 144
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEAD = ["layer", "head"]
TOP = 20
STABLE, RARE = 0.7, 0.2  # shares of splits, for single_phase_heads


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def standardize(x):
    return (x - x.mean()) / x.std()


def halves_of(splits: pd.DataFrame) -> dict[tuple[str, int, str], pd.Series]:
    """(group, split, half) -> head scores, where group is the task or the phase."""
    return {
        key: s.set_index(HEAD)["score"].sort_index()
        for key, s in splits.groupby(["group", "split", "half"])
    }


def across_groups(splits: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per group: correlation of half a with half b of the same group, and with b of the others."""
    h = halves_of(splits)
    groups, split_ids = splits["group"].unique(), splits["split"].unique()
    same = {g: np.mean([rank_corr(h[g, s, "a"], h[g, s, "b"]) for s in split_ids]) for g in groups}
    other = {
        g: np.mean([rank_corr(h[g, s, "a"], h[o, s, "b"]) for s in split_ids for o in groups if o != g])
        for g in groups
    }
    return pd.Series(same), pd.Series(other)


def preference_repeat(splits: pd.DataFrame) -> float:
    """How well each head's preference for a group repeats between the halves, mean over groups."""
    h = halves_of(splits)
    groups, split_ids = splits["group"].unique(), splits["split"].unique()
    values = []
    for s in split_ids:
        a = pd.DataFrame({g: standardize(h[g, s, "a"]) for g in groups})
        b = pd.DataFrame({g: standardize(h[g, s, "b"]) for g in groups})
        a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
        values += [rank_corr(a[g], b[g]) for g in groups]
    return float(np.mean(values))


def single_phase_heads(splits: pd.DataFrame) -> list[tuple[str, int, int]]:
    """(phase, layer, head) of the heads repeatably in the top 20 of one phase only."""
    h = halves_of(splits)
    n_splits = splits["split"].nunique()
    share = {}
    for phase in splits["group"].unique():
        count = pd.Series(0.0, index=h[phase, 0, "a"].index)
        for s in range(n_splits):
            in_both = set(h[phase, s, "a"].nsmallest(TOP).index) & set(h[phase, s, "b"].nsmallest(TOP).index)
            count[list(in_both)] += 1 / n_splits
        share[phase] = count
    share = pd.DataFrame(share)
    found = []
    for phase in share.columns:
        others = share.drop(columns=phase).max(axis=1)
        for layer, head in share.index[(share[phase] >= STABLE) & (others <= RARE)]:
            found.append((phase, layer, head))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="directory with the action_* files")
    args = parser.parse_args()

    scores = pd.read_csv(args.heads / "action_scores.csv")
    splits = pd.read_csv(args.heads / "action_splits.csv")
    report = pd.read_csv(args.heads / "action_report.csv")

    rows, listed = [], []
    for _, r in report.iterrows():
        key = (splits["stream"] == r["stream"]) & (splits["target"] == r["target"])
        everything = splits[key & (splits["phase"] == "all")].assign(group=lambda d: d["task"])
        phases = splits[key & (splits["task"] == r["task"]) & (splits["phase"] != "all")].assign(group=lambda d: d["phase"])
        same_task, other_task = across_groups(everything)
        same_phase, other_phase = across_groups(phases)
        single = single_phase_heads(phases)
        listed += [f"  {r['stream']} {r['target']} {r['task']}: {p} L{layer}H{head}" for p, layer, head in single]

        t = scores[
            (scores["stream"] == r["stream"]) & (scores["target"] == r["target"]) & (scores["task"] == r["task"])
            & (scores["k"] == r["k"]) & (scores["phase"] == "all")
        ]
        layer_mean = t.groupby("layer")["score"].transform("mean")
        layer_share = 1 - ((t["score"] - layer_mean) ** 2).sum() / ((t["score"] - t["score"].mean()) ** 2).sum()

        mine = everything[everything["task"] == r["task"]].copy()
        mine["score"] -= mine.groupby(["split", "half", "layer"])["score"].transform("mean")
        h = halves_of(mine)
        within_layer = np.mean([rank_corr(h[r["task"], s, "a"], h[r["task"], s, "b"]) for s in mine["split"].unique()])

        rows.append(
            {
                "stream": r["stream"],
                "target": r["target"],
                "task": r["task"],
                "other_task": other_task[r["task"]],
                "same_task": same_task[r["task"]],
                "task_preference": preference_repeat(everything),
                "other_phase": other_phase.mean(),
                "same_phase": same_phase.mean(),
                "phase_preference": preference_repeat(phases),
                "single_phase_heads": len(single),
                "layer_share": layer_share,
                "same_task_within_layer": within_layer,
                "beat_control": int((t["score"] < 0.99 * t["control"]).sum()),
            }
        )

    table = pd.DataFrame(rows).round(3)
    table.to_csv(args.heads / "action_specificity.csv", index=False)
    print(table.to_string(index=False))
    print("\nheads repeatably in the top 20 of one phase only:")
    print("\n".join(listed) if listed else "  none")


if __name__ == "__main__":
    main()
