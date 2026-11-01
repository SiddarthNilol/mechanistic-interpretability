"""Are there specialized heads? The tests on the scores from action_scores.py.

    python analysis/action_report.py --heads $SCRATCH/mechanistic-interpretability/pi05_base

Per stream, task and target, with k chosen as in the paper (the k whose 20 best heads have the
lowest mean score). Scores are errors: lower is better. "Top 20" = the 20 lowest scores.

Heads count as specialized only if all of these hold. They were fixed before seeing results:

    1. Better than the control    the top 20 clearly beat their own control
    2. Better than other heads    the top 20 clearly beat the typical head
    3. Repeatable                 the top 20 of two halves with no episode in common overlap
                                  far above chance (20 / 144 = 14%). Paper: 54-82%
    4. Holds on new episodes      heads picked on one half still beat the typical head on the
                                  other half, the only comparison free of selection bias
    5. Specific                   for tasks: top 20 shared between the tasks is lower than
                                  between two halves of one task. Same for phases

Columns of action_report.csv:

    k, cv                   chosen k; std / mean of the score over the 144 heads (paper's
                            table 4: 0.05 at the last prompt token, 0.47 at the first suffix token, pi0.5)
    best, top20, median     score of the best head, mean of the top 20, median head
    top20_vs_control        how much lower the top 20's scores are than their controls (%)
    top20_vs_median         how much lower than the median head (%), selected and scored on the
                            same frames, so biased upward
    overlap                 top-20 overlap between the two halves of a split (%), mean over splits
    rank_correlation        Spearman correlation of the two halves' scores over all 144 heads
    new_episodes_gain       heads picked on one half, scored on the other: how much lower the
                            top 20's scores are than the median head's there (%)
    new_episodes_worst      the same for the worst 20 of the first half (negative = higher error)
    other_task_overlap      top-20 overlap with the other task, half against half (%)
    phase_overlap_same      top-20 overlap of one phase between the two halves (%), over phases
    phase_overlap_other     top-20 overlap of two different phases, half against half (%)

What heads of pure noise give in these tests (test on random outputs, 10 splits), so anything
near these values is no evidence: cv 0.002-0.004, top20_vs_control under 1%, top20_vs_median
under 0.7%, overlap 8-18%, rank_correlation under 0.08, new_episodes_gain under 0.1%.

Writes to <heads>: action_report.csv, stable_heads.csv (share of splits in which a head is in the
top 20 of both halves, which have no episode in common; noise heads reach up to 0.5, because
the splits reuse the same episodes), and figures/action_<target>.png,
action_phases_<target>_<task>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_heads import STREAMS, draw_map, head_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from common import TASKS  # noqa: E402

TOP = 20
HEAD = ["layer", "head"]


def top_set(table: pd.DataFrame) -> set[tuple[int, int]]:
    """The TOP heads with the lowest score."""
    best = table.nsmallest(TOP, "score")
    return set(zip(best["layer"], best["head"]))


def overlap(a: set, b: set) -> float:
    """Share of the top heads two sets have in common, in %."""
    return 100 * len(a & b) / TOP


def choose_k(table: pd.DataFrame) -> int:
    """The paper's rule: the k whose TOP heads have the lowest mean score."""
    return int(table.groupby("k")["score"].apply(lambda s: s.nsmallest(TOP).mean()).idxmin())


def half_tests(splits: pd.DataFrame) -> dict[str, float]:
    """The repeatability and new-episodes tests, from one task's phase "all" scores in splits."""
    overlaps, correlations, gains, worst_gains = [], [], [], []
    for _, s in splits.groupby("split"):
        a = s[s["half"] == "a"].set_index(HEAD)["score"]
        b = s[s["half"] == "b"].set_index(HEAD)["score"].reindex(a.index)
        overlaps.append(overlap(set(a.nsmallest(TOP).index), set(b.nsmallest(TOP).index)))
        correlations.append(a.rank().corr(b.rank()))
        for picked_on, scored_on in ((a, b), (b, a)):
            median = scored_on.median()
            gains.append(100 * (1 - scored_on[picked_on.nsmallest(TOP).index].mean() / median))
            worst_gains.append(100 * (1 - scored_on[picked_on.nlargest(TOP).index].mean() / median))
    return {
        "overlap": np.mean(overlaps),
        "rank_correlation": np.mean(correlations),
        "new_episodes_gain": np.mean(gains),
        "new_episodes_worst": np.mean(worst_gains),
    }


def half_tops(splits: pd.DataFrame) -> dict[tuple[int, str], set]:
    """(split, half) -> top set, from scores of one phase."""
    return {key: top_set(s) for key, s in splits.groupby(["split", "half"])}


def phase_tests(splits: pd.DataFrame) -> dict[str, float]:
    """Top-20 overlap of a phase with itself and with other phases, between the two halves."""
    tops = {p: half_tops(s) for p, s in splits[splits["phase"] != "all"].groupby("phase")}
    same, other = [], []
    for p, tops_p in tops.items():
        for q, tops_q in tops.items():
            for split in {key[0] for key in tops_p}:
                value = overlap(tops_p[split, "a"], tops_q[split, "b"])
                (same if p == q else other).append(value)
    return {"phase_overlap_same": np.mean(same), "phase_overlap_other": np.mean(other)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="directory with action_scores.csv and action_splits.csv")
    args = parser.parse_args()

    scores = pd.read_csv(args.heads / "action_scores.csv")
    splits = pd.read_csv(args.heads / "action_splits.csv")
    figures = args.heads / "figures"
    figures.mkdir(exist_ok=True)
    all_scores, all_splits = scores[scores["phase"] == "all"], splits[splits["phase"] == "all"]

    rows, stable, chosen = [], [], {}
    for (stream, target), by_task in all_scores.groupby(["stream", "target"]):
        for task, table in by_task.groupby("task"):
            k = chosen[stream, target, task] = choose_k(table)
            t = table[table["k"] == k]
            top = t.nsmallest(TOP, "score")
            s = all_splits[(all_splits["stream"] == stream) & (all_splits["target"] == target)]
            mine = s[s["task"] == task]
            others = s[s["task"] != task]
            other_tops = half_tops(others)
            other_task = np.mean([overlap(top_a, other_tops[key]) for key, top_a in half_tops(mine).items()])
            detailed = splits[
                (splits["stream"] == stream) & (splits["target"] == target) & (splits["task"] == task)
            ]
            rows.append(
                {
                    "stream": stream,
                    "target": target,
                    "task": task,
                    "k": k,
                    "cv": t["score"].std() / t["score"].mean(),
                    "best": t["score"].min(),
                    "top20": top["score"].mean(),
                    "median": t["score"].median(),
                    "top20_vs_control": 100 * (1 - top["score"].mean() / top["control"].mean()),
                    "top20_vs_median": 100 * (1 - top["score"].mean() / t["score"].median()),
                    **half_tests(mine),
                    "other_task_overlap": other_task,
                    **phase_tests(detailed),
                }
            )
            halves = half_tops(mine)
            n_splits = len(halves) // 2
            both = [h for split in range(n_splits) for h in halves[split, "a"] & halves[split, "b"]]
            counts = pd.Series(both, dtype=object).value_counts() / n_splits
            stable += [
                {"stream": stream, "target": target, "task": task, "layer": h[0], "head": h[1], "share_of_splits": v}
                for h, v in counts.items()
            ]

    report = pd.DataFrame(rows).round(3)
    report.to_csv(args.heads / "action_report.csv", index=False)
    stable = pd.DataFrame(stable).round(2)
    stable.to_csv(args.heads / "stable_heads.csv", index=False)
    print(report.drop(columns=["best"]).to_string(index=False))
    print("\nmost repeatable heads (share of splits with the head in the top 20 of both halves; noise heads reach 0.5):")
    for key, s in stable.groupby(["stream", "target", "task"]):
        print(" ", *key, " ".join(f"L{r['layer']}H{r['head']}:{r['share_of_splits']:.2f}" for _, r in s.head(6).iterrows()))

    # Maps: how much lower each head's score is than its control, in %.
    for target, by_target in all_scores.groupby("target"):
        task_names = sorted(by_target["task"].unique())
        fig, axes = plt.subplots(len(task_names), 2, figsize=(7, 4.2 * len(task_names)), squeeze=False, layout="constrained")
        gains = 100 * (1 - by_target["score"] / by_target["control"])
        vmax = max(float(gains.max()), 1.0)
        for row, task in enumerate(task_names):
            for col, stream in enumerate(STREAMS):
                t = by_target[(by_target["task"] == task) & (by_target["stream"] == stream)]
                t = t[t["k"] == chosen[stream, target, task]]
                t = t.assign(gain=100 * (1 - t["score"] / t["control"]))
                best = t.loc[t["score"].idxmin()]
                image = draw_map(axes[row, col], head_map(t, "gain"), 0, vmax, top_set(t),
                                 f"{task} · {stream}\nbest L{best['layer']}H{best['head']}: {best['score']:.2f} (control {best['control']:.2f})")
        fig.colorbar(image, ax=axes, shrink=0.3, label="error below control (%)")
        fig.suptitle(f"Action score, target: {target} (outlined: top {TOP})", fontsize=10)
        fig.savefig(figures / f"action_{target}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for task in task_names:
            detail = scores[(scores["target"] == target) & (scores["task"] == task) & (scores["phase"] != "all")]
            phases = TASKS[task]["phases"]
            fig, axes = plt.subplots(2, len(phases), figsize=(2.2 * len(phases), 8), squeeze=False, layout="constrained")
            gains = 100 * (1 - detail["score"] / detail["control"])
            vmax = max(float(gains.max()), 1.0)
            for row, stream in enumerate(STREAMS):
                for col, phase in enumerate(phases):
                    t = detail[(detail["stream"] == stream) & (detail["phase"] == phase)]
                    t = t[t["k"] == chosen[stream, target, task]]
                    image = draw_map(axes[row, col], head_map(t.assign(gain=100 * (1 - t["score"] / t["control"])), "gain"),
                                     0, vmax, top_set(t), f"{phase} · {stream}", axis_labels=col == 0)
            fig.colorbar(image, ax=axes, shrink=0.3, label="error below control (%)")
            fig.suptitle(f"{task}, target {target}: action score per phase (outlined: top {TOP})", fontsize=10)
            fig.savefig(figures / f"action_phases_{target}_{task.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"\nwrote {args.heads / 'action_report.csv'} and {figures}")


if __name__ == "__main__":
    main()
