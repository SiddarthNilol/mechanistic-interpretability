"""Stage 2 analysis: forced-failure benchmark. numpy + matplotlib.

    python src/analysis/stage2.py      # reads results/data/stage2_<disturbance>.jsonl (+ stage2_commitment.json)

Per task, disturbance and condition: success over the 50 paired initial states (Wilson 95%),
steered vs unsteered (exact McNemar, paired by initial state), and the disturbance's cost against
the no-disturbance reference of the same condition (also paired).

Behavior after the disturbance, one label per episode, from the per-replan trace (simulator
state, used here only for scoring). Checked in this order, the first that applies wins:

    not disturbed       the trigger never happened (for example the bowl was never carried)
    reached moved goal  moved_goal: success (the bowl went to the goal's new position)
    landed on goal      drop: success within LANDED steps of the forced opening (fell onto it)
    grasped wrong       another bowl was held after the disturbance (any time, for distractor)
    grasped target      the target was held again after the disturbance (drop, shift); for
                        distractor and none: the target was held at some point
    placed at old goal  moved_goal: the target ended within OLD_GOAL of where the goal was
    still holding       moved_goal: the target was still held at the end
    empty carry         drop: never re-grasped, and the gripper went to within EMPTY_CARRY of
                        the goal (and closer than at the drop)
    went to old spot    shift: before any grasp, the gripper went to within OLD_SPOT of where
                        the target was before the shift
    stalled             not a success, and the gripper moved less than STALL over the last
                        STALL_REPLANS replans
    other               none of these

Writes results/data/stage2_summary.json and results/figures/stage2_success.{png,pdf},
stage2_behavior.{png,pdf}, stage2_commitment.{png,pdf}.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage1 import mcnemar_exact, wilson  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
KINDS = ("none", "drop", "shift", "distractor", "moved_goal")
SHORT = {"none": "none", "drop": "drop", "shift": "shift", "distractor": "distr.", "moved_goal": "goal\nmoved"}
CONDITIONS = ("unsteered", "steered")
LABELS = ("reached moved goal", "landed on goal", "grasped target", "grasped wrong", "empty carry", "went to old spot", "placed at old goal", "still holding",
          "stalled", "other", "not disturbed")
LABEL_COLORS = dict(zip(LABELS, ("#2CA02C", "#55A868", "#4C72B0", "#C44E52", "#DD8452", "#937860", "#8172B3", "#64B5CD", "#8C8C8C",
                                 "#CCB974", "#E5E5E5")))
LANDED = 30  # steps: a success this soon after a drop means the bowl fell onto the goal
EMPTY_CARRY = 0.10  # m
OLD_SPOT = 0.03  # m
OLD_GOAL = 0.06  # m
STALL, STALL_REPLANS = 0.02, 10  # m, replans


def _h(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b))[:2]))


def label(r: dict) -> str:
    d, trace = r["disturbance"], r["trace"]
    kind = d["kind"]
    if kind != "none" and (not d.get("fired") or d.get("skipped")):
        return "not disturbed"
    t0 = d.get("open_until", d.get("step", 0)) if kind != "none" else 0
    post = [e for e in trace if e["step"] > t0] if kind not in ("none", "distractor") else trace
    if kind == "moved_goal" and r["success"]:
        return "reached moved goal"
    if kind == "drop" and r["success"] and r["steps"] <= d["open_until"] + LANDED:
        return "landed on goal"
    if any(e["other_held"] for e in post):
        return "grasped wrong"
    if kind in ("drop", "shift") and any(e["held"] for e in post):
        return "grasped target"
    if kind in ("none", "distractor") and any(e["held"] for e in trace):
        return "grasped target"
    if kind == "moved_goal" and not r["success"]:
        if _h(trace[-1]["target"], d["goal_before"]) < OLD_GOAL:
            return "placed at old goal"
        if trace[-1]["held"]:
            return "still holding"
    if kind == "drop" and post:
        at_drop = _h(d["eef"], d["goal_before"])
        nearest = min(_h(e["eef"], e["goal"]) for e in post)
        if nearest < EMPTY_CARRY and nearest < at_drop:
            return "empty carry"
    if kind == "shift" and post:
        if any(_h(e["eef"], d["target_before"]) < OLD_SPOT for e in post):
            return "went to old spot"
    if not r["success"]:
        tail = np.array([e["eef"] for e in trace[-STALL_REPLANS:]])
        if len(tail) and np.linalg.norm(tail.max(0) - tail.min(0)) < STALL:
            return "stalled"
    return "other"


def summarize(rows: list[dict]) -> dict:
    by = {(r["task"], r["disturbance"]["kind"], r["state"], r["condition"]): r for r in rows}
    out = {}
    for task in sorted({r["task"] for r in rows}):
        for kind in KINDS:
            states = sorted({s for (t, k, s, c) in by if t == task and k == kind and all((t, k, s, cc) in by for cc in CONDITIONS)})
            if not states:
                continue
            entry = {"pairs": len(states)}
            succ = {c: np.array([by[task, kind, s, c]["success"] for s in states]) for c in CONDITIONS}
            for c in CONDITIONS:
                rs = [by[task, kind, s, c] for s in states]
                k = int(succ[c].sum())
                entry[c] = {
                    "success": k, "ci": wilson(k, len(states)),
                    "fired": sum(1 for r in rs if kind == "none" or (r["disturbance"].get("fired") and not r["disturbance"].get("skipped"))),
                    "labels": dict(collections.Counter(label(r) for r in rs)),
                    "success_by_label": dict(collections.Counter(label(r) for r in rs if r["success"])),
                }
                ref = [s for s in states if (task, "none", s, c) in by]
                if kind != "none" and ref:
                    a = np.array([by[task, "none", s, c]["success"] for s in ref])
                    b = np.array([by[task, kind, s, c]["success"] for s in ref])
                    entry[c]["cost_vs_none"] = {"none": int(a.sum()), "disturbed": int(b.sum()), "pairs": len(ref),
                                                "mcnemar_p": mcnemar_exact(int((a & ~b).sum()), int((b & ~a).sum()))}
            b = int((succ["steered"] & ~succ["unsteered"]).sum())
            c_ = int((succ["unsteered"] & ~succ["steered"]).sum())
            entry["only_steered"], entry["only_unsteered"], entry["mcnemar_p"] = b, c_, mcnemar_exact(b, c_)
            both = [i for i, s in enumerate(states) if kind == "none" or all(
                by[task, kind, s, cc]["disturbance"].get("fired") and not by[task, kind, s, cc]["disturbance"].get("skipped")
                for cc in CONDITIONS)]
            entry["both_disturbed"] = {"pairs": len(both), **{c: int(succ[c][both].sum()) for c in CONDITIONS}}
            out[f"{task}/{kind}"] = entry
    return out


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    return plt


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / name}.png and .pdf")


def plot_success(summary: dict):
    plt = _style()
    tasks = sorted({k.split("/")[0] for k in summary}, key=int)
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"unsteered": "#8C8C8C", "steered": "#C44E52"}
    worst = []
    for ax, task in zip(axes, tasks):
        kinds = [k for k in KINDS if f"{task}/{k}" in summary]
        for i, kind in enumerate(kinds):
            e = summary[f"{task}/{kind}"]
            for j, c in enumerate(CONDITIONS):
                n, k = e["pairs"], e[c]["success"]
                lo, hi = e[c]["ci"]
                x = i + (j - 0.5) * 0.38
                ax.bar(x, k / n, 0.36, color=colors[c], label=c if i == 0 else None)
                ax.errorbar(x, k / n, yerr=[[max(0, k / n - lo)], [max(0, hi - k / n)]], color="black", capsize=3)
                ax.text(x, hi + 0.02, f"{k}", ha="center", fontsize=10)
            ax.text(i, -0.2, f"p = {e['mcnemar_p']:.2g}", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
            if kind != "none":
                worst.append((e["steered"]["success"] / n, task, kind))
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([k.replace("_", " ") for k in kinds])
        ax.set_title(f"task {task}")
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("success rate (50 paired episodes)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, fontsize=10)
    rate, t, k = min(worst)
    fig.suptitle(f"Oracle Flow Control is weakest under {k.replace('_', ' ')} (task {t}: {rate:.0%} steered success)",
                 fontweight="bold", y=1.02)
    fig.text(0.01, -0.08, "Bars: success with Wilson 95% intervals; numbers above bars are successes out of 50. p: exact "
             "McNemar, steered vs unsteered, paired by initial state. Steered uses the privileged scripted controller.",
             fontsize=9, color="#444444", ha="left", va="top", wrap=True)
    fig.tight_layout()
    _save(fig, "stage2_success")


def plot_behavior(summary: dict):
    plt = _style()
    keys = [(k.split("/")[0], k.split("/")[1], c) for k in summary for c in CONDITIONS if not k.endswith("/none")]
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(keys)), 5.5))
    for i, (task, kind, c) in enumerate(keys):
        labels = summary[f"{task}/{kind}"][c]["labels"]
        bottom = 0
        for name in LABELS:
            v = labels.get(name, 0)
            if v:
                ax.bar(i, v, 0.8, bottom=bottom, color=LABEL_COLORS[name], label=name if name not in ax.get_legend_handles_labels()[1] else None)
                bottom += v
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{t}\n{SHORT[k]}\n{c[0].upper()}" for t, k, c in keys], fontsize=9)
    ax.set_ylabel("episodes (of 50)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    drop = [summary[f"{t}/drop"]["unsteered"]["labels"] for t in sorted({k[0] for k in keys}) if f"{t}/drop" in summary]
    regrasp = sum(d.get("grasped target", 0) for d in drop)
    fired = sum(sum(v for n, v in d.items() if n != "not disturbed") for d in drop)
    ax.set_title(f"After a drop, the unsteered policy grasps the bowl again in {regrasp} of {fired} dropped episodes")
    fig.text(0.01, -0.1, "Labels from simulator state after the disturbance (see src/analysis/stage2.py). Columns: task, "
             "disturbance, U = unsteered, S = steered (privileged scripted controller).", fontsize=9, color="#444444",
             ha="left", va="top", wrap=True)
    fig.tight_layout()
    _save(fig, "stage2_behavior")


def plot_commitment(rows: list[dict]) -> dict:
    plt = _style()
    stats = {}
    tasks = sorted({r["task"] for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, task in zip(axes, tasks):
        groups = {"early": [r for r in rows if r["task"] == task and r["frame"] == "early"],
                  "after drop": [r for r in rows if r["task"] == task and r["frame"].startswith("post")]}
        for i, (name, g) in enumerate(groups.items()):
            v = np.array([r["on_axis"] for r in g])
            if not len(v):
                continue
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=12, alpha=0.5, color="#4C72B0")
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="black", lw=2)
            stats[f"{task}/{name}"] = {"n": len(v), "median": float(np.median(v)), "mean": float(v.mean()),
                                       "toward_median": float(np.median([r["toward"] for r in g])),
                                       "unsteered_toward_median": float(np.median([r["unsteered"] for r in g]))}
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(groups))
        ax.set_title(f"task {task}")
        ax.axhline(0, color="#999999", lw=0.8)
    axes[0].set_ylabel("on-axis response: (+cmd) - (-cmd)\nmean of first 5 actions, LIBERO units")
    ratios = [stats[f"{t}/after drop"]["median"] / stats[f"{t}/early"]["median"] for t in tasks
              if f"{t}/after drop" in stats and f"{t}/early" in stats and stats[f"{t}/early"]["median"] != 0]
    if ratios and max(ratios) < 0.8:
        title = f"Commands move the policy less after a drop ({min(ratios):.0%} to {max(ratios):.0%} of the early response)"
    elif ratios and min(ratios) > 1.2:
        title = "Commands move the policy more after a drop than early in the episode"
    else:
        title = "Commands move the policy about as much after a drop as early in the episode"
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.text(0.01, -0.06, "Each dot: one frame, response averaged over 8 noise seeds. Early: step 10. After drop: the "
             "first two replans after the forced opening. Axis: dominant axis from the gripper to the object.",
             fontsize=9, color="#444444", ha="left", va="top", wrap=True)
    fig.tight_layout()
    _save(fig, "stage2_commitment")
    return stats


def main():
    rows = [json.loads(line) for path in sorted(DATA.glob("stage2_*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]
    summary = summarize(rows)
    for key, e in summary.items():
        print(f"{key}: unsteered {e['unsteered']['success']}/{e['pairs']} (fired {e['unsteered']['fired']}), "
              f"steered {e['steered']['success']}/{e['pairs']} (fired {e['steered']['fired']}), McNemar p = {e['mcnemar_p']:.2g}")
        for c in CONDITIONS:
            print(f"    {c}: {e[c]['labels']}  cost vs none: {e[c].get('cost_vs_none')}")
    commitment = DATA / "stage2_commitment.json"
    if commitment.exists():
        summary["commitment"] = plot_commitment(json.loads(commitment.read_text()))
        print("commitment:", json.dumps(summary["commitment"], indent=1))
    (DATA / "stage2_summary.json").write_text(json.dumps(summary, indent=1))
    plot_success({k: v for k, v in summary.items() if k != "commitment"})
    plot_behavior({k: v for k, v in summary.items() if k != "commitment"})


if __name__ == "__main__":
    main()
