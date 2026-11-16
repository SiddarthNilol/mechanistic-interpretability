"""Stage 3 diagnostic analysis: commitment and gripper control at wrong grasps. numpy + matplotlib.

    python src/analysis/stage3_diagnostic.py      # reads results/data/stage3_diagnostic.json

Frame groups (src/policy/grasp_diagnostic.py): early (step 10), right2/right1 (the 2 replans
before the first correct grasp), wrong2/wrong1 (before the first wrong-bowl grasp). Per task and
group: medians of the on-axis response toward the target and of the unsteered motion toward it,
and the fraction of (frame, seed) chunks that close the gripper unsteered, with the gripper
written open, and with the planned veto (toward-target command plus open).

Writes results/data/stage3_diagnostic_summary.json and results/figures/stage3_diagnostic.{png,pdf}.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
GROUPS = ("early", "right2", "right1", "wrong2", "wrong1")


def summarize(rows: list[dict]) -> dict:
    out = {}
    for task in sorted({r["task"] for r in rows}):
        for g in GROUPS:
            rs = [r for r in rows if r["task"] == task and r["frame"] == g]
            if not rs:
                continue
            m = lambda k: float(np.median([r[k] for r in rs]))  # noqa: E731
            mean = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            out[f"{task}/{g}"] = {
                "frames": len(rs), "on_axis": m("on_axis"), "unsteered_toward_target": m("unsteered"),
                "toward": m("toward"), "grip_response": m("grip_response"),
                "closes_free": mean("closes_free"), "closes_open": mean("closes_open"),
                "closes_close": mean("closes_close"), "closes_veto": mean("closes_veto"),
                "veto_toward": m("veto_toward"),
                "eef_to_target_cm": m("eef_to_target_cm"), "eef_to_nearest_other_cm": m("eef_to_nearest_other_cm"),
                "eef_height_cm": m("eef_height_cm"),
            }
    return out


def plot(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    tasks = sorted({k.split("/")[0] for k in summary}, key=int)
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.5 * len(tasks), 8.5), squeeze=False)
    for col, task in enumerate(tasks):
        groups = [g for g in GROUPS if f"{task}/{g}" in summary]
        x = np.arange(len(groups))
        s = [summary[f"{task}/{g}"] for g in groups]
        ax = axes[0, col]
        ax.bar(x - 0.2, [e["on_axis"] for e in s], 0.4, color="#4C72B0", label="on-axis response (+cmd - -cmd)")
        ax.bar(x + 0.2, [e["unsteered_toward_target"] for e in s], 0.4, color="#8C8C8C", label="unsteered motion toward target")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\nn={e['frames']}" for g, e in zip(groups, s)], fontsize=10)
        ax.set_title(f"task {task}: translation toward the target")
        ax.set_ylabel("median, LIBERO units")
        ax = axes[1, col]
        for i, (key, color, label) in enumerate((("closes_free", "#8C8C8C", "unsteered"),
                                                ("closes_open", "#55A868", "gripper written open"),
                                                ("closes_veto", "#C44E52", "toward target + open"))):
            ax.bar(x + (i - 1) * 0.27, [e[key] for e in s], 0.27, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of chunks that close")
        ax.set_title(f"task {task}: does the gripper close in the next 5 steps")
    axes[0, 0].legend(fontsize=9, loc="upper left")
    axes[1, 0].legend(fontsize=9, loc="upper left")
    wrong = [summary[f"{t}/wrong1"] for t in tasks if f"{t}/wrong1" in summary]
    gap = max(abs(e["closes_free"] - e["closes_open"]) for e in wrong)
    ratio = max(e["on_axis"] / summary[f"{t}/early"]["on_axis"] for t, e in zip(tasks, wrong))
    if gap < 0.1:
        title = (f"Writing \"open\" into the noise does not stop a wrong grasp (closing changes by at most {gap:.0%}); "
                 f"translation commands keep at most {ratio:.0%} of their early effect")
    else:
        title = f"Writing \"open\" into the noise changes closing by up to {gap:.0%} before a wrong grasp"
    fig.suptitle(title, fontweight="bold", fontsize=12, y=1.0)
    fig.text(0.01, -0.03, "Frames from rollouts of the frozen pi05_libero (task 60: none and distractor; task 12: none), "
             "8 noise seeds each. wrong1/right1: the replan just before the first wrong or correct grasp. Commands written "
             "into the starting noise for the first 5 chunk steps (sigma 1.5).", fontsize=9, color="#444444", ha="left",
             va="top", wrap=True)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage3_diagnostic.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / 'stage3_diagnostic'}.png and .pdf")


def main():
    rows = json.loads((DATA / "stage3_diagnostic.json").read_text())
    summary = summarize(rows)
    for k, v in summary.items():
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
    (DATA / "stage3_diagnostic_summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary)


if __name__ == "__main__":
    main()
