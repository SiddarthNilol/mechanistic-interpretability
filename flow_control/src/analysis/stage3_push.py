"""Stage 3 stronger push analysis: fidelity, conformity and gripper control per M. numpy + matplotlib.

    python src/analysis/stage3_push.py      # reads results/data/stage3_push.json

Rows (src/policy/push_test.py): one per (frame, M), task 60 frames from none and distractor
rollouts. Groups: early (step 10), wrong5/wrong3/wrong1 (replans before the first wrong-bowl
grasp; wrong1 is the last), right3/right1 (before the first correct grasp). Per group and M:
medians of on_axis, toward, fidelity and conformity; means of the closing fractions.

Writes results/data/stage3_push_summary.json and results/figures/stage3_push.{png,pdf}.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
GROUPS = ("early", "wrong5", "wrong3", "wrong1", "right3", "right1")
MEDIAN = ("on_axis", "toward", "fidelity", "conformity", "conformity_open", "conformity_free", "grip_response")
MEAN = ("closes_open", "closes_close", "closes_free")


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g in GROUPS:
        for m in sorted({r["m"] for r in rows}):
            rs = [r for r in rows if r["frame"] == g and r["m"] == m]
            if not rs:
                continue
            e = {"frames": len(rs)}
            e.update({k: float(np.median([r[k] for r in rs])) for k in MEDIAN})
            e.update({k: float(np.mean([r[k] for r in rs])) for k in MEAN})
            e["fidelity_iqr"] = [float(q) for q in np.percentile([r["fidelity"] for r in rs], (25, 75))]
            out[f"{g}/{m}"] = e
    return out


def plot(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    groups = [g for g in GROUPS if f"{g}/0" in summary]
    ms = sorted({int(k.split("/")[1]) for k in summary})
    colors = dict(zip(GROUPS, ("#8C8C8C", "#F2B880", "#E07B39", "#C44E52", "#8FB8DE", "#4C72B0")))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax = axes[0]
    for g in groups:
        s = [summary[f"{g}/{m}"] for m in ms]
        ax.plot([e["conformity"] for e in s], [e["fidelity"] for e in s], "-o", color=colors[g], lw=2,
                label=f"{g} (n={s[0]['frames']})")
        for m, e in zip(ms, s):
            if g == groups[0] or g == "wrong1":
                ax.annotate(f"M={m}", (e["conformity"], e["fidelity"]), textcoords="offset points", xytext=(5, -12),
                            fontsize=8, color=colors[g])
    ax.set_xlabel("conformity loss of the steered chunk (median, lower = more like the policy)")
    ax.set_ylabel("fidelity (median cosine to the command)")
    ax.set_title("translation command toward the target")
    ax.legend(fontsize=9, loc="lower right")
    ax = axes[1]
    for g in groups:
        ax.plot(ms, [summary[f"{g}/{m}"]["closes_open"] for m in ms], "-o", color=colors[g], lw=2, label=g)
        ax.axhline(summary[f"{g}/0"]["closes_free"], color=colors[g], lw=0.8, ls=":")
    ax.set_xticks(ms)
    ax.set_xlabel("M (Euler steps with the push)")
    ax.set_ylabel("fraction of chunks that close (gripper written open)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("gripper command \"open\" (dotted: unsteered)")
    w0, w6 = summary["wrong1/0"], summary["wrong1/6"]
    title = (f"Before a wrong grasp, a push for M=6 Euler steps raises fidelity from {abs(w0['fidelity']):.2f} to "
             f"{w6['fidelity']:.2f} at {w6['conformity'] / w0['conformity']:.0f}x the conformity loss, and "
             f"\"open\" with M>=4 cuts closing from {w0['closes_free']:.0%} to {summary['wrong1/4']['closes_open']:.0%} or less")
    fig.suptitle(title, fontweight="bold", fontsize=12, y=1.0)
    fig.text(0.01, -0.03, "Task 60 frames (none and distractor rollouts of the frozen pi05_libero), 4 noise seeds each. "
             "Command in the starting noise (sigma 1.5, first 5 chunk steps) plus a push of sigma/sqrt(10) after each of "
             "the first M of 10 Euler steps (paper Appendix D). M = 0 is plain Flow Control.", fontsize=9,
             color="#444444", ha="left", va="top", wrap=True)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage3_push.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / 'stage3_push'}.png and .pdf")


def main():
    rows = json.loads((DATA / "stage3_push.json").read_text())
    summary = summarize(rows)
    for k, v in summary.items():
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
    (DATA / "stage3_push_summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary)


if __name__ == "__main__":
    main()
