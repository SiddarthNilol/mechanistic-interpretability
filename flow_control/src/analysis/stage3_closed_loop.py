"""Stage 3 closed loop analysis: success, wrong-grasp rate and steps per condition. numpy + matplotlib.

    python3 src/analysis/stage3_closed_loop.py      # reads results/data/stage3_cl_*.jsonl

Conditions (src/sim/rollout.py): unsteered, steered (plain Flow Control, oracle), veto, adaptive,
both. Settings: task 60 (distractor, none), task 12 (none, shift). Per setting and condition:
successes out of 50 with Wilson 95% intervals; exact McNemar test against steered, paired by
initial state; wrong-grasp rate (another bowl held at any replan); median steps of successes;
for the extensions, the share of replans with the gate active and the mean translation M used.

Writes results/data/stage3_closed_loop_summary.json and results/figures/stage3_closed_loop.{png,pdf}.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
CONDITIONS = ("unsteered", "steered", "veto", "adaptive", "both")
SETTINGS = ((60, "distractor"), (60, "none"), (12, "none"), (12, "shift"))
COLORS = dict(zip(CONDITIONS, ("#8C8C8C", "#4C72B0", "#55A868", "#DD8452", "#C44E52")))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load() -> list[dict]:
    rows = []
    for path in sorted(DATA.glob("stage3_cl_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for task, kind in SETTINGS:
        by = {c: {r["state"]: r for r in rows if r["task"] == task and r["disturbance"]["kind"] == kind
                  and r["condition"] == c} for c in CONDITIONS}
        if not by["steered"]:
            continue
        for c in CONDITIONS:
            rs = by[c]
            if not rs:
                continue
            k, n = sum(r["success"] for r in rs.values()), len(rs)
            wrong = [any(t["other_held"] for t in r["trace"]) for r in rs.values()]
            e = {"success": k, "n": n, "wilson": wilson(k, n), "wrong_grasp": int(sum(wrong)),
                 "median_steps_success": float(np.median([r["steps"] for r in rs.values() if r["success"]])) if k else None}
            if c != "steered":
                paired = [s for s in rs if s in by["steered"]]
                b = sum(rs[s]["success"] and not by["steered"][s]["success"] for s in paired)
                cc = sum(by["steered"][s]["success"] and not rs[s]["success"] for s in paired)
                e.update(vs_steered={"only_this": b, "only_steered": cc, "p": mcnemar(b, cc)})
            ext = [x for r in rs.values() for x in r.get("extensions", [])]
            if ext:
                a = np.array(ext)
                e["gate_share"] = float(a[:, 0].mean())
                e["mean_m_when_gated"] = float(a[a[:, 0] == 1, 1].mean()) if a[:, 0].any() else 0.0
            out[f"{task}/{kind}/{c}"] = e
    return out


def plot(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    settings = [s for s in SETTINGS if f"{s[0]}/{s[1]}/steered" in summary]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(settings))
    width = 0.8 / len(CONDITIONS)
    for i, c in enumerate(CONDITIONS):
        es = [summary.get(f"{t}/{k}/{c}") for t, k in settings]
        xs = x + (i - (len(CONDITIONS) - 1) / 2) * width
        for ax, key in ((axes[0], "success"), (axes[1], "wrong_grasp")):
            vals = [e[key] if e else np.nan for e in es]
            ax.bar(xs, vals, width, color=COLORS[c], label=c)
        lo = [e["success"] - e["n"] * e["wilson"][0] if e else 0 for e in es]
        hi = [e["n"] * e["wilson"][1] - e["success"] if e else 0 for e in es]
        axes[0].errorbar(xs, [e["success"] if e else np.nan for e in es], yerr=[lo, hi], fmt="none",
                         ecolor="#333333", lw=0.8, capsize=2)
    for ax, label in ((axes[0], "successes out of 50 (Wilson 95%)"), (axes[1], "episodes with a wrong-bowl grasp")):
        ax.set_xticks(x)
        ax.set_xticklabels([f"task {t}\n{k}" for t, k in settings])
        ax.set_ylabel(label)
        ax.set_ylim(0, 52)
    axes[0].set_title("success")
    axes[1].set_title("wrong-bowl grasps")
    handles, labels = axes[0].get_legend_handles_labels()
    d = {c: summary.get(f"60/distractor/{c}") for c in CONDITIONS}
    best = max((c for c in ("veto", "adaptive", "both") if d[c]), key=lambda c: d[c]["success"])
    name = {"both": "veto + adaptive"}.get(best, best)
    title = (f"Task 60 with a distractor: {name} reaches {d[best]['success']}/50 against {d['steered']['success']}/50 "
             f"for plain Flow Control (exact McNemar p = {d[best]['vs_steered']['p']:.2g})")
    fig.suptitle(title, fontweight="bold", fontsize=12, y=1.0)
    fig.legend(handles, labels, fontsize=10, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.07, "Frozen pi05_libero, 50 initial states per setting, paired noise (stream 0) and disturbance seeds. "
             "Oracle state. Extensions act only while the target is not held and the gripper is more than 8 cm from it: "
             "veto = gripper \"open\" pushed for 6 Euler steps; adaptive = smallest M in {0,2,4,6,8} with fidelity >= 0.8.",
             fontsize=9, color="#444444", ha="left", va="top", wrap=True)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage3_closed_loop.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / 'stage3_closed_loop'}.png and .pdf")


def main():
    summary = summarize(load())
    for k, v in summary.items():
        print(k, json.dumps(v))
    (DATA / "stage3_closed_loop_summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary)


if __name__ == "__main__":
    main()
