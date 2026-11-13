"""Stage 1 analysis: our Flow Control reproduction next to the paper's numbers. numpy + matplotlib.

    python src/analysis/stage1.py        # reads results/data/stage1.jsonl

For each task: success counts of unsteered and steered episodes over the same initial states,
Wilson 95% intervals, and the exact McNemar test on the paired episodes (as in the paper: only
the discordant pairs count, two-sided binomial test with p = 1/2). Writes
results/data/stage1_summary.json and results/figures/stage1.png and .pdf.

Gate (CLAUDE.md): steered clearly above unsteered on both tasks, in the paper's direction.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
PAPER = {60: {"unsteered": 27, "steered": 47}, 12: {"unsteered": 23, "steered": 33}}  # of 50, Appendix G.1
TASK_NAMES = {12: "task 12: bowl at the back onto the plate", 60: "task 60: left bowl into the tray"}
COLORS = {"unsteered": "#8C8C8C", "steered": "#C44E52"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only steered succeeded, c = only unsteered succeeded."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for task in sorted({r["task"] for r in rows}):
        by = {(r["state"], r["condition"]): r["success"] for r in rows if r["task"] == task}
        states = sorted({s for s, c in by if (s, "unsteered") in by and (s, "steered") in by})
        u = np.array([by[s, "unsteered"] for s in states])
        st = np.array([by[s, "steered"] for s in states])
        b, c = int((st & ~u).sum()), int((u & ~st).sum())
        out[task] = {
            "pairs": len(states),
            "unsteered": int(u.sum()), "steered": int(st.sum()),
            "unsteered_ci": wilson(int(u.sum()), len(states)), "steered_ci": wilson(int(st.sum()), len(states)),
            "only_steered_succeeded": b, "only_unsteered_succeeded": c, "mcnemar_p": mcnemar_exact(b, c),
            "paper": PAPER.get(task),
        }
    return out


def plot(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    tasks = sorted(summary)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.2
    for i, task in enumerate(tasks):
        s = summary[task]
        n = s["pairs"]
        for j, cond in enumerate(("unsteered", "steered")):
            ours = s[cond] / n
            lo, hi = s[f"{cond}_ci"]
            x = i + (j - 0.5) * 2 * width
            ax.bar(x - width / 2, ours, width, color=COLORS[cond], label=f"{cond}, ours" if i == 0 else None)
            ax.errorbar(x - width / 2, ours, yerr=[[max(0.0, ours - lo)], [max(0.0, hi - ours)]], color="black", capsize=4)
            ax.text(x - width / 2, hi + 0.02, f"{s[cond]}/{n}", ha="center", fontsize=11)
            if s["paper"]:
                paper = s["paper"][cond] / 50
                ax.bar(x + width / 2, paper, width, color=COLORS[cond], alpha=0.35, hatch="//", edgecolor=COLORS[cond],
                       label=f"{cond}, paper" if i == 0 else None)
                ax.text(x + width / 2, paper + 0.02, f"{s['paper'][cond]}/50", ha="center", fontsize=11, color="#555555")
        ax.text(i, -0.17, f"McNemar p = {s['mcnemar_p']:.2g}", ha="center", fontsize=11, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_NAMES.get(t, f"task {t}") for t in tasks])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("success rate (episodes)")
    ax.legend(loc="upper left", ncol=2, fontsize=11)
    gains = {t: summary[t]["steered"] - summary[t]["unsteered"] for t in tasks}
    clear = [t for t in tasks if gains[t] > 0 and summary[t]["mcnemar_p"] < 0.05]
    parts = ", ".join(f"task {t} {gains[t]:+d} (p = {summary[t]['mcnemar_p']:.2g})" for t in tasks)
    if len(clear) == len(tasks):
        title = f"Scripted Flow Control raises success on both tasks: {parts}"
    elif clear:
        title = f"Scripted Flow Control clearly helps only on task {', '.join(map(str, clear))}: {parts}"
    else:
        title = f"Scripted Flow Control gives no clear gain: {parts}"
    ax.set_title(title)
    fig.text(0.01, -0.06, "pi05_libero on LIBERO-90 tasks held out from its fine-tuning. Paired episodes: each task's fixed "
             "initial states, the same noise seed at each replan in both conditions. Bars: ours (solid, Wilson 95% "
             "intervals) and the paper's Appendix G.1 counts (hatched). Exact McNemar test on the paired episodes.",
             wrap=True, fontsize=9, color="#444444", ha="left", va="top")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"stage1.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / 'stage1'}.png and .pdf")


def main():
    rows = [json.loads(line) for line in (DATA / "stage1.jsonl").read_text().splitlines() if line.strip()]
    summary = summarize(rows)
    for task, s in summary.items():
        paper = s["paper"]
        print(f"task {task}: unsteered {s['unsteered']}/{s['pairs']}, steered {s['steered']}/{s['pairs']} "
              f"(only steered {s['only_steered_succeeded']}, only unsteered {s['only_unsteered_succeeded']}, "
              f"McNemar p = {s['mcnemar_p']:.3g}); paper {paper['unsteered']}/50 -> {paper['steered']}/50")
    (DATA / "stage1_summary.json").write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=1))
    plot(summary)


if __name__ == "__main__":
    main()
