"""F8 (optional), part 3 of 3: plot the steering rollouts. No GPU.

    python src/f8_plot.py        # reads results/data/f8_rollouts.jsonl

Success rate and mean end-effector speed against the steering strength alpha, one line per
direction. Intervals: Wilson 95% for the success rate, 95% bootstrap over episodes for speed.
"""

from __future__ import annotations

import json

import numpy as np

from model_diffing.archive.src.common import DATA, GRAY, bootstrap_ci, save_figure, style

COLORS = {"progress": "#DD8452", "progress_readable": "#C44E52", "random": GRAY}
LABELS = {"progress": "progress direction", "progress_readable": "its readable part (P$_k$v, same norm)",
          "random": "random direction"}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    records = [json.loads(line) for line in (DATA / "f8_rollouts.jsonl").read_text().splitlines()]
    baseline = [r for r in records if r["direction"] == "none"]
    plt = style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    for direction in COLORS:
        mine = [r for r in records if r["direction"] == direction]
        alphas = sorted({r["alpha"] for r in mine})
        groups = [baseline] + [[r for r in mine if r["alpha"] == a] for a in alphas]
        x = [0.0, *alphas]
        success = np.array([np.mean([r["success"] for r in g]) for g in groups])
        bounds = np.array([wilson(sum(r["success"] for r in g), len(g)) for g in groups])
        left.errorbar(x, success, yerr=np.abs(bounds.T - success), marker="o", capsize=4, color=COLORS[direction],
                      label=LABELS[direction])
        speeds = [np.array([r["mean_speed_m_per_s"] for r in g]) for g in groups]
        mean = np.array([s.mean() for s in speeds])
        ci = np.array([bootstrap_ci(lambda rows, s=s: float(s[rows].mean()), len(s)) for s in speeds])
        right.errorbar(x, mean, yerr=np.abs(ci.T - mean), marker="o", capsize=4, color=COLORS[direction])
    left.set_ylim(-0.05, 1.05)
    left.set_xlabel("steering strength alpha (token residual norms)")
    left.set_ylabel("success rate")
    left.legend(fontsize=11)
    right.set_xlabel("steering strength alpha (token residual norms)")
    right.set_ylabel("mean end-effector speed (m/s)")
    n = len(baseline)
    fig.suptitle(f"10-episode pilot: steering pi05_libero along the progress direction and its readable part "
                 f"(baseline success {np.mean([r['success'] for r in baseline]):.0%})", fontweight="bold")
    save_figure(fig, "F8", f"Closed-loop steering in LIBERO ({records[0]['suite']}, task: {records[0]['task']}): "
                f"alpha x v is added to every prefix token's residual at one backbone layer during rollouts. "
                f"A {n}-episode pilot per condition: intervals are wide (Wilson 95% for success, bootstrap for "
                "speed) and one task says nothing about others.")
    plt.close(fig)


if __name__ == "__main__":
    main()
