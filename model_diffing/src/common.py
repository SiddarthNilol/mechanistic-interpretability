"""Shared by every script: paths, the seed, bootstrap intervals, figure saving, the run log."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))  # RESULTS_DIR: write somewhere else (tests)
FIGURES = RESULTS / "figures"
DATA = RESULTS / "data"  # arrays behind the figures (not committed)
STATES_FILE = RESULTS / "states.json"
RUN_LOG = RESULTS / "run_log.md"

SEED = 0
REAL_DIMS = 7  # LIBERO actions fill the first 7 of the model's 32 padded action dims
NUM_STEPS = 10  # flow integration steps, openpi's default for inference

# Every caption carries this (CLAUDE.md, "Important caveat for all pilots").
CAVEAT = (
    "Pilot: pi05_base vs pi05_libero, which was fine-tuned on the full LIBERO data. "
    "This is a large fine-tune; the few-demo regime is left for later."
)

BASE_COLOR, FT_COLOR, GRAY = "#4C72B0", "#C44E52", "#8C8C8C"


def bootstrap_ci(statistic, n: int, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """95% interval of statistic(rows) over resamples of range(n) with replacement."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n, n)) for _ in range(n_boot)]
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def recovery(l_base: np.ndarray, l_patched: np.ndarray, rows=slice(None)) -> float:
    """Sufficiency R = (L_base - L_patched) / (L_base - L_ft) over the given states, as a ratio of
    means. L_ft is 0 because L is measured against the fine-tuned model's own chunk."""
    return float(1.0 - l_patched[rows].mean() / l_base[rows].mean())


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100,
    })
    return plt


def save_figure(fig, name: str, caption: str):
    """Adds the caption (always ending with the pilot caveat) under the figure and saves
    results/figures/<name>.png (200 dpi) and .pdf."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    text = "\n".join(textwrap.wrap(f"{caption} {CAVEAT}", width=int(fig.get_figwidth() * 13)))
    for ax in fig.axes:  # long finding-titles wrap instead of running off the figure
        if ax.get_title():
            inches = ax.get_position().width * fig.get_figwidth()
            ax.set_title("\n".join(textwrap.wrap(ax.get_title(), width=max(30, int(inches * 8.5)))))
    fig.tight_layout()
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=9, color="#444444")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGURES / name}.png and .pdf", flush=True)


def openpi_source() -> str:
    """Where the openpi that is actually imported lives, and its commit."""
    try:
        import openpi

        repo = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
        return f"`{commit}`{' (with uncommitted changes)' if dirty.strip() else ''} at `{repo}`"
    except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(name: str, started: float, **fields):
    """Appends one run to results/run_log.md: command, checkpoints, seed, number of states, wall time."""
    from pair import CHECKPOINTS  # here, so plotting-only scripts need no JAX

    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        f"\n### {name}, {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"- command: `python {shlex.join(sys.argv)}`",
        f"- checkpoints: base `{CHECKPOINTS['base']}`, fine-tuned `{CHECKPOINTS['ft']}`",
        f"- openpi: {openpi_source()}",
        f"- SLURM job: {os.environ.get('SLURM_JOB_ID', 'none')}",
        *(f"- {key.replace('_', ' ')}: {value}" for key, value in fields.items()),
        f"- wall time: {(time.time() - started) / 60:.1f} min",
    ]
    with RUN_LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")
