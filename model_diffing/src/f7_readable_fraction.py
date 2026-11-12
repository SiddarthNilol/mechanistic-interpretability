"""F7: how much of each candidate direction lies in the action-readable subspace? No GPU.

    python src/f7_readable_fraction.py      # after f5_progress_probe.py and f6_readable_spectrum.py

For each of F6's layers, with P_k the projection on the top k eigenvectors of M (k holds 90% of
M's total, from F6):

    rho(v) = ||P_k v||^2 / ||v||^2          0 = invisible to the action expert, 1 = fully readable

for three directions v at that layer:

    progress probe      the ridge probe's weight vector from F5 (of the model F6 was run on)
    fine-tuning shift   mean over states of (fine-tuned - base) mean-pooled residual, from F5
    random              100 Gaussian directions: mean and spread. Expected value k / 2048.

Also written to results/data/f7.json, not plotted: rho of the late-minus-early mean difference
(a second way to define a progress direction), and for every direction a measure that does not
depend on k, the gain v^T M v / ||v||^2 relative to the average direction (trace(M) / 2048).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import DATA, GRAY, SEED, save_figure, style
from f6_readable_spectrum import LAYERS

N_RANDOM = 100


def analyse() -> dict:
    spectrum = json.loads((DATA / "f6.json").read_text())
    subspaces = np.load(DATA / "f6_subspaces.npz")
    directions = np.load(DATA / "f5_directions.npz")
    model = spectrum["model"]
    rng = np.random.default_rng(SEED)
    result = {"model": model, "layers": {}}
    for r in LAYERS:
        U, eigenvalues = subspaces[f"U_{r}"].astype(np.float64), subspaces[f"eigenvalues_{r}"]
        k = spectrum["layers"][str(r)]["k90"]
        width = U.shape[0]

        def rho(v, U=U, k=k):
            return float(np.square(U[:, :k].T @ v).sum() / np.square(v).sum())

        def gain(v, U=U, eigenvalues=eigenvalues, width=width):
            return float((eigenvalues * np.square(U.T @ v)).sum() / np.square(v).sum() / (eigenvalues.sum() / width))

        candidates = {
            "progress probe": directions[f"progress_{model}"][r],
            "fine-tuning shift": directions["shift"][r],
            "progress mean difference": directions[f"meandiff_{model}"][r],
        }
        random = rng.standard_normal((N_RANDOM, width))
        entry = {"k": k, "width": width, "expected_random": k / width}
        for name, v in candidates.items():
            entry[name] = {"rho": rho(v), "gain": gain(v)}
        entry["random"] = {
            "rho_mean": float(np.mean([rho(v) for v in random])), "rho_std": float(np.std([rho(v) for v in random])),
            "gain_mean": float(np.mean([gain(v) for v in random])),
        }
        result["layers"][str(r)] = entry
        print(f"after {r} layers (k = {k}): " + ", ".join(
            f"{name} rho = {entry[name]['rho']:.3f}" for name in candidates) + f", random {entry['random']['rho_mean']:.3f}")
    return result


def plot(result: dict):
    plt = style()
    fig, ax = plt.subplots(figsize=(9, 5))
    layers = list(result["layers"])
    x = np.arange(len(layers))
    bars = (("progress probe", "#DD8452", -0.27), ("fine-tuning shift", "#C44E52", 0.0), ("random", GRAY, 0.27))
    for name, color, offset in bars:
        if name == "random":
            values = [result["layers"][r]["random"]["rho_mean"] for r in layers]
            errors = [result["layers"][r]["random"]["rho_std"] for r in layers]
            ax.bar(x + offset, values, 0.26, yerr=errors, capsize=4, color=color, label=f"{N_RANDOM} random directions (mean, s.d.)")
        else:
            values = [result["layers"][r][name]["rho"] for r in layers]
            ax.bar(x + offset, values, 0.26, color=color, label=f"{name} direction")
        tops = [result["layers"][r]["random"]["rho_std"] if name == "random" else 0.0 for r in layers]
        for xi, v, top in zip(x + offset, values, tops):
            ax.text(xi, v + top + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    for xi, r in zip(x, layers):
        expected = result["layers"][r]["expected_random"]
        ax.plot([xi - 0.42, xi + 0.42], [expected] * 2, color="black", linestyle=":", linewidth=1.2,
                label="k / width (chance)" if xi == 0 else None)
    ax.set_xticks(x, [f"after layer {r}\n(k = {result['layers'][r]['k']})" for r in layers])
    ax.set_ylabel("readable fraction rho = ||P$_k$v||$^2$ / ||v||$^2$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=11)
    ratios = [result["layers"][r]["progress probe"]["rho"] / result["layers"][r]["expected_random"] for r in layers]
    if min(ratios) > 1:
        verdict = f"more readable than chance at every layer ({min(ratios):.1f}x to {max(ratios):.1f}x)"
    elif max(ratios) < 1:
        verdict = f"less readable than chance at every layer ({min(ratios):.1f}x to {max(ratios):.1f}x)"
    else:
        verdict = f"near chance ({min(ratios):.1f}x to {max(ratios):.1f}x, depending on the layer)"
    ax.set_title(f"The progress direction is {verdict}")
    model = "pi05_libero" if result["model"] == "ft" else "pi05_base"
    save_figure(fig, "F7", f"Fraction of each direction inside the action-readable subspace of {model} (top k "
                "eigenvectors of M from F6, holding 90% of its total). H3 predicts that the progress direction has "
                "low rho, which would explain why progress is decodable but does not steer. rho is about a "
                "first-order response; it is not a steering result (F8).")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    result = analyse()
    (DATA / "f7.json").write_text(json.dumps(result, indent=1))
    plot(result)


if __name__ == "__main__":
    main()
