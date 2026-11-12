"""F6: how many backbone directions does the action expert read? One GPU, about 15 minutes.

    python src/f6_readable_spectrum.py

At backbone layer r, let d (2048 numbers) be a vector added to every prefix token's residual
after r layers, and J the Jacobian of the action expert's predicted velocity (10 actions x 7
real dims = 70 numbers) with respect to d, at d = 0. M = E[J^T J], averaged over states and flow
times, measures how strongly the actions respond to each direction of d. Its top eigenvectors
span the "action-readable subspace" of that layer.

No full Jacobians: for a standard Gaussian u (70 numbers), g = J^T u costs one vector-Jacobian
product (one backward pass), and E[g g^T] = E[J^T u u^T J] = E[J^T J] = M. So with all g as
columns of G (2048 x N), M is estimated by G G^T / N: its eigenvalues are the squared singular
values of G / N, its eigenvectors the left singular vectors of G.

    states       every other patch state of results/states.json (120), with its fixed noise
    model        the fine-tuned model (the one steered in F8); --model base for the base model
    flow times   1.0, 0.5 and 0.1 (the first, sixth and last of the 10 flow steps), at the noisy
                 actions the model's own denoising reaches there
    probes       4 Gaussian u per state and flow time -> N = 120 x 3 x 4 = 1440 columns per layer
    layers       residual after 3, 9 and 15 of the 18 layers (early, middle, late). After layer
                 18 nothing is read, so the latest useful residual is after layer 17.

k = the number of top eigenvectors holding 90% of the total (the trace of M). As a stability
check, k is also computed on each half of the states.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import DATA, REAL_DIMS, SEED, log_run, save_figure, style

LAYERS = (3, 9, 15)  # residual after this many layers
FLOW_STEPS = (0, 5, 9)  # of the 10 flow steps: flow times 1.0, 0.5, 0.1
COLORS = ("#4C72B0", "#55A868", "#C44E52")


def compute(args):
    import pair as P
    import states as S

    pair = P.Pair.load()
    states = S.load_states(patch_only=True)[:: args.every][: args.limit]
    inputs = S.Inputs(states)
    name = args.model

    def gradients(models, obs, noise, us, r, step):
        """g = J^T u for each probe: us (probes, b, horizon, 7) -> (probes, b, width)."""
        model = models[name]
        obs = P.preprocess(obs)
        tokens, mask = P.embed(model, obs)
        kv, _ = P.run_prefix(model, tokens, mask)
        _, xs, times = P.trajectory(model, obs, mask, kv, noise)
        x_t, flow_time = xs[step], times[step]

        def predicted_velocity(delta):  # delta (b, 1, width): one vector per state, added to all its tokens
            shifted, _ = P.run_prefix(model, tokens, mask, add_at=r, add_vec=delta)
            return P.velocity(model, obs, mask, shifted, x_t, flow_time)[..., :REAL_DIMS]

        _, vjp = jax.vjp(predicted_velocity, jnp.zeros((tokens.shape[0], 1, tokens.shape[-1]), tokens.dtype))
        return jax.vmap(lambda u: vjp(u)[0][:, 0])(us), flow_time

    run = pair.jit(gradients)
    key = jax.random.key(args.seed)
    columns = {r: [] for r in LAYERS}
    index = {r: [] for r in LAYERS}  # (state id, flow step) of every column
    flow_times = {}
    for i, (rows, obs, noise) in enumerate(inputs.batches(args.batch_size)):
        for r in LAYERS:
            for step in FLOW_STEPS:
                key, sub = jax.random.split(key)
                us = jax.random.normal(sub, (args.probes, *noise.shape[:2], REAL_DIMS))
                g, flow_time = run(obs, noise, us, r, step)
                g = np.asarray(g)[:, : len(rows)]  # (probes, rows, width)
                columns[r].append(g.reshape(-1, g.shape[-1]))
                index[r] += [(states[row]["id"], step) for _ in range(args.probes) for row in rows]
                flow_times[step] = float(flow_time)
        print(f"batch {i + 1}: {rows[-1] + 1}/{len(inputs)} states", flush=True)
    np.savez(
        DATA / "f6_gradients.npz", model=name, flow_steps=FLOW_STEPS,
        flow_times=[flow_times[s] for s in FLOW_STEPS],
        **{f"G_{r}": np.concatenate(columns[r]).T for r in LAYERS},  # (width, N)
        **{f"index_{r}": np.array(index[r]) for r in LAYERS},  # (N, 2): state id, flow step
    )
    return len(states)


def spectrum(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of the estimate G G^T / N of M."""
    U, s, _ = np.linalg.svd(G.astype(np.float64), full_matrices=False)
    return np.square(s) / G.shape[1], U


def k_for(eigenvalues: np.ndarray, share: float = 0.9) -> int:
    return int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), share) + 1)


def analyse(saved) -> dict:
    result = {"model": str(saved["model"]), "flow_times": saved["flow_times"].tolist(), "layers": {}}
    bases = {}
    for r in LAYERS:
        G, index = saved[f"G_{r}"], saved[f"index_{r}"]
        eigenvalues, U = spectrum(G)
        k = k_for(eigenvalues)
        ids = np.unique(index[:, 0])
        first_half = np.isin(index[:, 0], ids[::2])
        entry = {
            "width": G.shape[0], "columns": G.shape[1], "states": len(ids), "k90": k,
            "k90_half_a": k_for(spectrum(G[:, first_half])[0]), "k90_half_b": k_for(spectrum(G[:, ~first_half])[0]),
            "k50": k_for(eigenvalues, 0.5), "k99": k_for(eigenvalues, 0.99),
            "k90_by_flow_time": {
                f"{t:.1f}": k_for(spectrum(G[:, index[:, 1] == step])[0])
                for step, t in zip(saved["flow_steps"], saved["flow_times"])
            },
            "trace": float(eigenvalues.sum()),
            "eigenvalues": eigenvalues.tolist(),
        }
        result["layers"][str(r)] = entry
        bases[f"U_{r}"] = U.astype(np.float32)
        bases[f"eigenvalues_{r}"] = eigenvalues
        print(f"after {r} layers: k90 = {k} of {G.shape[0]} (halves: {entry['k90_half_a']}, {entry['k90_half_b']}; "
              f"N = {G.shape[1]} columns)", flush=True)
    np.savez(DATA / "f6_subspaces.npz", **bases)
    return result


def plot(result: dict):
    plt = style()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ks = []
    for (r, entry), color in zip(result["layers"].items(), COLORS):
        eigenvalues = np.array(entry["eigenvalues"])
        ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues / eigenvalues[0], color=color,
                label=f"after layer {r}: k = {entry['k90']} (halves of the states: {entry['k90_half_a']}, {entry['k90_half_b']})")
        ax.axvline(entry["k90"], color=color, linestyle="--", linewidth=1.2)
        ks.append(entry["k90"])
    first = next(iter(result["layers"].values()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("direction (eigenvector of M, sorted)")
    ax.set_ylabel("eigenvalue of M = E[J$^T$J],\nrelative to the largest")
    ax.legend(title="dashed: k directions hold 90% of the total", loc="lower left", fontsize=11, title_fontsize=11)
    # k is only a finding if it has stopped growing with the number of probe vectors
    halves = [max(e["k90_half_a"], e["k90_half_b"]) / e["k90"] for e in result["layers"].values()]
    if min(halves) < 0.85:
        ax.set_title(f"Not converged: k = {min(ks)} to {max(ks)} of {first['width']} directions with "
                     f"{first['columns']} probes, but k still grows with the sample (see halves)")
    else:
        ax.set_title(f"The action expert reads {min(ks)} to {max(ks)} of the backbone's {first['width']} directions "
                     "(90% of its sensitivity)")
    model = "pi05_libero" if result["model"] == "ft" else "pi05_base"
    save_figure(fig, "F6", f"Spectrum of the action expert's sensitivity to a vector added to every prefix token's "
                f"residual, {model}, {first['states']} LIBERO training states x flow times "
                f"{', '.join(f'{t:.1f}' for t in result['flow_times'])} x Gaussian probes = {first['columns']} "
                "vector-Jacobian products per layer. The estimate cannot see more directions than it has columns. "
                "Sensitivity is local (a first-order response at zero shift).")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("ft", "base"), default="ft")
    parser.add_argument("--every", type=int, default=2, help="use every n-th patch state (2 -> 120 states)")
    parser.add_argument("--probes", type=int, default=4, help="Gaussian vectors per state and flow time")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the Gaussian probes")
    parser.add_argument("--limit", type=int, help="use only the first N of the chosen states (for a quick test)")
    parser.add_argument("--plot-only", action="store_true", help="recompute spectra from results/data/f6_gradients.npz")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    n_states = None if args.plot_only else compute(args)
    result = analyse(np.load(DATA / "f6_gradients.npz"))
    (DATA / "f6.json").write_text(json.dumps(result))
    plot(result)
    if not args.plot_only:
        log_run("F6 (action-readable spectrum)", started, seed=args.seed, states=n_states, model=args.model,
                probes=args.probes, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
