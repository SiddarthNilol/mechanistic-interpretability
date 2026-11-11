"""F5: is task progress linearly decodable from each backbone layer? One GPU, a few minutes.

    python src/f5_progress_probe.py

For all 960 states of results/states.json (80 episodes, 12 frames each) and both models: the
residual after r layers, r = 0 (token embeddings) to 18, averaged over the valid prefix tokens
(camera and prompt tokens), a 2048-dim feature. Then, per layer and model, ridge regression
from the standardized feature to task progress = frame index / episode length.

Held-out R^2: 5 folds split by episode, so no episode has frames on both sides. The ridge
strength is chosen inside each training fold, again split by episode. Bands: 95% bootstrap over
episodes. Control: the same probe trained on progress labels shuffled across the training frames.

Saved for F7 and F8 (results/data/f5_directions.npz), per layer, in residual coordinates:

    progress_<model>    the probe's weight vector, fit on all states: the "progress direction".
                        Adding v to every prefix token shifts the pooled feature by v, so the
                        direction lives in the same space as F6's readable subspace.
    meandiff_<model>    mean feature of late frames (progress > 2/3) minus early (< 1/3)
    shift               mean over states of (fine-tuned feature - base feature)
    token_norm_<model>  mean norm of one token's residual, the scale for steering in F8
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import BASE_COLOR, DATA, FT_COLOR, GRAY, SEED, log_run, save_figure, style

ALPHAS = np.logspace(0, 6, 7)  # ridge strengths tried (features are standardized)
FOLDS = 5


def extract(args):
    import jax.numpy as jnp

    import pair as P
    import states as S

    pair = P.Pair.load()
    states = S.load_states()[: args.limit]
    inputs = S.Inputs(states)

    def features(models, obs):
        obs = P.preprocess(obs)
        out = {}
        for name, model in models.items():
            tokens, mask = P.embed(model, obs)
            _, residual = P.run_prefix(model, tokens, mask)
            pooled = P.mean_pool(residual, mask)  # (19, b, width)
            norm = P.mean_pool(jnp.linalg.norm(residual, axis=-1, keepdims=True), mask)[..., 0]  # (19, b)
            out[name] = (pooled, norm)
        return out

    run = pair.jit(features)
    pooled = {name: [] for name in ("base", "ft")}
    norms = {name: [] for name in ("base", "ft")}
    for i, (rows, obs, _) in enumerate(inputs.batches(args.batch_size)):
        for name, (feature, norm) in run(obs).items():
            pooled[name].append(np.asarray(feature)[:, : len(rows)].swapaxes(0, 1))  # (b, 19, width)
            norms[name].append(np.asarray(norm)[:, : len(rows)].T)
        if i % 10 == 0:
            print(f"batch {i + 1}: {rows[-1] + 1}/{len(inputs)} states", flush=True)
    np.savez(
        DATA / "f5_features.npz",
        ids=[s["id"] for s in states],
        episode=[s["episode_index"] for s in states],
        progress=[s["frame_index"] / s["episode_length"] for s in states],
        **{f"features_{name}": np.concatenate(v) for name, v in pooled.items()},  # (states, 19, width)
        **{f"token_norm_{name}": np.concatenate(v).mean(0) for name, v in norms.items()},  # (19,)
    )


# -- ridge regression, in numpy (n < width, so through the n x n kernel) ------------------


def standardize(train: np.ndarray, *others: np.ndarray):
    mean, std = train.mean(0), train.std(0) + 1e-8
    return mean, std, [(x - mean) / std for x in (train, *others)]


def ridge_path(x_train, y_train, x_test) -> np.ndarray:
    """Predictions on x_test for every strength in ALPHAS: (len(ALPHAS), n_test)."""
    eigenvalues, eigenvectors = np.linalg.eigh(x_train @ x_train.T)
    projected = eigenvectors.T @ (y_train - y_train.mean())
    cross = x_test @ x_train.T @ eigenvectors
    return np.stack([cross @ (projected / (eigenvalues + a)) for a in ALPHAS]) + y_train.mean()


def fold_of(groups: np.ndarray, folds: int, rng) -> np.ndarray:
    """A fold number for every sample, the same for all samples of a group (episode)."""
    unique = np.unique(groups)
    assignment = dict(zip(unique, rng.permutation(len(unique)) % folds))
    return np.array([assignment[g] for g in groups])


def best_alpha(x, y, groups, rng) -> int:
    """Index into ALPHAS with the lowest held-out squared error, over episode-wise folds of x."""
    fold = fold_of(groups, FOLDS - 1, rng)
    errors = np.zeros(len(ALPHAS))
    for f in range(FOLDS - 1):
        _, _, (a, b) = standardize(x[fold != f], x[fold == f])
        errors += np.square(ridge_path(a, y[fold != f], b) - y[fold == f]).sum(1)
    return int(errors.argmin())


def held_out_predictions(x, y, groups, seed: int, shuffle_labels: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = fold_of(groups, FOLDS, rng)
    predictions = np.zeros(len(y))
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        y_train = rng.permutation(y[train]) if shuffle_labels else y[train]
        alpha = best_alpha(x[train], y_train, groups[train], rng)
        _, _, (a, b) = standardize(x[train], x[test])
        predictions[test] = ridge_path(a, y_train, b)[alpha]
    return predictions


def r_squared(y, predictions, rows=slice(None)) -> float:
    y, predictions = y[rows], predictions[rows]
    return float(1.0 - np.square(y - predictions).sum() / np.square(y - y.mean()).sum())


def episode_bootstrap(y, predictions, groups, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    values = []
    for _ in range(n_boot):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        values.append(r_squared(y, predictions, rows))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def direction(x, y, groups, seed: int) -> np.ndarray:
    """The probe fit on all states, as a weight vector in residual coordinates."""
    alpha = ALPHAS[best_alpha(x, y, groups, np.random.default_rng(seed))]
    _, std, (z,) = standardize(x)
    weights = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(y)), y - y.mean())
    return weights / std


def probe(saved) -> tuple[dict, dict]:
    y, groups = saved["progress"], saved["episode"]
    result = {"states": len(y), "episodes": len(np.unique(groups)), "layers": list(range(saved["features_ft"].shape[1]))}
    directions = {"shift": (saved["features_ft"] - saved["features_base"]).mean(0)}
    for name in ("base", "ft"):
        features = saved[f"features_{name}"].astype(np.float64)
        scores, intervals, control, weights = [], [], [], []
        for r in range(features.shape[1]):
            x = features[:, r]
            predictions = held_out_predictions(x, y, groups, SEED)
            scores.append(r_squared(y, predictions))
            intervals.append(episode_bootstrap(y, predictions, groups))
            control.append(r_squared(y, held_out_predictions(x, y, groups, SEED, shuffle_labels=True)))
            weights.append(direction(x, y, groups, SEED))
            print(f"{name}, after {r} layers: held-out R^2 = {scores[-1]:.3f} (shuffled labels {control[-1]:.3f})", flush=True)
        result[name] = {"r2": scores, "ci": intervals, "r2_shuffled_labels": control}
        directions[f"progress_{name}"] = np.stack(weights)
        directions[f"meandiff_{name}"] = features[y > 2 / 3].mean(0) - features[y < 1 / 3].mean(0)
        directions[f"token_norm_{name}"] = saved[f"token_norm_{name}"]
    return result, directions


def plot(result: dict):
    plt = style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = result["layers"]
    for name, color, label in (("base", BASE_COLOR, "pi05_base"), ("ft", FT_COLOR, "pi05_libero (fine-tuned)")):
        scores, ci = np.array(result[name]["r2"]), np.array(result[name]["ci"])
        ax.plot(x, scores, marker="o", color=color, label=label)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.2)
    ax.plot(x, result["ft"]["r2_shuffled_labels"], color=GRAY, linestyle="--", label="control: shuffled progress labels")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("backbone layers applied (0 = token embeddings)")
    ax.set_ylabel("held-out R$^2$ of the progress probe")
    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.0)
    ax.legend(loc="best", fontsize=11)
    best = {name: int(np.argmax(result[name]["r2"])) for name in ("base", "ft")}
    ax.set_title(
        f"Task progress is linearly decodable: R$^2$ = {result['ft']['r2'][best['ft']]:.2f} fine-tuned "
        f"(layer {best['ft']}), {result['base']['r2'][best['base']]:.2f} base (layer {best['base']})"
    )
    save_figure(fig, "F5", f"Ridge probe from the mean-pooled prefix residual to task progress (frame index / "
                f"episode length), {result['states']} frames of {result['episodes']} LIBERO training episodes, "
                "5 folds split by episode; bands are 95% bootstrap intervals over episodes. Decodable does not "
                "mean used: F6 to F8 test whether the action expert reads this direction.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="use only the first N states (for a quick test)")
    parser.add_argument("--probe-only", action="store_true", help="refit from results/data/f5_features.npz (no GPU)")
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/data/f5.json")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    if not (args.probe_only or args.plot_only):
        extract(args)
    if not args.plot_only:
        result, directions = probe(np.load(DATA / "f5_features.npz"))
        (DATA / "f5.json").write_text(json.dumps(result, indent=1))
        np.savez(DATA / "f5_directions.npz", **directions)
    result = json.loads((DATA / "f5.json").read_text())
    plot(result)
    if not (args.probe_only or args.plot_only):
        log_run("F5 (progress probe)", started, seed=SEED, states=result["states"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
