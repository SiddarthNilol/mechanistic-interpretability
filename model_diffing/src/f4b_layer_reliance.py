"""F4b: a control for F4. Which backbone layers does the action expert rely on in general?

    python src/f4b_layer_reliance.py

F4 found that the fine-tuned keys and values (KV) of layer 15 alone recover 46% of the
base-to-fine-tuned action change. That can mean "fine-tuning changed what layer 15 carries", or
only "the action expert leans most on layer 15, in any model". The architecture does not decide
it: expert layer l reads backbone layer l's KV, at all 18 layers. This control uses no second
model. Inside one model, the KV of a state are exchanged with the KV of another state (the
donor), which carries the wrong scene:

    restore l     donor KV at every layer, the state's own KV at layer l only. How much of the
                  state's own actions does layer l alone bring back? The analogue of F4 (b).
    scramble l    own KV at every layer, donor KV at layer l only. How much damage does it do?

L = squared error to the same model's own unpatched action chunk (7 real dims), with the same
noise. L_all = L with donor KV at every layer. As ratios of means over states:

    restore share(l)  = 1 - L_restore(l) / L_all       1 = layer l alone is enough
    scramble share(l) = L_scramble(l) / L_all           1 = as bad as scrambling every layer

Donors have exactly the same prefix mask as the state (same number of prompt tokens), so only
the content of the KV changes. Two kinds, each drawn with the seed:

    scene    same task (same prompt), the other episode: only the images differ
    task     a different task whose prompt has the same number of tokens: images and prompt
             differ. States with no such donor are left out of this kind.

Run for both models. If both show the F4 profile, the F4 peak says where the action expert
reads, not where fine-tuning acted.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import BASE_COLOR, DATA, FT_COLOR, GRAY, REAL_DIMS, SEED, bootstrap_ci, log_run, save_figure, style

KINDS = ("scene", "task")
MODELS = (("base", BASE_COLOR, "pi05_base"), ("ft", FT_COLOR, "pi05_libero"))


def choose_donors(states: list[dict], prompt_mask: np.ndarray, seed: int) -> dict:
    """For every state, the row of its donor of each kind, or -1 if there is none."""
    rng = np.random.default_rng(seed)
    length = prompt_mask.sum(axis=1)
    task = np.array([s["task_index"] for s in states])
    episode = np.array([s["episode_index"] for s in states])
    donors = {kind: np.full(len(states), -1) for kind in KINDS}
    for i in range(len(states)):
        candidates = {
            "scene": np.flatnonzero((task == task[i]) & (episode != episode[i])),
            "task": np.flatnonzero((task != task[i]) & (length == length[i])),
        }
        for kind, rows in candidates.items():
            if len(rows):
                donors[kind][i] = rng.choice(rows)
    return donors


def reliance(models, obs, donor, noise):
    """Per model: L of "scramble l" and "restore l" (18, b), and of donor KV everywhere (b)."""
    import jax
    import jax.numpy as jnp

    import pair as P

    obs, donor = P.preprocess(obs), P.preprocess(donor)
    out = {}
    for name, model in models.items():
        tokens, mask = P.embed(model, obs)
        kv, _ = P.run_prefix(model, tokens, mask)
        donor_kv, _ = P.run_prefix(model, *P.embed(model, donor))
        clean = P.denoise(model, obs, mask, kv, noise)[..., :REAL_DIMS]

        def loss(cache, model=model, mask=mask, clean=clean):
            chunk = P.denoise(model, obs, mask, cache, noise)[..., :REAL_DIMS]
            return jnp.square(chunk - clean).mean(axis=(1, 2))

        def one_layer(l, kv=kv, donor_kv=donor_kv, loss=loss):
            swap = lambda a, b: jax.tree.map(lambda x, y: x.at[l].set(y[l]), a, b)  # noqa: E731
            return loss(swap(kv, donor_kv)), loss(swap(donor_kv, kv))

        scramble, restore = jax.lax.map(one_layer, jnp.arange(kv[0].shape[0]))
        out[name] = {"scramble": scramble, "restore": restore, "all": loss(donor_kv)}
    return out


def compute(args) -> dict:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    import pair as P
    import states as S

    pair = P.Pair.load()
    states = S.load_states(patch_only=True)[: args.limit]
    inputs = S.Inputs(states)
    prompt_mask = np.asarray(inputs.data["tokenized_prompt_mask"])
    donors = choose_donors(states, prompt_mask, args.seed)
    run = pair.jit(reliance)

    def observation(rows):
        return _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x[rows]), inputs.data))

    saved = {"ids": np.array([s["id"] for s in states])}
    for kind in KINDS:
        valid = donors[kind] >= 0
        donor_rows = np.where(valid, donors[kind], np.arange(len(states)))  # no donor: itself (L = 0), left out later
        assert (prompt_mask[donor_rows] == prompt_mask).all(), "a donor's prefix mask differs from its state's"
        parts = {name: {"scramble": [], "restore": [], "all": []} for name, _, _ in MODELS}
        for start in range(0, len(states), args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, len(states)))
            padded = np.pad(rows, (0, args.batch_size - len(rows)), mode="edge")
            out = run(observation(padded), observation(donor_rows[padded]), inputs.noise(padded))
            for name, values in out.items():
                for key, value in values.items():
                    parts[name][key].append(np.asarray(value)[..., : len(rows)])
            print(f"{kind} donors: {rows[-1] + 1}/{len(states)} states", flush=True)
        saved[f"{kind}_valid"] = valid
        saved[f"{kind}_donor_id"] = np.where(valid, saved["ids"][donor_rows], -1)
        for name, values in parts.items():
            saved[f"{kind}_{name}_all"] = np.concatenate(values["all"])  # (states,)
            for key in ("scramble", "restore"):
                saved[f"{kind}_{name}_{key}"] = np.concatenate(values[key], axis=1).T  # (states, 18)
    np.savez(DATA / "f4b.npz", **saved)
    return saved


def summarize(saved) -> dict:
    result = {"states": int(len(saved["ids"])), "kinds": {}}
    for kind in KINDS:
        valid = np.asarray(saved[f"{kind}_valid"])
        n = int(valid.sum())
        entry = {"states": n}
        for name, _, _ in MODELS:
            l_all = saved[f"{kind}_{name}_all"][valid]
            shares = {"L_all_mean": float(l_all.mean())}
            for key in ("scramble", "restore"):
                losses = saved[f"{kind}_{name}_{key}"][valid]
                values, intervals = [], []
                for j in range(losses.shape[1]):
                    ratio = lambda rows, p=losses[:, j]: float(p[rows].mean() / l_all[rows].mean())  # noqa: E731
                    share = ratio if key == "scramble" else (lambda rows, ratio=ratio: 1.0 - ratio(rows))
                    values.append(share(np.arange(n)))
                    intervals.append(bootstrap_ci(share, n, n_boot=1000))
                shares[key] = {"share": values, "ci": intervals}
            entry[name] = shares
        result["kinds"][kind] = entry
    result["x"] = list(range(1, len(result["kinds"]["scene"]["base"]["restore"]["share"]) + 1))
    return result


def plot(result: dict):
    from matplotlib.lines import Line2D

    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
    x = result["x"]
    panels = (("restore", "restore share: own keys and values at layer x only\n(1 = layer x alone gives back the actions)"),
              ("scramble", "scramble share: donor keys and values at layer x only\n(1 = as bad as scrambling every layer)"))
    for ax, (key, ylabel) in zip(axes, panels):
        for name, color, _ in MODELS:
            for kind, linestyle in zip(KINDS, ("-", "--")):
                data = result["kinds"][kind][name][key]
                values, ci = np.array(data["share"]), np.array(data["ci"])
                ax.plot(x, values, marker="o", markersize=4, color=color, linestyle=linestyle)
                if kind == "scene":
                    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xlabel("backbone layer x")
        ax.set_ylabel(ylabel)
    f4 = DATA / "f4.json"
    handles = [Line2D([], [], color=color, marker="o", markersize=4, label=label) for _, color, label in MODELS]
    if f4.exists():  # the curve this control is for
        kv = json.loads(f4.read_text())["kv"]
        axes[0].plot(kv["x"], kv["R"], color=GRAY, linewidth=2, linestyle=":")
        handles.append(Line2D([], [], color=GRAY, linewidth=2, linestyle=":", label="F4 (b): fine-tuned KV into base"))
    handles += [Line2D([], [], color="black", linestyle="-", label="donor: same task, other episode"),
                Line2D([], [], color="black", linestyle="--", label="donor: other task")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=10)
    restore = {name: np.array(result["kinds"]["scene"][name]["restore"]["share"]) for name, _, _ in MODELS}
    peaks = {name: x[int(np.argmax(values))] for name, values in restore.items()}
    where = (f"layer {peaks['base']}" if peaks["base"] == peaks["ft"]
             else f"layers {peaks['base']} (base) and {peaks['ft']} (fine-tuned)")
    axes[0].set_title(f"With the wrong scene in every other layer, the keys and values of {where} alone give back "
                      f"{restore['base'].max():.0%} (base) and {restore['ft'].max():.0%} (fine-tuned) of the actions")
    scene, task = result["kinds"]["scene"]["states"], result["kinds"]["task"]["states"]
    save_figure(fig, "F4b", "Control for F4, inside one model at a time: a state's prefix keys and values are "
                f"exchanged with those of another state with the same prefix mask ({scene} LIBERO training states "
                f"with a same-task donor, {task} with an other-task donor). Error is measured to the same model's "
                "own unpatched actions; bands are 95% bootstrap intervals over states (same-task donors). "
                "This shows which layers the action expert reads the scene from, not what fine-tuning changed.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the donor choice")
    parser.add_argument("--limit", type=int, help="use only the first N patch states (for a quick test)")
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/data/f4b.npz")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    saved = np.load(DATA / "f4b.npz") if args.plot_only else compute(args)
    result = summarize(saved)
    (DATA / "f4b.json").write_text(json.dumps(result, indent=1))
    for kind in KINDS:
        for name, _, _ in MODELS:
            for key in ("restore", "scramble"):
                values = np.round(result["kinds"][kind][name][key]["share"], 3).tolist()
                print(f"{kind} donors ({result['kinds'][kind]['states']} states), {name}, {key} share per layer = {values}")
    plot(result)
    if not args.plot_only:
        log_run("F4b (per-layer reliance control)", started, seed=args.seed, states=result["states"],
                states_with_same_task_donor=result["kinds"]["scene"]["states"],
                states_with_other_task_donor=result["kinds"]["task"]["states"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
