"""F3: module-level activation patching. One GPU, a few minutes.

    python src/f3_module_patching.py

pi0.5 has three parts in a chain: the vision encoder makes image tokens, the backbone turns the
prefix (image tokens + prompt) into the KV cache, the action expert reads the KV cache. For every
state this runs all 8 ways of taking each part from the base or the fine-tuned model (same
observation, prompt, noise and flow steps; only the weights differ), and measures

    L = mean squared error to the fine-tuned model's action chunk (7 real action dims,
        normalized action units), so L_ft = 0

    sufficiency of a part   base run with the fine-tuned part swapped in:
                            R = (L_base - L_patched) / (L_base - L_ft)
    necessity of a part     fine-tuned run with the base part swapped in:
                            R = (L_patched - L_ft) / (L_base - L_ft)

Both are 1 if the part carries the whole change and 0 if it carries none. The parts, as in
CLAUDE.md: vision tokens only; the full prefix KV (vision encoder + backbone); the action expert.

Because the action expert reads nothing but the KV cache, two pairs of bars are the same run:
"fine-tuned prefix KV into base" is "base action expert into fine-tuned", so sufficiency(prefix
KV) + necessity(action expert) = 1 exactly, and likewise sufficiency(expert) + necessity(prefix
KV) = 1. The six bars hold four independent numbers.

R is a ratio of means over states; intervals are 95% bootstrap over states.
"""

from __future__ import annotations

import argparse
import json
import time

import jax.numpy as jnp
import numpy as np

from model_diffing.archive.src.common import BASE_COLOR, DATA, FT_COLOR, REAL_DIMS, SEED, bootstrap_ci, log_run, save_figure, style

NAMES = ("base", "ft")
# part -> (vision encoder, backbone, action expert) sources of the sufficiency and the necessity run
PARTS = {
    "vision tokens": ("ft/base/base", "base/ft/ft"),
    "prefix KV\n(vision + backbone)": ("ft/ft/base", "base/base/ft"),
    "action expert": ("base/base/ft", "ft/ft/base"),
}


def compute(args) -> dict:
    import model_diffing.archive.src.pair as P
    import model_diffing.archive.src.states as S

    pair = P.Pair.load()
    inputs = S.Inputs(S.load_states(patch_only=True)[: args.limit])

    def chunks(models, obs, noise):
        """The action chunk (b, horizon, 7) of all 8 mixtures, keyed 'vision/backbone/expert'."""
        obs = P.preprocess(obs)
        embedded = {name: P.embed(model, obs) for name, model in models.items()}
        mask = embedded["base"][1]
        n_image = P.n_image_tokens(obs, embedded["base"][0])
        out = {}
        for vision in NAMES:
            for backbone in NAMES:  # the backbone's model also embeds the prompt
                tokens = jnp.concatenate([embedded[vision][0][:, :n_image], embedded[backbone][0][:, n_image:]], axis=1)
                kv, _ = P.run_prefix(models[backbone], tokens, mask)
                for expert in NAMES:
                    actions = P.denoise(models[expert], obs, mask, kv, noise)
                    out[f"{vision}/{backbone}/{expert}"] = actions[..., :REAL_DIMS]
        return out

    run = pair.jit(chunks)
    collected: dict[str, list] = {}
    for i, (rows, obs, noise) in enumerate(inputs.batches(args.batch_size)):
        for key, value in run(obs, noise).items():
            collected.setdefault(key, []).append(np.asarray(value)[: len(rows)])
        print(f"batch {i + 1}: {rows[-1] + 1}/{len(inputs)} states", flush=True)
    actions = {key: np.concatenate(value) for key, value in collected.items()}
    target = actions["ft/ft/ft"]
    losses = {key: np.square(value - target).mean(axis=(1, 2)) for key, value in actions.items()}
    np.savez(DATA / "f3.npz", ids=[s["id"] for s in inputs.states],
             **{f"loss:{k}": v for k, v in losses.items()}, **{f"actions:{k}": v for k, v in actions.items()})
    return losses


def summarize(losses: dict) -> dict:
    l_base = losses["base/base/base"]
    n = len(l_base)
    result = {"states": n, "L_base_mean": float(l_base.mean()), "bars": {}, "all_mixtures": {}}
    for part, (sufficient, necessary) in PARTS.items():
        for kind, key in (("sufficiency", sufficient), ("necessity", necessary)):
            patched = losses[key]
            if kind == "sufficiency":
                value = lambda rows, p=patched: float(1.0 - p[rows].mean() / l_base[rows].mean())  # noqa: E731
            else:
                value = lambda rows, p=patched: float(p[rows].mean() / l_base[rows].mean())  # noqa: E731
            result["bars"][f"{kind}: {part}"] = {
                "R": value(np.arange(n)), "ci": bootstrap_ci(value, n), "mixture": key,
                # check for surprises: how often the patch makes a state worse than the base model
                "fraction_of_states_worse_than_base": float((patched > l_base).mean()),
            }
    for key, patched in losses.items():
        result["all_mixtures"][key] = {"L_mean": float(patched.mean()), "L_over_L_base": float(patched.mean() / l_base.mean())}
    return result


def plot(result: dict):
    plt = style()
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = list(PARTS)
    x = np.arange(len(parts))
    for offset, kind, color in ((-0.2, "sufficiency", BASE_COLOR), (0.2, "necessity", FT_COLOR)):
        bars = [result["bars"][f"{kind}: {part}"] for part in parts]
        values = np.array([b["R"] for b in bars])
        errors = np.abs(np.array([b["ci"] for b in bars]).T - values)
        label = ("sufficiency: base run, fine-tuned part swapped in" if kind == "sufficiency"
                 else "necessity: fine-tuned run, base part swapped in")
        ax.bar(x + offset, values, 0.38, yerr=errors, capsize=4, color=color, label=label)
        for xi, v in zip(x + offset, values):
            ax.text(xi, max(v, 0) + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x, parts)
    ax.set_ylabel("recovery share R (fraction of the\nbase-to-fine-tuned action change)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=11)
    sufficiency = {part: result["bars"][f"sufficiency: {part}"]["R"] for part in parts}
    best = max(sufficiency, key=sufficiency.get)
    name = best.split("\n")[0]
    ax.set_title(f"Swapping in the fine-tuned {name} alone recovers {sufficiency[best]:.0%} of the action change")
    save_figure(fig, "F3", f"Module-level activation patching on {result['states']} LIBERO training states; error "
                "bars are 95% bootstrap intervals over states. L is the squared error to the fine-tuned model's "
                "action chunk. By construction sufficiency(prefix KV) + necessity(action expert) = 1 and "
                "sufficiency(action expert) + necessity(prefix KV) = 1, because the expert reads only the KV cache.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="use only the first N patch states (for a quick test)")
    parser.add_argument("--plot-only", action="store_true", help="redraw from results/data/f3.npz")
    args = parser.parse_args()
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        saved = np.load(DATA / "f3.npz")
        losses = {k.removeprefix("loss:"): saved[k] for k in saved.files if k.startswith("loss:")}
    else:
        losses = compute(args)
    result = summarize(losses)
    (DATA / "f3.json").write_text(json.dumps(result, indent=1))
    for name, bar in result["bars"].items():
        print(f"{name.replace(chr(10), ' ')}: R = {bar['R']:.3f} [{bar['ci'][0]:.3f}, {bar['ci'][1]:.3f}]")
        if not -0.05 <= bar["R"] <= 1.05:
            print("  SURPRISING: R outside [0, 1]. Check the pairing rule and check_pair.py before reporting.")
    plot(result)
    if not args.plot_only:
        log_run("F3 (module-level patching)", started, seed=SEED, states=result["states"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
