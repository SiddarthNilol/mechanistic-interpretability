# CLAUDE.md: Preliminary results for Ideas A + C (VLA fine-tuning localization)

## Goal

Produce a small set of preliminary figures and a one-page summary that show my advisor (Prof Clark) what the full project will do. These are pilots, not final results. Each one should be cheap (under about 2 H200-hours), honest about what it can and cannot show, and point to the full experiment it previews.

Deliverables:

- `results/figures/F1.png` to `F7.png` (plus optional F8)
- `results/summary.md`: one paragraph per figure (what it shows, what it does not show, what the full experiment will do)
- `results/run_log.md`: every run with command, checkpoint, seed, number of states, and wall time

## Project in two sentences

Idea A: find, by cause and effect (activation patching), where few-shot fine-tuning changes a flow-matching VLA (pi0.5), measuring in-task gain, robustness and forgetting separately.
Idea C: find the "action-readable subspace" at each backbone layer (directions that move the action expert's output), and test whether a direction's readable fraction predicts whether it can steer the robot.

Motivation: my reproduction of Robotic Steering (task-specific head fine-tuning, ECCV 2026) failed on a real unhook-ring task. Progress is known to be linearly decodable from pi0.5 but not steerable (arXiv 2608.13474). RETAIN (arXiv 2512.08333) shows weight merging helps robustness but does not explain why.

## Environment

- Compute: NYU HPC Torch cluster, H200 GPUs, via SLURM. Ask me for the SLURM account name and partition before submitting the first job. Do not guess them.
- Code base: openpi (https://github.com/Physical-Intelligence/openpi). Clone it into `third_party/openpi`. Do not edit files inside it; put all our code in `src/`.
- Simulator: LIBERO (only needed for F8, the optional closed-loop pilot).
- Python env: follow openpi's install instructions exactly. Record the commit hash of openpi in `results/run_log.md`.

## Before writing any analysis code (verify, do not assume)

Check each item by reading the openpi code or docs, and write the answers at the top of `results/run_log.md`:

1. Which pi0.5 checkpoints openpi provides (expected names include `pi05_base` and `pi05_libero`; confirm the real names and download paths).
2. JAX or PyTorch: does openpi have a working PyTorch path for pi0.5? Pick ONE framework for all hooks and say why.
3. Architecture facts from the config: number of backbone layers, heads per layer, hidden width, action expert size, action horizon, real action dimensions for LIBERO (expected 7 of a padded vector).
4. How the action expert reads the backbone: confirm it attends to the prefix keys and values (KV cache) and find the exact function where prefix KV is computed and where the suffix (noisy actions + flow time) is processed.
5. How inference samples noise and how many flow integration steps it uses. We need to fix both for patching.
6. The LIBERO training data source openpi uses (expected a LeRobot-format dataset on Hugging Face) and how to load a few hundred states from it.

If any expected fact is wrong, stop and tell me before continuing.

## Important caveat for all pilots

`pi05_libero` was fine-tuned on full LIBERO data, not on 20 demos. So base vs pi05_libero is a "large fine-tune" pilot, not the few-shot setting of the real project. Every figure caption must say this. If time allows after F7, a 20-demo LoRA fine-tune on one LIBERO-10 task (see Stretch) gives a true few-shot pair.

## Shared setup for F3 to F7

- States: 200 to 300 observations sampled from LIBERO training demos (fixed list of episode and frame IDs saved to `results/states.json`). Use at least 3 LIBERO suites.
- Pairing rule: base and fine-tuned models get the same observation, the same prompt, the same noise sample and the same flow steps. Only the weights differ.
- Output metric L: mean squared error between a model's predicted action chunk and the fine-tuned model's chunk, real action dimensions only.
- Both models fit on one H200 together; run them side by side and patch online. Do not cache full activations to disk.
- Sanity check first (must pass before any figure): patching base activations into the base model reproduces base actions exactly (max absolute difference under 1e-5 in float32, or report the float tolerance you used).

## Figures

### F1. Model map (no GPU needed beyond loading)

Bar chart of parameter counts by module: vision encoder, language backbone, action expert, other. Add a small table in `summary.md` with layers, heads per layer, width, prefix token count per observation (images + text).
Purpose: shows the professor what parts exist and how big each is.

### F2. Where weights moved (static, not causal)

Heatmap: rows = modules (vision encoder, backbone, action expert), columns = layer index, value = relative weight change ||W_ft - W_base|| / ||W_base||, summed over each layer's matrices. Use pi05_base vs pi05_libero.
Caption must say: this shows where weights moved, not which changes matter. This is the Häon et al. / RETAIN-style view that Idea A goes beyond.

### F3. Module-level patching (first causal pilot of Idea A)

Recovery share for component c:

    R(c) = (L_base - L_base_with_c_from_ft) / (L_base - L_ft)

Components, both directions:

- Sufficiency (base run, fine-tuned piece swapped in): vision tokens only; full prefix KV (vision + backbone); action expert weights only.
- Necessity (fine-tuned run, base piece swapped in): same three.

Grouped bar chart, 6 bars with 95% bootstrap intervals over states.
Purpose: first evidence for or against "the gain lives in the action expert and the prefix KV link".

### F4. Layer-level patching

Line plot of R per backbone layer, for (a) the residual stream after layer l and (b) the prefix KV at layer l only. Sufficiency direction. One line each, with bootstrap bands.
Purpose: shows whether the gain is concentrated in a few layers or spread out.

### F5. Progress probe by layer (replicates 2608.13474 in miniature)

For each backbone layer, mean-pool the residual over prefix tokens, fit ridge regression to task progress (frame index / episode length), split by episode (no frame leakage). Plot held-out R squared by layer, for the base and fine-tuned models.
Save the probe weight vector at each layer as the "progress direction" for F7.

### F6. Action-readable spectrum (first pilot of Idea C)

For 3 backbone layers (early, middle, late), estimate M_l = E[J_l^T J_l], where J_l is the Jacobian of the action expert's predicted velocity (real action dims only) with respect to a vector added to every prefix token's residual at layer l.

Method (no full Jacobians):

1. For each of ~100 states and 3 flow times (early, mid, late), draw a few Gaussian vectors u in action space.
2. Compute g = J_l^T u with one vector-Jacobian product each.
3. Stack all g as columns of G. The top left singular vectors of G span the readable subspace (since E[J^T u u^T J] = E[J^T J] for standard Gaussian u).

Plot the eigenvalue spectrum (log scale) per layer and mark k = number of directions holding 90% of the total.
Purpose: shows how few directions the action expert actually reads.

### F7. Readable fraction of candidate directions

For each of the 3 layers, compute rho(v) = ||P_k v||^2 / ||v||^2 for:

- the progress probe direction from F5
- the fine-tuning shift: mean of (h_ft - h_base) over states at that layer
- 100 random Gaussian directions (show mean and spread; expected near k / width)

Bar chart per layer. Caption: H3 predicts progress has low rho, which would explain why it does not steer.

### F8 (optional). Closed-loop steering pilot

Only if F1 to F7 are done. In LIBERO, add alpha * v at one layer during rollouts of pi05_libero, for v = progress direction, its projection P_k v rescaled to the same norm, and a random direction. 3 alphas, 10 episodes each, one task. Plot success rate and mean end-effector speed against alpha. Label clearly as a 10-episode pilot.

## Stretch: a true few-shot pair

LoRA fine-tune pi05_base on 20 demos of one LIBERO-10 task (for example the mugs-on-plates task) with openpi's LoRA config. Re-run F3 on that pair. Only start this if I say so, since it costs more GPU time.

## Rules

- Never overclaim. Each summary paragraph has three parts: what it shows, what it does not show, which full experiment it previews.
- Fix and log all random seeds. Save the exact state list.
- Keep every job under about 2 H200-hours. If an estimate is higher, tell me first.
- Do not modify downloaded checkpoints. Do not commit checkpoints to git.
- Plots: matplotlib, one finding per figure, title states the finding (not the topic), axes labeled with units, readable at slide size. Save PNG at 200 dpi and PDF.
- If a result looks surprising (for example R above 1 or below 0 often), check the pairing rule and the sanity check before reporting it.
- Do not use em dashes in any written text.

## Suggested order

1. Verification list above, then F1 and F2 (no patching needed).
2. Patching hooks and the sanity check.
3. F3, then F4.
4. F5, then F6, then F7.
5. `results/summary.md`, then F8 if time allows.

## Background references

- Robotic Steering: https://arxiv.org/abs/2511.22697
- Häon et al., steering VLAs via FFN neurons: https://arxiv.org/abs/2509.00328
- RETAIN, weight merging for robust fine-tuning: https://arxiv.org/abs/2512.08333
- Decoding task progress from VLA representations: https://arxiv.org/abs/2608.13474
- Not All Layers Need Tuning: https://arxiv.org/abs/2609.18084
- pi0.5: https://arxiv.org/abs/2504.16054
- openpi: https://github.com/Physical-Intelligence/openpi
