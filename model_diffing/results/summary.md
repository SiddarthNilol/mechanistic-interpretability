# Preliminary results: where fine-tuning changes a VLA, and what its action expert reads

Status: F1 to F7 were run on 2026-11-11 (one H200, about 35 minutes of compute; details in
`run_log.md`). F6 and F7 were rerun the same day with 8 times the probe vectors (59 minutes). The sanity check passed exactly: patching a model's own activations into itself
changed the actions by 0, and our forward code matched openpi's with difference 0. F8 has not
been run.

Three things to take away, each a pilot result:

1. The action change of fine-tuning is carried mostly by what the backbone writes for the action
   expert, and is concentrated in the keys and values of backbone layers 15 and 16 (F3, F4).
   The weights, in contrast, moved almost uniformly everywhere (F2).
2. Task progress is decodable (R^2 about 0.8), but already from the token embeddings before any
   backbone layer, so the probe may be reading the images, not a computed progress signal (F5).
3. H3 is not supported: the progress direction is more readable by the action expert than a
   random direction, not less (F7). This held when F6 was rerun with 8 times the probe vectors.
   The readable subspace is not small: pooled over states it is at least 800 of 2048 directions
   after layers 3 and 9, and still growing with the sample (F6).

Caveat for every figure: `pi05_libero` was fine-tuned on the full LIBERO data, so base vs
`pi05_libero` is a large fine-tune. The few-demo regime is parked for now (decided 2026-11-11)
and comes later if this pair gives good results. Also, both
models are run on `pi05_libero`'s input format (10-action chunks, no robot state in the prompt),
which is not the format `pi05_base` was pretrained on (see `run_log.md`).

## F1. Model map

| module | parameters | layers | heads per layer | width |
|---|---|---|---|---|
| vision encoder (SigLIP So400m/14) | 415 M | 27 | 16 | 1152 |
| language backbone (Gemma 2B) | 2509 M | 18 | 8 | 2048 |
| action expert (Gemma 300M) | 428 M | 18 | 8 | 1024 |
| other (action and time projections) | 2.2 M | | | |

The backbone's count includes its 527 M token embedding table. Prefix tokens per observation:
3 camera slots x 256 image tokens + 200 prompt tokens = 968; in LIBERO two cameras are real, so
512 image tokens plus the prompt's own tokens (about 20) are attended to. Suffix: 10 action
tokens, 32 padded action dims, 7 real. (Numbers from the checkpoint metadata; `f1_f2_weights.py`
recomputes them from the weights and writes `results/data/f1_table.md`.)

- Shows: what parts exist and how big each is. The action expert is 13% of the parameters.
- Does not show: anything about function. Size is not importance.
- Previews: the modules are the units of the module-level patching in the full Idea A experiment.

## F2. Where weights moved

- Shows: fine-tuning moved the weights by a similar relative amount almost everywhere: 12% to
  19% per layer in all three modules (backbone mean 16.8%, vision encoder 16.0%, action expert
  14.3%). Two exceptions in the backbone: layer 17 (10%) and layer 18 (1.4%). Layer 18's
  residual output is read by nothing, so most of its weights get no training signal. The token
  embedding table (0.04%) and the image position embeddings (0.7%) barely moved.
- Does not show: which changes matter. A large change can be irrelevant to the actions and a
  small one decisive. This is the static view of Häon et al. and RETAIN.
- Previews: the comparison "static weight change vs causal recovery share" per layer in the full
  experiment.

## F3. Module-level patching

- Shows (240 states, ratio of means, 95% bootstrap intervals over states):

  | part | sufficiency (fine-tuned part into base) | necessity (base part into fine-tuned) |
  |---|---|---|
  | vision tokens | 0.09 [0.03, 0.15] | 0.25 [0.21, 0.30] |
  | prefix KV (vision + backbone) | 0.75 [0.70, 0.79] | 0.49 [0.44, 0.55] |
  | action expert | 0.51 [0.45, 0.56] | 0.25 [0.21, 0.30] |

  The prefix side carries more than the action expert: the base expert reading the fine-tuned
  prefix gets 75% of the way, the fine-tuned expert reading the base prefix gets 51%. The two
  sufficiencies add up to more than 1, so the parts overlap: either alone recovers much of the
  change. The fine-tuned backbone alone (base vision, base expert; in `f3.json`) recovers 59%,
  the vision encoder alone only 9%, and it made 38% of the states worse than the base model.
  The six bars hold four independent numbers (see `run_log.md`).
- Does not show: the few-demo regime; closed-loop success (L is an open-loop distance to the
  fine-tuned model's action chunk on training states); robustness or forgetting; and how much of
  L_base comes from the input-format mismatch.
- Previews: the same decomposition with in-task gain, robustness and forgetting measured
  separately in closed loop, and later on few-demo fine-tunes.

## F4. Layer-level patching

- Shows: the effect is concentrated late in the backbone. Replacing the keys and values of a
  single layer recovers 46% [42%, 51%] at layer 15 and 27% at layer 16; every other layer is at
  15% or below, and layers 1 to 9 are at 6% or below. The residual patch rises steadily from
  about 5% to 10% in the first layers to 66% after layer 14, then drops (28%, 12%, 4% after
  layers 15, 16, 17), because a later patch reaches fewer layers' keys and values.
- Does not show: single-layer KV patches do not add up to the whole (the layers interact), so a
  flat, low curve does not mean the prefix is unimportant. Sufficiency direction only.
- Previews: path-level patching (which layers' KV, read by which expert layers) in the full
  experiment, and the comparison with "Not All Layers Need Tuning".

## F5. Progress probe by layer

- Shows: progress is linearly decodable at every layer, held-out R^2 between 0.70 and 0.81
  (960 frames, 80 episodes, folds split by episode); best 0.81 for the base model (after layer
  8) and 0.79 for the fine-tuned model (after layer 12). The shuffled-label control is at 0.
  It is already 0.75 (base) and 0.70 (fine-tuned) at layer 0, the token embeddings, before any
  backbone layer. The backbone adds little, and fine-tuning does not increase it.
- Does not show: that the model computes or uses progress. Since the embeddings already give
  R^2 0.7, the probe is most likely reading how the scene looks (where the arm and objects are),
  which correlates with progress. A control is needed that removes this, for example probing
  progress within a fixed arm pose, or testing across tasks the probe was not trained on.
- Previews: the replication of arXiv 2608.13474 at full scale, and the source of the progress
  direction tested in F7 and F8.

## F6. Action-readable spectrum

- Shows (rerun: 240 states x 3 flow times x 16 probes = 11520 probe vectors per layer): 90% of
  the measured sensitivity lies in k = 802, 749 and 421 directions after layers 3, 9 and 15, out
  of 2048. These are lower bounds, not estimates: k still grows with the sample (575, 548 and
  304 on half of the states; 335, 313 and 197 in the first run with 1440 vectors). Adding states
  raises k faster than adding probes per state (the last doubling: +37% from states, +16% from
  probes, after layer 3), so different states are read along different directions. The pooled
  readable subspace is therefore not small. The sensitivity is still uneven: half of it lies in
  171, 138 and 25 directions, and after layer 15 the top 10 directions hold 32%. The total
  sensitivity after layer 15 is about 100 times smaller than after layers 3 and 9, because only
  layers 16 to 18 read that residual.
- Does not show: a converged k, or "the action expert reads few directions". For one state and
  one flow time the readable subspace has at most 70 directions (70 output numbers); the pooled
  one is large because it varies with the state. M is a first-order sensitivity at zero shift,
  on training states and three flow times.
- Previews: the full project should measure readability per state (or per task) and not through
  one pooled subspace with a fixed k, at every layer, for the base and the fine-tuned model.

## F7. Readable fraction of candidate directions

- Shows (fine-tuned model, on the rerun F6 subspace; chance = k / 2048, random directions have
  s.d. 0.01):

  | after layer | progress probe | fine-tuning shift | chance | gain, progress | gain, shift |
  |---|---|---|---|---|---|
  | 3 | 0.52 | 0.07 | 0.39 | 1.5 | 0.39 |
  | 9 | 0.56 | 0.07 | 0.37 | 2.2 | 0.29 |
  | 15 | 0.26 | 0.36 | 0.21 | 1.8 | 5.8 |

  Gain is v^T M v / ||v||^2 relative to an average direction; it does not depend on k. H3
  predicted a low rho for progress. The pilot shows the opposite: the progress direction is 1.2
  to 1.5 times more readable than chance at all three layers, with gain 1.5 to 2.2 (2.9 to 4.7
  for the late-minus-early mean difference). So "the action expert cannot see progress" does not
  explain why progress does not steer. The fine-tuning shift is far below chance after layers 3
  and 9 and far above it after layer 15, in line with F4: what fine-tuning changes becomes
  visible to the action expert late in the backbone. The gains barely moved between the two F6
  runs (first run: progress 1.5 to 2.1, shift 0.37, 0.29, 6.5), so these conclusions do not
  depend on the unconverged k. The rho values do: they rose with k.
- Does not show: steering. rho and gain are predictions about steering, not a test of it. The
  progress direction is one probe's weight vector, and F5 suggests it may encode how the scene
  looks. The pooled subspace hides that readability varies with the state (F6).
- Previews: the test of H3 in the full project: does readability predict, across many
  directions, which ones steer the robot in closed loop. The k-free gain, measured per state, is
  the better predictor to carry forward.

## F8 (optional). Closed-loop steering pilot

Not run.
