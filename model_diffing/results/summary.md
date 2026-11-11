# Preliminary results: where fine-tuning changes a VLA, and what its action expert reads

Status: the scripts are written and tested on small random models; **no figure has been produced
on the real models yet**. Lines marked TODO are filled in from `results/data/*.json` after the
runs. The "does not show" and "previews" parts do not depend on the numbers.

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

- Shows: TODO (which module and layers have the largest relative weight change).
- Does not show: which changes matter. A large change can be irrelevant to the actions and a
  small one decisive. This is the static view of Häon et al. and RETAIN.
- Previews: the comparison "static weight change vs causal recovery share" per layer in the full
  experiment.

## F3. Module-level patching

- Shows: TODO (R for the three parts, both directions, with intervals). Note that the six bars
  hold four independent numbers (see `run_log.md`).
- Does not show: the few-demo regime; closed-loop success (L is an open-loop distance to the
  fine-tuned model's action chunk on training states); robustness or forgetting; and how much of
  L_base comes from the input-format mismatch.
- Previews: the same decomposition with in-task gain, robustness and forgetting measured
  separately in closed loop, and later on few-demo fine-tunes.

## F4. Layer-level patching

- Shows: TODO (whether R is concentrated in a few layers or spread out; the residual curve and
  the single-layer KV curve).
- Does not show: single-layer KV patches do not add up to the whole (the layers interact), so a
  flat, low curve does not mean the prefix is unimportant. Sufficiency direction only.
- Previews: path-level patching (which layers' KV, read by which expert layers) in the full
  experiment, and the comparison with "Not All Layers Need Tuning".

## F5. Progress probe by layer

- Shows: TODO (held-out R^2 per layer for both models, against the shuffled-label control).
- Does not show: that the model uses progress. A linear probe finds what is decodable. Progress
  is also correlated with what is visible in the images, so the probe may read scene content.
- Previews: the replication of arXiv 2608.13474 at full scale, and the source of the progress
  direction tested in F7 and F8.

## F6. Action-readable spectrum

- Shows: TODO (k per layer, with the split-half values).
- Does not show: a global property. M is a first-order sensitivity at zero shift, on training
  states and three flow times. The estimate uses 1440 probe vectors per layer and cannot resolve
  more directions than that; if k comes out near that ceiling, more probes are needed.
- Previews: the full estimate of the readable subspace at every layer, for the base and the
  fine-tuned model, and how fine-tuning changes it.

## F7. Readable fraction of candidate directions

- Shows: TODO (rho of the progress direction and of the fine-tuning shift against chance k / 2048,
  and the k-free gain from `f7.json`).
- Does not show: steering. rho is a prediction about steering, not a test of it. The progress
  direction is one probe's weight vector; `f7.json` also has rho for the late-minus-early mean
  difference as a second definition.
- Previews: the test of H3 in the full project: does rho predict, across many directions, which
  ones steer the robot in closed loop.

## F8 (optional). Closed-loop steering pilot

Not run.
