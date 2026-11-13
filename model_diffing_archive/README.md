# model_diffing: pilots for Ideas A + C

Preliminary figures on where fine-tuning changes pi0.5 (Idea A, activation patching) and which
backbone directions its action expert reads (Idea C). The plan and the rules are in `CLAUDE.md`.
Pair: `pi05_base` vs `pi05_libero`, a fine-tune on the full LIBERO data (every caption says so).
The few-demo regime is parked until this pair gives good results.

Everything runs on NYU HPC from this directory. The verified facts about openpi, and every run,
are in `results/run_log.md`. The conclusions go in `results/summary.md`.

## Files

| file | what it does | needs |
|---|---|---|
| `src/pair.py` | the two models side by side and the forward pieces: prefix pass with patching, suffix pass, flow steps | |
| `src/states.py`, `src/prepare.py` | the fixed state list (`results/states.json`) and its model inputs; downloads | internet |
| `src/check_pair.py` | our forward pieces equal openpi's; the patching sanity check | CPU (small random models) or GPU (`--real`) |
| `src/f1_f2_weights.py` | F1 model map, F2 where weights moved | CPU, 64 GB |
| `src/f3_module_patching.py` | F3 module-level patching, both directions | GPU |
| `src/f4_layer_patching.py` | F4 patching per backbone layer: residual stream and KV | GPU |
| `src/f5_progress_probe.py` | F5 progress probe per layer; saves the progress directions | GPU, then CPU |
| `src/f6_readable_spectrum.py` | F6 action-readable spectrum from vector-Jacobian products | GPU |
| `src/f7_readable_fraction.py` | F7 readable fraction of the progress and fine-tuning directions | CPU |
| `src/f8_*.py` | F8 optional closed-loop steering in LIBERO. Not run yet, see below | GPU + LIBERO |

Each figure script saves its numbers to `results/data/` and can redraw with `--plot-only`.
`--limit N` runs on the first N states for a quick test.

## Commands

Step by step in `COMMANDS.md`. In short: `source jobs/env.sh`, `python src/prepare.py` once on a
login node, then `sbatch jobs/weights_job.sh` (F1, F2) and `sbatch jobs/gpu_job.sh` (sanity check,
F3 to F7).

The environment is the existing lair-openpi one. That fork differs from upstream openpi only in
additions that are off by default (train-time RTC, loss options, dataset hooks); the inference
path the pilots use is the same. `check_pair.py --real` confirms it on the real models at the
start of every GPU job, and each run logs the openpi path and commit it imported.
`jobs/setup_env.sh` builds an upstream environment instead, if ever needed.

`python src/check_pair.py` (no arguments) tests the code on two small random models on a CPU in
about 2 minutes. Run it after any change to `src/pair.py`.

## F8 (optional, not run yet)

Only after F1 to F7. Needs the LIBERO simulator in its own Python 3.8 environment, set up as in
`third_party/openpi/examples/libero/README.md` ("Without Docker"), after
`git -C third_party/openpi submodule update --init --recursive`.

The commands are in `COMMANDS.md`, step 7.

The three F8 scripts were tested only up to the simulator: the steered policy answers requests
(on a small random model), and the plot was drawn from made-up records. Run one episode per
condition first (`--episodes 1`), and tell me the time estimate before the full run.

## Few-demo pair (parked)

Parked on 2026-11-11: the study uses the base and the fully fine-tuned model for now.
For later, the 20-demo LoRA pair: `pair.py` assumes plain weights: an openpi LoRA checkpoint keeps its
low-rank weights separate, so they must be merged into the plain weights when loading before F3
can be re-run on that pair.
