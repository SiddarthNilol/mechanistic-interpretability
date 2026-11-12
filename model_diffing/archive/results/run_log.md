# Run log

openpi, as read for the verification below: upstream commit `215abfb217dbac7d5f1273282331b9b1866c0479`
(2026-08-24), cloned to `third_party/openpi`, not edited.

openpi, as run on HPC: the lair-openpi fork in `$SCRATCH/lair-openpi/.venv` (decided 2026-11-11).
Compared with upstream at the local copy's commit `aec5d4d`: `pi0.py`, `gemma.py`,
`pi0_config.py` and `training/config.py` differ only by additions that are off by default
(train-time RTC with per-token flow times, `loss_action_dim`, dataset and eval hooks);
`model.py`, `siglip.py`, `lora.py`, `tokenizer.py`, `transforms.py`, `libero_policy.py`,
`download.py` and `normalize.py` are identical, and `pi05_libero`'s config is the same. Every
run below logs the path and commit of the openpi it imported.

## Verified before writing analysis code

Checked on 2026-11-11 by reading the openpi code at the commit above, the checkpoint metadata in
the `openpi-assets` bucket, and the dataset's metadata on Hugging Face. Nothing here has been
run on a GPU yet.

1. **Checkpoints.** `gs://openpi-assets/checkpoints/pi05_base` and `gs://openpi-assets/checkpoints/pi05_libero`
   exist (12.4 GB each; openpi also has `pi05_droid` and `pi05_droid_jointpos`). Each has `params/`
   (an orbax checkpoint) and `assets/`. `openpi.shared.download.maybe_download` caches them under
   `$OPENPI_DATA_HOME`. The two checkpoints hold the same 51 weight arrays under the same names.
   `pi05_libero/assets/physical-intelligence/libero/norm_stats.json` has the normalization
   statistics; `pi05_base` has none for LIBERO.
2. **Framework: JAX.** openpi has a working PyTorch path for pi0.5 (`src/openpi/models_pytorch`,
   validated on LIBERO), but it needs every checkpoint converted first and files of the installed
   `transformers` package overwritten. The released checkpoints are JAX, swapping a module between
   models is a swap of entries in the weight tree, and vector-Jacobian products (F6) are built in.
   The cost: openpi runs the 18 layers inside one `scan`, so a single layer cannot be hooked.
   `src/pair.py` therefore has its own copy of the backbone's prefix loop (about 40 lines), and
   `src/check_pair.py` tests it against openpi (exact agreement on small random models).
3. **Architecture** (`gemma.get_config`, `pi0_config.Pi0Config`, shapes in the checkpoint):
   backbone = Gemma 2B, 18 layers, 8 heads per layer (sharing 1 key/value head), width 2048,
   head dim 256, MLP 16384. Action expert = Gemma 300M, 18 layers, 8 heads, width 1024, MLP 4096.
   Vision encoder = SigLIP So400m/14, 27 layers, 16 heads, width 1152, 256 tokens per image.
   Parameters: 3353 M total = vision 415 M + backbone 2509 M (of which 527 M token embeddings)
   + action expert 428 M + projections 2.2 M. Action dim 32 (padded), 7 real for LIBERO
   (`LiberoOutputs` keeps `[..., :7]`). Action horizon: 50 by default, **10 in `pi05_libero`**.
   Prompt length 200 tokens, so the prefix is 3 x 256 + 200 = 968 tokens.
4. **How the action expert reads the backbone.** Confirmed: through the prefix keys and values
   only. `Pi0.sample_actions` (`src/openpi/models/pi0.py`) computes the prefix KV cache once,
   `_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)`, and its inner `step` function
   processes the suffix: `embed_suffix` (noisy actions through `action_in_proj`; the flow time
   through `time_mlp_in/out`, used as the condition of the expert's adaptive RMS norms), then
   `self.PaliGemma.llm([None, suffix_tokens], kv_cache=kv_cache, adarms_cond=...)`. In
   `gemma.Attention.__call__` the cached keys and values are concatenated before the suffix's own.
5. **Noise and flow steps.** `sample_actions` draws `jax.random.normal(rng, (batch, horizon, 32))`
   unless `noise=` is given (the policy splits its key on every call), and integrates with
   `num_steps=10` Euler steps from time 1 (noise) to 0 (actions). We fix both: one noise sample per
   state from seed 0 and the state's id, and 10 steps.
6. **Data.** `physical-intelligence/libero` on Hugging Face (LeRobot v2.0): 1693 episodes, 273,465
   frames, 40 tasks, 10 fps, one parquet file per episode with both 256x256 images embedded, an
   8-dim state and 7-dim actions. Tasks are numbered suite by suite (10 each: libero_10,
   libero_goal, libero_object, libero_spatial). We read the parquet files of the sampled episodes
   directly and apply openpi's own `pi05_libero` input transforms.

### Facts that CLAUDE.md does not anticipate

- **The two models were trained on different input formats.** `pi05_libero` uses an action
  horizon of 10 and `discrete_state_input=False`: the robot state is not in the prompt, and pi0.5
  has no other state input, so `pi05_libero` does not see proprioception at all. `pi05_base` was
  pretrained with horizon 50 and the state written into the prompt. The pairing rule needs one
  input for both, so both models are run in `pi05_libero`'s format. The base model is then
  somewhat outside its pretraining format, and part of L_base reflects that, not only missing
  LIBERO knowledge.
- **The third camera slot is masked** for LIBERO (two real cameras), so 256 of the 968 prefix
  tokens are never attended to.
- **Six bars, four numbers (F3).** The action expert reads only the KV cache, so "fine-tuned prefix
  KV into the base run" and "base action expert into the fine-tuned run" are the same computation.
  Sufficiency(prefix KV) + necessity(action expert) = 1 exactly, and the same for the other pair.
- **The residual after layer 18 is read by nothing.** F4 shows residual patches after layers 0 to
  17, and F6's "late" layer is 15.

### Choices

- float32 everywhere (openpi's inference default is bfloat16), and full-precision matrix products,
  so that the sanity check can be exact. Sanity check tolerance: 1e-5 (max absolute difference).
- L is measured in normalized action units (the model's output space, `pi05_libero`'s quantile
  normalization), over all 10 actions of the chunk and the 7 real dims.
- R is a ratio of means over states, not a mean of per-state ratios (stable when L_base is small).
- States: seed 0, 2 episodes per task (80 episodes, 20 per suite), 12 frames per episode (960
  states, used by F5), 3 of them per episode for patching (240 states; F6 uses every other, 120).

### Decided with Siddarth, 2026-11-11

- SLURM: account `torch_pr_564_tandon_advanced`, partitions `h200_tandon` and `h200_public` (H200 GPUs).
- Running `pi05_base` in `pi05_libero`'s input format is accepted for these pilots. The study stays
  in simulation.

- Scope: the few-demo regime is parked. The study uses `pi05_base` vs the fully fine-tuned
  `pi05_libero` for now; low-data fine-tunes come later if this pair gives good results.

## Runs

Every script appends its run below (command, checkpoints, seed, states, wall time).

### F1 + F2 (weights), 2026-11-11 17:57

- command: `python src/f1_f2_weights.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22746373
- seed: 0 (not used: no sampling)
- states: 0 (weights only)
- wall time: 2.9 min

### check_pair (sanity check), 2026-11-11 18:25

- command: `python src/check_pair.py --real`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22749249
- seed: 0
- states: 4
- ours vs openpi, KV cache: 0.00e+00 (max abs difference, float32)
- ours vs openpi, action chunk: 0.00e+00 (max abs difference, float32)
- self-patch, residual, worst layer: 0.00e+00 (max abs difference, float32)
- self-patch, KV, worst layer: 0.00e+00 (max abs difference, float32)
- adding a zero vector, worst layer: 0.00e+00 (max abs difference, float32)
- cross-model patch, smallest effect over layers: 5.63e-02 (max abs difference, float32)
- wall time: 2.0 min

### F3 (module-level patching), 2026-11-11 18:28

- command: `python src/f3_module_patching.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22749249
- seed: 0
- states: 240
- batch size: 8
- wall time: 3.0 min

### F4 (layer-level patching), 2026-11-11 18:37

- command: `python src/f4_layer_patching.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22749249
- seed: 0
- states: 240
- batch size: 8
- wall time: 9.5 min

### F5 (progress probe), 2026-11-11 18:44

- command: `python src/f5_progress_probe.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22749249
- seed: 0
- states: 960
- batch size: 16
- wall time: 6.8 min

### F6 (action-readable spectrum), 2026-11-11 18:56

- command: `python src/f6_readable_spectrum.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22749249
- seed: 0
- states: 120
- model: ft
- probes: 4
- batch size: 4
- wall time: 11.6 min

### check_pair (sanity check), 2026-11-11 21:05

- command: `python src/check_pair.py --real`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22769350
- seed: 0
- states: 4
- ours vs openpi, KV cache: 0.00e+00 (max abs difference, float32)
- ours vs openpi, action chunk: 0.00e+00 (max abs difference, float32)
- self-patch, residual, worst layer: 0.00e+00 (max abs difference, float32)
- self-patch, KV, worst layer: 0.00e+00 (max abs difference, float32)
- adding a zero vector, worst layer: 0.00e+00 (max abs difference, float32)
- cross-model patch, smallest effect over layers: 5.63e-02 (max abs difference, float32)
- wall time: 2.1 min

### F6 (action-readable spectrum), 2026-11-11 22:04

- command: `python src/f6_readable_spectrum.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22769350
- seed: 0
- states: 240
- model: ft
- probes: 16
- batch size: 4
- wall time: 59.1 min

### check_pair (sanity check), 2026-11-11 22:09

- command: `python src/check_pair.py --real`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22775605
- seed: 0
- states: 4
- ours vs openpi, KV cache: 0.00e+00 (max abs difference, float32)
- ours vs openpi, action chunk: 0.00e+00 (max abs difference, float32)
- self-patch, residual, worst layer: 0.00e+00 (max abs difference, float32)
- self-patch, KV, worst layer: 0.00e+00 (max abs difference, float32)
- adding a zero vector, worst layer: 0.00e+00 (max abs difference, float32)
- cross-model patch, smallest effect over layers: 5.63e-02 (max abs difference, float32)
- wall time: 2.1 min

### F4b (per-layer reliance control), 2026-11-11 22:19

- command: `python src/f4b_layer_reliance.py`
- checkpoints: base `gs://openpi-assets/checkpoints/pi05_base`, fine-tuned `gs://openpi-assets/checkpoints/pi05_libero`
- openpi: `aec5d4d9effe4206d5a92606ab9343a0436fbd24` at `/scratch/sk12590/lair-openpi`
- SLURM job: 22775605
- seed: 0
- states: 240
- states with same task donor: 240
- states with other task donor: 228
- batch size: 8
- wall time: 10.0 min
