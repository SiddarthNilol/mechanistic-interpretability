"""The fixed list of LIBERO states (results/states.json) and their model inputs.

The data is openpi's LIBERO training set, the LeRobot dataset physical-intelligence/libero on
Hugging Face: 1693 demos of 40 tasks, one parquet file per episode with the two 256x256 camera
images, the 8-dim state and the 7-dim action of every frame. Its tasks are numbered suite by
suite (the order of openpi's conversion script), 10 tasks each, so suite = task_index // 10.

A state is one frame of one episode. Its model input is built by openpi's own transforms for
pi05_libero (LiberoInputs, quantile normalization with the checkpoint's statistics, resize to
224, prompt tokenization), so both models see exactly what pi05_libero saw in training. Each
state also gets its own fixed noise sample, drawn from the seed and the state's id.
"""

from __future__ import annotations

import io
import json

import numpy as np

from model_diffing.archive.src.common import SEED, STATES_FILE

DATASET = "physical-intelligence/libero"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")  # task_index // 10


def dataset_file(filename: str, revision: str | None = None) -> str:
    """Local path of one file of the dataset (downloaded to the Hugging Face cache on first use)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, filename, repo_type="dataset", revision=revision)


def episode_file(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def load_states(*, patch_only: bool = False) -> list[dict]:
    """The states of results/states.json: all of them (used by the progress probe), or only the
    subset marked for patching (F3, F4, F6)."""
    states = json.loads(STATES_FILE.read_text())["states"]
    return [s for s in states if s["patch"]] if patch_only else states


def transforms():
    """openpi's transforms of the pi05_libero policy, as (input, output): raw frame -> model
    input, and model output -> LIBERO action (un-normalized, 7 dims)."""
    import openpi.transforms as _transforms
    from openpi.shared import download
    from openpi.shared import normalize
    from openpi.training import config as _config

    import model_diffing.archive.src.pair as P

    train = _config.get_config(P.CONFIG_NAME)
    data = train.data.create(train.assets_dirs, train.model)
    # the statistics pi05_libero was trained with, from its checkpoint (as openpi's policy does)
    stats = normalize.load(download.maybe_download(P.CHECKPOINTS["ft"] + "/assets") / data.asset_id)
    return (
        _transforms.compose([
            *data.data_transforms.inputs,
            _transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]),
        _transforms.compose([
            *data.model_transforms.outputs,
            _transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]),
    )


def _decode(cell) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


class Inputs:
    """Model inputs of a list of states, kept as numpy arrays (row i = states[i])."""

    def __init__(self, states: list[dict], seed: int = SEED):
        import pyarrow.parquet as pq

        revision = json.loads(STATES_FILE.read_text())["revision"]
        transform, _ = transforms()
        by_episode: dict[int, list[int]] = {}
        for row, s in enumerate(states):
            by_episode.setdefault(s["episode_index"], []).append(row)
        items: list = [None] * len(states)
        for episode, rows in by_episode.items():
            table = pq.read_table(dataset_file(episode_file(episode), revision)).to_pandas()
            table = table.set_index("frame_index")
            for row in rows:
                frame = table.loc[states[row]["frame_index"]]
                items[row] = transform({
                    "observation/image": _decode(frame["image"]),
                    "observation/wrist_image": _decode(frame["wrist_image"]),
                    "observation/state": np.asarray(frame["state"], dtype=np.float32),
                    "prompt": states[row]["task"],
                })
        import jax

        self.states = states
        self.seed = seed
        self.data = jax.tree.map(lambda *xs: np.stack(xs), *items)
        self.action_shape = None  # (horizon, action_dim), set on first use

    def __len__(self):
        return len(self.states)

    def noise(self, rows) -> "np.ndarray":
        """Each state's fixed noise sample (horizon, action_dim): a function of the seed and the
        state's id only, so every script and both models get the same one."""
        import jax

        import model_diffing.archive.src.pair as P

        if self.action_shape is None:
            config = P.model_config()
            self.action_shape = (config.action_horizon, config.action_dim)
        key = jax.random.key(self.seed)
        return jax.numpy.stack([
            jax.random.normal(jax.random.fold_in(key, self.states[r]["id"]), self.action_shape) for r in rows
        ])

    def batches(self, batch_size: int):
        """Yields (rows, observation, noise). The last batch is padded to batch_size by repeating
        its last state, so every batch has one shape; use only the first len(rows) outputs."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        for start in range(0, len(self), batch_size):
            rows = np.arange(start, min(start + batch_size, len(self)))
            padded = np.pad(rows, (0, batch_size - len(rows)), mode="edge")
            batch = jax.tree.map(lambda x: jnp.asarray(x[padded]), self.data)
            yield rows, _model.Observation.from_dict(batch), self.noise(padded)
