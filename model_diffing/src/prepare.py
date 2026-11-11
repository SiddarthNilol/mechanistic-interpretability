"""Step 0: pick the states and download everything the jobs need. No GPU; needs internet.

    python src/prepare.py                 # writes results/states.json, downloads data + checkpoints
    python src/prepare.py --no-download   # only writes results/states.json

Sampling (seed 0): 2 episodes of each of the 40 tasks, so 20 episodes from each of the 4 suites.
In every episode, 12 frames: one at random in each twelfth of the episode. These 960 states are
used by the progress probe (F5). 3 of the 12 frames per episode are marked "patch": these 240
states are used by the patching and Jacobian pilots (F3, F4, F6).

Downloads, each to its cache (set HF_HOME and OPENPI_DATA_HOME to scratch first, see jobs/env.sh):
the 80 episode files (about 2.5 GB), the two checkpoints (about 12.5 GB each), the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from common import RESULTS, SEED, STATES_FILE
from states import DATASET, SUITES, dataset_file, episode_file


def sample_states(seed: int, episodes_per_task: int, frames: int, patch_frames: int) -> dict:
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET).sha  # pin the dataset version
    tasks = [json.loads(line) for line in open(dataset_file("meta/tasks.jsonl", revision))]
    episodes = [json.loads(line) for line in open(dataset_file("meta/episodes.jsonl", revision))]
    task_index = {t["task"]: t["task_index"] for t in tasks}
    if len(tasks) != 10 * len(SUITES):
        raise RuntimeError(f"expected 40 tasks (4 suites of 10), found {len(tasks)}")

    rng = np.random.default_rng(seed)
    states = []
    for task in sorted(tasks, key=lambda t: t["task_index"]):
        candidates = [e for e in episodes if e["tasks"] == [task["task"]]]
        for i in sorted(rng.choice(len(candidates), episodes_per_task, replace=False)):
            episode = candidates[i]
            edges = np.linspace(0, episode["length"], frames + 1).astype(int)
            chosen = [int(rng.integers(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]
            patch = set(rng.choice(frames, patch_frames, replace=False).tolist())
            for j, frame in enumerate(chosen):
                states.append({
                    "id": len(states),
                    "episode_index": episode["episode_index"],
                    "frame_index": frame,
                    "episode_length": episode["length"],
                    "task_index": task_index[task["task"]],
                    "task": task["task"],
                    "suite": SUITES[task_index[task["task"]] // 10],
                    "patch": j in patch,
                })
    return {"dataset": DATASET, "revision": revision, "seed": seed, "states": states}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--frames", type=int, default=12, help="frames per episode (probe states)")
    parser.add_argument("--patch-frames", type=int, default=3, help="of those, frames marked for patching")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    print("prepare.py started", flush=True)
    if STATES_FILE.exists():
        print(f"{STATES_FILE} exists, keeping it (delete it to resample)")
        sampled = json.loads(STATES_FILE.read_text())
    else:
        sampled = sample_states(args.seed, args.episodes_per_task, args.frames, args.patch_frames)
        RESULTS.mkdir(parents=True, exist_ok=True)
        STATES_FILE.write_text(json.dumps(sampled, indent=1))
    states = sampled["states"]
    for suite in SUITES:
        mine = [s for s in states if s["suite"] == suite]
        print(f"{suite}: {len({s['episode_index'] for s in mine})} episodes, {len(mine)} states, "
              f"{sum(s['patch'] for s in mine)} for patching")
    print(f"total: {len(states)} states, {sum(s['patch'] for s in states)} for patching -> {STATES_FILE}")
    if args.no_download:
        return

    episodes = sorted({s["episode_index"] for s in states})
    print(f"downloading {len(episodes)} episode files (about 30 MB each) to {os.environ.get('HF_HOME', '~/.cache/huggingface')}", flush=True)
    for i, episode in enumerate(episodes):
        dataset_file(episode_file(episode), sampled["revision"])
        print(f"\repisode files: {i + 1}/{len(episodes)}", end="", flush=True)
    print()

    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import download

    import pair as P

    logging.basicConfig(level=logging.INFO, force=True)  # openpi reports its downloads through logging
    cache = os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")
    print("downloading the tokenizer", flush=True)
    PaligemmaTokenizer(P.model_config().max_token_len)
    for name, uri in P.CHECKPOINTS.items():
        print(f"downloading checkpoint {name} (12.4 GB, can take 10 minutes or more) to {cache}\n"
              f"  to watch it from another shell: du -sh {cache}/openpi-assets/checkpoints/*", flush=True)
        print(f"checkpoint {name}: {download.maybe_download(uri + '/params')}", flush=True)
    print(f"norm stats: {download.maybe_download(P.CHECKPOINTS['ft'] + '/assets')}", flush=True)
    print("done: everything the jobs need is downloaded", flush=True)


if __name__ == "__main__":
    main()
