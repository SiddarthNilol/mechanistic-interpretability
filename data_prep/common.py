"""Shared helpers for preparing the head-specialization dataset.

Reads a LeRobot v3.0 dataset directly with pyarrow (no lerobot install needed) and
defines the task phases.

Each episode is split into six phases by five boundary frames. The two tasks are mirror
images, so they share every boundary and all phases but one:

                                   unhook ring                    hook ring
    [0,         t_align)    approach  arm moves toward the ring   (same, ring on the table)
    [t_align,   t_grasp)    align     gripper within ALIGN_RADIUS_M of where it grasps
    [t_grasp,   t_closed)   grasp     gripper closing on the ring
    [t_closed,  t_pipe)     unhook    ring held, on the pipe     carry  ring held, off the pipe
    [t_pipe,    t_release)  carry     ring held, off the pipe    hook   ring held, on the pipe
    [t_release, length)     retract   ring released, arm moves away

t_pipe is the frame where the ring comes off the pipe (unhook) or goes onto it (hook). It
can't be read from robot state, so it's annotated by hand; the other four are bootstrapped
from the active arm's gripper and end-effector position (bootstrap_boundaries).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BOUNDARIES = ["t_align", "t_grasp", "t_closed", "t_pipe", "t_release"]
# Per task string: its phases in order, and whether the ring is on the pipe before t_pipe.
TASKS = {
    "unhook ring": {"phases": ["approach", "align", "grasp", "unhook", "carry", "retract"], "starts_on_pipe": True},
    "hook ring": {"phases": ["approach", "align", "grasp", "carry", "hook", "retract"], "starts_on_pipe": False},
}

# Index of each gripper in observation.state. Gripper state is ~1 when open, ~0 when closed.
GRIPPER_DIM = {"left": 7, "right": 15}
GRIPPER_CLOSING = 0.9  # below this the gripper has started to close (fully open reads 0.954-1.0)
GRIPPER_CLOSED = 0.2  # below this the gripper is closed on the ring
GRIPPER_REOPENED = 0.5  # above this (after being closed) the ring is released
ALIGN_RADIUS_M = 0.05  # "align" starts when the end effector stays within 5 cm of its grasp position


# -- reading the dataset -------------------------------------------------------


def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def read_episodes(root: Path) -> pd.DataFrame:
    """One row per episode (frame range, data file, video files), indexed by episode_index."""
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    episodes = episodes[[c for c in episodes.columns if not c.startswith("stats/")]]
    return episodes.set_index("episode_index").sort_index()


def read_excluded(root: Path) -> list[int]:
    """Episodes listed in meta/excluded_episodes.json (none if the file doesn't exist)."""
    path = root / "meta" / "excluded_episodes.json"
    if not path.exists():
        return []
    return [int(e) for e in json.loads(path.read_text())["excluded_episodes"]]


def read_episode_frames(root: Path, episodes: pd.DataFrame, episode_index: int) -> pd.DataFrame:
    """All per-frame rows of one episode (state, action, poses, ...), sorted by frame_index."""
    row = episodes.loc[episode_index]
    path = (
        root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(path, filters=[("episode_index", "=", episode_index)])
    frames = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
    if len(frames) != int(row["length"]) or not np.array_equal(frames["frame_index"], np.arange(len(frames))):
        raise ValueError(f"episode {episode_index}: frames in {path} don't match its length {row['length']}")
    return frames


def iter_video_frames(root: Path, episodes: pd.DataFrame, episode_index: int, camera: str):
    """Yield (frame_index, PIL image) for every frame of one episode from video feature
    `camera`, in order. Each video file holds many episodes back to back."""
    import av  # only needed for video, so the other steps don't require it

    row = episodes.loc[episode_index]
    chunk, file = int(row[f"videos/{camera}/chunk_index"]), int(row[f"videos/{camera}/file_index"])
    path = root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    start = float(row[f"videos/{camera}/from_timestamp"])
    fps, length = float(read_info(root)["fps"]), int(row["length"])

    expected = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            index = round((float(frame.pts * stream.time_base) - start) * fps)
            if index < 0:
                continue  # frames of the previous episode before the seek target
            if index != expected:
                raise ValueError(f"{path}: expected frame {expected} of episode {episode_index}, got {index}")
            yield index, frame.to_image()
            expected += 1
            if expected == length:
                return
    raise ValueError(f"{path}: episode {episode_index} ends after {expected} of {length} frames")


def column(frames: pd.DataFrame, name: str) -> np.ndarray:
    """A per-frame vector column as a (num_frames, dim) float array."""
    return np.stack(frames[name].to_numpy()).astype(np.float64)


def gripper(frames: pd.DataFrame, arm: str) -> np.ndarray:
    """Gripper state of one arm per frame (1 = open, 0 = closed)."""
    return column(frames, "observation.state")[:, GRIPPER_DIM[arm]]


def ee_position(frames: pd.DataFrame, arm: str) -> np.ndarray:
    """Commanded end-effector position (x, y, z) of one arm per frame, in meters."""
    return column(frames, f"observation.commanded_pose.{arm}_arm.translation")


def active_arm(frames: pd.DataFrame) -> str:
    """The arm that grasps the ring: the one whose gripper closes furthest."""
    return min(GRIPPER_DIM, key=lambda arm: gripper(frames, arm).min())


def count_grasps(gripper_state: np.ndarray) -> int:
    """How many times the gripper goes from open to closed (>1 means a regrasp)."""
    closed = gripper_state < GRIPPER_REOPENED
    return int(closed[0]) + int(np.sum(~closed[:-1] & closed[1:]))


# -- phases --------------------------------------------------------------------


def bootstrap_boundaries(gripper_state: np.ndarray, ee_xyz: np.ndarray) -> dict[str, int]:
    """Automatic phase boundaries of one episode from its active arm (t_pipe is annotated
    by hand). If the gripper closes more than once, this uses the longest closure: the one
    that moves the ring. The others are failed grabs before it (they fall in approach/align)
    or a brief re-touch after the release (falls in retract)."""
    g = gripper_state
    held = np.r_[False, g < GRIPPER_REOPENED, False]
    starts = np.flatnonzero(~held[:-1] & held[1:])  # first frame of each closure
    ends = np.flatnonzero(held[:-1] & ~held[1:])  # first frame after it
    if len(starts) == 0:
        raise ValueError(f"the gripper never closes below {GRIPPER_REOPENED}")
    longest = int(np.argmax(ends - starts))
    start, t_release = int(starts[longest]), int(ends[longest])  # t_release = length if never reopened

    fully_closed = np.flatnonzero(g[start:t_release] < GRIPPER_CLOSED)
    t_closed = start + (int(fully_closed[0]) if len(fully_closed) else 0)
    still_open = np.flatnonzero(g[:t_closed] >= GRIPPER_CLOSING)
    t_grasp = int(still_open[-1]) + 1 if len(still_open) else 0

    # Last frame before the grasp that is still ALIGN_RADIUS_M or further from the grasp position.
    dist = np.linalg.norm(ee_xyz[:t_grasp] - ee_xyz[t_grasp], axis=1)
    far = np.flatnonzero(dist >= ALIGN_RADIUS_M)
    t_align = int(far[-1]) + 1 if len(far) else 0

    return {"t_align": t_align, "t_grasp": t_grasp, "t_closed": t_closed, "t_release": t_release}


def phase_ids(length: int, boundaries: dict) -> np.ndarray:
    """Index into the task's phases for every frame 0..length-1 of an episode. All five
    boundaries must be set and in order."""
    cuts = [int(boundaries[b]) for b in BOUNDARIES]
    return np.searchsorted(cuts, np.arange(length), side="right")


def check_boundaries(episode_index: int, length: int, boundaries: dict) -> None:
    """Raise if a boundary is missing or the boundaries aren't 0 <= ... <= length in order."""
    values = []
    for b in BOUNDARIES:
        v = boundaries[b]
        if pd.isna(v):
            raise ValueError(f"episode {episode_index}: {b} is not set in phase_boundaries.csv")
        values.append(int(v))
    if not all(a <= b for a, b in zip([0, *values], [*values, length])):
        pairs = ", ".join(f"{b}={v}" for b, v in zip(BOUNDARIES, values))
        raise ValueError(f"episode {episode_index}: boundaries out of order ({pairs}, length={length})")
