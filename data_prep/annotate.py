"""Step 3: a browser GUI to check the phase boundaries and annotate t_pipe.

    python data_prep/annotate.py --dataset /path/to/dataset --out prepared/hook_unhook

then open http://localhost:8765 (on HPC, through an SSH tunnel; see the README).

For each chosen episode it shows the head camera and the active arm's wrist camera, a
timeline with the phases, the gripper trace and the five boundaries. Set a boundary to the
current frame with keys 1-5 (or drag its marker), then "Save & next".

Every save writes <out>/phase_boundaries.csv right away (the episode's `reviewed` becomes 1),
so you can stop and restart at any time. Once all episodes are reviewed, it also writes
<out>/frames.csv (build_labels.py), and does so again after any later save.

Needs episodes.csv and phase_boundaries.csv from steps 1 and 2.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from build_labels import build_labels, summary
from common import BOUNDARIES, TASKS, check_boundaries, gripper, iter_video_frames, read_episode_frames, read_episodes

PAGE = Path(__file__).with_name("annotate.html")
IMAGE_WIDTH = 480  # frames are downscaled to this width before being sent


class FrameCache:
    """JPEG frames of the most recently opened episodes, for both cameras.

    A whole episode is decoded at once because the videos have a keyframe only every 250
    frames, which makes seeking to single frames slow (~2.5 s per episode this way)."""

    def __init__(self, dataset: Path, episodes: pd.DataFrame, arms: dict[int, str], keep: int = 4):
        self.dataset, self.episodes, self.arms, self.keep = dataset, episodes, arms, keep
        self.lock = threading.Lock()
        self.decoded: collections.OrderedDict[int, dict[str, list[bytes]]] = collections.OrderedDict()
        self.loading: dict[int, threading.Event] = {}

    def frame(self, ep: int, camera: str, index: int) -> bytes:
        return self.episode(ep)[camera][index]

    def episode(self, ep: int) -> dict[str, list[bytes]]:
        """Decode an episode unless it's cached or another thread is already decoding it."""
        with self.lock:
            if ep in self.decoded:
                self.decoded.move_to_end(ep)
                return self.decoded[ep]
            done = self.loading.get(ep)
            mine = done is None
            if mine:
                done = self.loading[ep] = threading.Event()
        if not mine:
            done.wait()
            with self.lock:
                return self.decoded[ep]
        try:
            cameras = {"head": "observation.images.head", "wrist": f"observation.images.{self.arms[ep]}_wrist"}
            frames = {
                name: [jpeg(image) for _, image in iter_video_frames(self.dataset, self.episodes, ep, feature)]
                for name, feature in cameras.items()
            }
            with self.lock:
                self.decoded[ep] = frames
                while len(self.decoded) > self.keep:
                    self.decoded.popitem(last=False)
            return frames
        finally:
            with self.lock:
                del self.loading[ep]
            done.set()

    def prefetch(self, ep: int) -> None:
        threading.Thread(target=self.episode, args=(ep,), daemon=True).start()


def jpeg(image) -> bytes:
    image.thumbnail((IMAGE_WIDTH, IMAGE_WIDTH))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


class Annotation:
    """phase_boundaries.csv in memory; every save writes it back to disk."""

    def __init__(self, dataset: Path, out: Path, stride: int, episodes: pd.DataFrame):
        self.dataset, self.out, self.stride = dataset, out, stride
        self.path = out / "phase_boundaries.csv"
        self.table = pd.read_csv(self.path)
        if "reviewed" not in self.table:
            self.table["reviewed"] = 0
        self.table[BOUNDARIES] = self.table[BOUNDARIES].astype("Int64")  # t_pipe may be empty
        self.table["note"] = self.table["note"].fillna("")
        self.auto = pd.read_csv(out / "phase_boundaries_auto.csv").set_index("episode_index")
        self.chosen = pd.read_csv(out / "episodes.csv").set_index("episode_index")
        self.gripper = {
            ep: gripper(read_episode_frames(dataset, episodes, ep), arm).round(3).tolist()
            for ep, arm in self.chosen["active_arm"].items()
        }
        self.lock = threading.Lock()

    def episodes_json(self) -> list[dict]:
        """Everything the page needs about each episode, gripper trace included."""
        result = []
        with self.lock:
            rows = self.table.to_dict("records")
        for row in rows:
            ep = int(row["episode_index"])
            task = self.chosen.loc[ep, "task"]
            result.append(
                {
                    "episode_index": ep,
                    "task": task,
                    "phases": TASKS[task]["phases"],
                    "starts_on_pipe": TASKS[task]["starts_on_pipe"],
                    "length": int(row["length"]),
                    "info": {c: str(v) for c, v in self.chosen.loc[ep].items() if c != "length" and not pd.isna(v)},
                    "note": row["note"],
                    "reviewed": bool(row["reviewed"]),
                    "boundaries": {b: None if pd.isna(row[b]) else int(row[b]) for b in BOUNDARIES},
                    "auto": {b: None if pd.isna(self.auto.loc[ep, b]) else int(self.auto.loc[ep, b]) for b in BOUNDARIES},
                    "gripper": self.gripper[ep],
                }
            )
        return result

    def save(self, ep: int, boundaries: dict) -> dict:
        """Store one episode's boundaries; once every episode is reviewed, write frames.csv."""
        with self.lock:
            row = self.table.index[self.table["episode_index"] == ep][0]
            check_boundaries(ep, int(self.table.at[row, "length"]), boundaries)
            for b in BOUNDARIES:
                self.table.at[row, b] = int(boundaries[b])
            self.table.at[row, "reviewed"] = 1
            tmp = self.path.with_suffix(".tmp")
            self.table.to_csv(tmp, index=False)
            os.replace(tmp, self.path)  # never leaves a half-written csv

            reviewed = int(self.table["reviewed"].sum())
            result = {"reviewed": reviewed, "total": len(self.table), "summary": None}
            if reviewed == len(self.table):
                frames_table = build_labels(self.dataset, self.out, self.stride)
                result["summary"] = f"wrote {self.out / 'frames.csv'}\n{summary(frames_table)}"
                print(result["summary"], flush=True)
            return result


def make_handler(annotation: Annotation, cache: FrameCache):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = self.path.strip("/").split("/")
            try:
                if self.path == "/":
                    self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/api/episodes":
                    self.reply(json.dumps(annotation.episodes_json()).encode(), "application/json")
                elif parts[:2] == ["api", "frame"] and len(parts) == 5:  # /api/frame/<ep>/<camera>/<index>
                    self.reply(cache.frame(int(parts[2]), parts[3], int(parts[4])), "image/jpeg", cacheable=True)
                elif parts[:2] == ["api", "prefetch"] and len(parts) == 3:  # /api/prefetch/<ep>
                    cache.prefetch(int(parts[2]))
                    self.reply(b"{}", "application/json")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, IndexError, ValueError) as e:
                self.send_error(HTTPStatus.BAD_REQUEST, str(e))

        def do_POST(self):
            if self.path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                result = annotation.save(int(body["episode_index"]), body["boundaries"])
            except (IndexError, KeyError, ValueError) as e:  # bad request, or boundaries that fail the check
                self.reply(json.dumps({"error": str(e)}).encode(), "application/json", HTTPStatus.BAD_REQUEST)
                return
            self.reply(json.dumps(result).encode(), "application/json")

        def reply(self, body: bytes, content_type: str, status=HTTPStatus.OK, cacheable=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cacheable else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # one line per frame request would flood the terminal

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 dataset root")
    parser.add_argument("--out", type=Path, required=True, help="directory holding phase_boundaries.csv")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stride", type=int, default=5, help="frame stride for frames.csv (see build_labels.py)")
    args = parser.parse_args()

    episodes = read_episodes(args.dataset)
    annotation = Annotation(args.dataset, args.out, args.stride, episodes)
    cache = FrameCache(args.dataset, episodes, annotation.chosen["active_arm"].to_dict())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(annotation, cache))
    print(f"annotating {len(annotation.table)} episodes; open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
