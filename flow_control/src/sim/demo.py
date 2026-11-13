"""Demo videos: unsteered vs steered, side by side, on the same starting scene. LIBERO env.

    python src/sim/demo.py --pairs 3                     # picks scenes from results/data/stage1.jsonl
    python src/sim/demo.py --task 60 --states 4 17       # or name them

Needs the policy server (run through jobs/rollout_job.sh, see COMMANDS below). For each chosen
(task, starting scene) it plays the episode twice, exactly as Stage 1 does (same scene, same
noise seed at every replan; steered = the scripted controller's command written into the
noise), records the front camera, and writes one video:

    results/videos/task<T>_state<S>.mp4      left: unsteered, right: steered

Each half shows the condition, the step, the active command (for example "command: +x", the
axis in the robot's frame) and, at the end, the outcome of this run. When one episode ends
first, its last frame is held.

Picking scenes from Stage 1 (default): per task, up to --pairs scenes where the steered episode
succeeded and the unsteered one failed, then one where both failed, if any. The replay is a new
run: GPU arithmetic is not bit-exact between runs, so an outcome can differ from Stage 1. The
video shows the replay's own outcome, never Stage 1's.

    sbatch jobs/rollout_job.sh src/sim/demo.py --pairs 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rollout  # noqa: E402

ROOT = HERE.parents[1]
SCALE = 2  # 256 -> 512 pixels per half, readable on a slide
FPS = 20  # LIBERO's control rate: the video plays in real time
GREEN, RED, WHITE, BLACK = (40, 160, 70), (200, 50, 50), (255, 255, 255), (0, 0, 0)


def pick_scenes(path: pathlib.Path, pairs: int) -> list[tuple[int, int, str]]:
    """(task, state, why) from Stage 1's results."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    outcome = {(r["task"], r["state"], r["condition"]): r["success"] for r in rows}
    chosen = []
    for task in sorted({r["task"] for r in rows}):
        states = sorted({s for t, s, _ in outcome if t == task})
        both = [s for s in states if (task, s, "unsteered") in outcome and (task, s, "steered") in outcome]
        rescued = [s for s in both if outcome[task, s, "steered"] and not outcome[task, s, "unsteered"]]
        failed = [s for s in both if not outcome[task, s, "steered"] and not outcome[task, s, "unsteered"]]
        chosen += [(task, s, "steering rescued it in Stage 1") for s in rescued[:pairs]]
        chosen += [(task, s, "both failed in Stage 1") for s in failed[:1]]
    return chosen


def _font(size: int):
    """DejaVu Sans Bold from matplotlib's own font folder (installed in both environments), so
    the labels do not depend on the node's system fonts."""
    from PIL import ImageFont

    try:
        import matplotlib

        return ImageFont.truetype(str(pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), size)
    except (ImportError, OSError):
        return ImageFont.load_default()


def _label(frame: np.ndarray, lines: list[tuple[str, tuple]], banner: tuple[str, tuple] | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).resize((frame.shape[1] * SCALE, frame.shape[0] * SCALE), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    font = _font(22)
    y = 8
    for text, color in lines:
        draw.rectangle([6, y - 2, 14 + draw.textlength(text, font=font), y + 26], fill=BLACK)
        draw.text((10, y), text, fill=color, font=font)
        y += 32
    if banner:
        text, color = banner
        big = _font(40)
        width = draw.textlength(text, font=big)
        x, y = (image.width - width) / 2, image.height - 70
        draw.rectangle([x - 12, y - 6, x + width + 12, y + 50], fill=color)
        draw.text((x, y), text, fill=WHITE, font=big)
    return np.asarray(image)


def _command_text(command) -> str:
    if command is None:
        return "command: none"
    axis, sign = command
    return f"command: {'+' if sign > 0 else '-'}{'xyz'[axis]}"


def side_by_side(runs: dict) -> list[np.ndarray]:
    length = max(len(r["frames"]) for r in runs.values())
    video = []
    for i in range(length):
        halves = []
        for condition, color in (("unsteered", (200, 200, 200)), ("steered", (255, 120, 120))):
            r = runs[condition]
            j = min(i, len(r["frames"]) - 1)
            ended = i >= len(r["frames"]) - 1
            banner = (("SUCCESS", GREEN) if r["success"] else ("FAILURE", RED)) if ended else None
            lines = [(condition.upper(), color), (f"step {j}", WHITE)]
            if condition == "steered":
                lines.append((_command_text(r["frame_commands"][j]), WHITE))
            halves.append(_label(r["frames"][j], lines, banner))
        gap = np.full((halves[0].shape[0], 8, 3), 255, dtype=np.uint8)
        video.append(np.concatenate([halves[0], gap, halves[1]], axis=1))
    video += [video[-1]] * FPS  # hold the end for a second
    return video


def _play(job):
    return rollout.episode(job, record=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", default=str(ROOT / "results" / "data" / "stage1.jsonl"))
    parser.add_argument("--pairs", type=int, default=3, help="per task: scenes where steering rescued the episode")
    parser.add_argument("--task", type=int, help="with --states: play these scenes of this task instead")
    parser.add_argument("--states", type=int, nargs="+")
    parser.add_argument("--stream", type=int, default=0, help="noise stream (Stage 1 used 0)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "results" / "videos"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.states:
        scenes = [(args.task, s, "chosen by hand") for s in args.states]
    else:
        scenes = pick_scenes(pathlib.Path(args.stage1), args.pairs)
    print(f"{len(scenes)} scenes: {scenes}", flush=True)
    jobs = [(t, s, c, args.stream) for t, s, _ in scenes for c in ("unsteered", "steered")]

    import imageio

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=rollout._init, initargs=(args.host, args.port)) as pool:
        for r in pool.imap_unordered(_play, jobs):
            results.setdefault((r["task"], r["state"]), {})[r["condition"]] = r
            runs = results[r["task"], r["state"]]
            print(f"task {r['task']} state {r['state']} {r['condition']}: "
                  f"{'success' if r['success'] else 'failure'} in {r['steps']} steps", flush=True)
            if len(runs) == 2:
                path = out / f"task{r['task']}_state{r['state']}.mp4"
                imageio.mimwrite(path, side_by_side(runs), fps=FPS, quality=8, macro_block_size=8)
                print(f"wrote {path}", flush=True)
    summary = [{"task": t, "state": s, "why": why,
                **{c: results[t, s][c]["success"] for c in ("unsteered", "steered")}} for t, s, why in scenes]
    (out / "videos.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
