"""Where each head looks: its attention split by kind of input.

    python analysis/attention_groups.py --heads $SCRATCH/mechanistic-interpretability/pi05_base

The prompt holds the task and the state as text, so a head can score on task or phase by
reading it rather than by looking at the cameras. This sums every head's attention weights
(attn_*.npy from extract_heads.py) over groups of keys and averages over frames:

    head_cam, left_wrist, right_wrist   the 256 image tokens of each camera
    task                                the task words ("hook ring")
    state                               the state numbers
    template                            the rest of the prompt: <bos>, "Task:", ", State:", ";\\nAction:"
    actions                             the 50 action tokens (action stream only)

Writes to <heads>:

    attention_groups.csv            stream, layer, head, group, share
    figures/attention_groups.png    per stream and group, an 18 x 8 map of the share
    figures/top_heads_attention.csv per label's top heads (if plot_heads.py ran), their
                                    mean share per group, next to the mean over all heads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sentencepiece  # noqa: E402
from openpi.shared import download  # noqa: E402

from plot_heads import draw_map, head_map  # noqa: E402

GROUPS = ["head_cam", "left_wrist", "right_wrist", "task", "state", "template", "actions"]
TOKENIZER = "gs://big_vision/paligemma_tokenizer.model"  # the one openpi's PaligemmaTokenizer uses


def prompt_groups(ids: np.ndarray, sp: sentencepiece.SentencePieceProcessor) -> list[str]:
    """The group (task, state or template) of every real token of one prompt
    "Task: <task>, State: <numbers>;\\nAction: ". A token belongs to the part of the text
    where its first non-space character is."""
    pieces = [sp.id_to_piece(int(i)).replace("▁", " ") for i in ids]
    text = "".join(pieces[1:])  # pieces[0] is <bos>
    task = (text.index("Task: ") + len("Task: "), text.index(", State: "))
    state = (task[1] + len(", State: "), text.index(";", task[1]))
    groups, position = ["template"], 0
    for piece in pieces[1:]:
        first = position + len(piece) - len(piece.lstrip())
        groups.append("task" if task[0] <= first < task[1] else "state" if state[0] <= first < state[1] else "template")
        position += len(piece)
    return groups


def key_groups(tokens: np.ndarray, n_keys: int, sp) -> np.ndarray:
    """(frames, n_keys) group number (index in GROUPS) of every key, -1 for prompt padding.
    Keys: 3 x 256 image tokens, the prompt tokens, then the action tokens."""
    n_frames, n_prompt = tokens.shape
    groups = np.full((n_frames, n_keys), -1)
    for g, start in enumerate([0, 256, 512]):
        groups[:, start : start + 256] = g
    groups[:, 768 + n_prompt :] = GROUPS.index("actions")
    for f, ids in enumerate(tokens):
        real = ids[ids != 0]
        groups[f, 768 : 768 + len(real)] = [GROUPS.index(g) for g in prompt_groups(real, sp)]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=Path, required=True, help="output directory of extract_heads.py")
    args = parser.parse_args()

    with download.maybe_download(TOKENIZER, gs={"token": "anon"}).open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    tokens = np.load(args.heads / "tokens.npy")
    info = json.loads((args.heads / "info.json").read_text())

    rows = []
    for stream in ["vlm", "action"]:
        attn = np.load(args.heads / f"attn_{stream}.npy", mmap_mode="r")  # (frames, layers, heads, keys)
        groups = key_groups(tokens, attn.shape[-1], sp)
        for layer in range(attn.shape[1]):
            a = np.asarray(attn[:, layer], dtype=np.float32)  # (frames, heads, keys)
            for g, name in enumerate(GROUPS):
                if not (groups == g).any():
                    continue  # no action tokens in the vlm stream
                share = (a * (groups == g)[:, None, :]).sum(-1).mean(0)  # (heads,)
                rows += [(stream, layer, h, name, s) for h, s in enumerate(share)]
        print(f"{stream}: done", flush=True)
    table = pd.DataFrame(rows, columns=["stream", "layer", "head", "group", "share"])
    table.round(4).to_csv(args.heads / "attention_groups.csv", index=False)

    figures = args.heads / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(2.2 * len(GROUPS), 8), squeeze=False, layout="constrained")
    for row, stream in enumerate(["vlm", "action"]):
        for col, name in enumerate(GROUPS):
            t = table[(table["stream"] == stream) & (table["group"] == name)]
            if t.empty:
                axes[row, col].axis("off")
                continue
            image = draw_map(axes[row, col], head_map(t, "share"), 0, 1, title=f"{name} · {stream}",
                             axis_labels=col == 0)
    fig.colorbar(image, ax=axes, shrink=0.3, label="share of the head's attention")
    fig.suptitle(f"Where each head looks (prompt: {info.get('prompt', 'each frame’s task')}, "
                 f"state: {info.get('state', 'each frame’s')})", fontsize=10)
    fig.savefig(figures / "attention_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    shares = table.pivot_table(index=["stream", "layer", "head"], columns="group", values="share")
    print("\nmean share of attention (all heads):")
    print(shares.groupby("stream").mean()[[g for g in GROUPS if g in shares]].round(3).to_string())
    if (figures / "top_heads.csv").exists():
        top = pd.read_csv(figures / "top_heads.csv")
        top = top.join(shares, on=["stream", "layer", "head"])
        summary = top.groupby(["stream", "label"])[[g for g in GROUPS if g in shares]].mean()
        for stream in ["vlm", "action"]:
            summary.loc[(stream, "(all heads)"), :] = shares.loc[stream].mean()
        summary = summary.sort_index().round(3)
        summary.to_csv(figures / "top_heads_attention.csv")
        print("\nmean share of attention of each label's top heads:")
        print(summary.to_string())
    print(f"\nwrote {args.heads / 'attention_groups.csv'} and {figures / 'attention_groups.png'}")


if __name__ == "__main__":
    main()
