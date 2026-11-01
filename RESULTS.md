# Do individual attention heads of pi0.5 specialize in task and phase?

## The question

pi0.5 has 144 attention heads per model part (18 layers x 8 heads). Following
[arXiv:2511.22697](https://arxiv.org/abs/2511.22697), which picks a few heads in the
**base model** (before any fine-tuning) and then fine-tunes only those, we asked: in the base
pi0.5, do specific heads carry information about

- the **task**: hook ring vs unhook ring, and
- the **phase** within a task: approach, align, grasp, unhook / hook, carry, retract,

and about two facts of the scene: whether the ring is **grasped**, and whether it is **on the pipe**?

## What we did

**Data.** 40 episodes from our robot (20 hook ring, 20 unhook ring, picked at random), every
5th frame: 6,192 frames. Each frame has a task, a phase (boundaries checked and annotated by
hand) and the two scene facts.

**Model.** The released base pi0.5, never trained on our robot. It has two parts that share
attention: the **VLM** (reads the three camera images and a text prompt) and the
**action expert** (produces the actions). The prompt holds the task name and the robot's
state written as numbers.

**What we recorded.** For every frame, the output of every head (a 256-number vector):
for the 144 VLM heads at the last prompt token, and for the 144 action-expert heads at the
first action token (the position the paper uses).

**How we scored a head.** For each frame, we found the 20 frames whose head outputs are
most similar, taken only from other episode pairs, and let them vote on the label. The
**score** is the share of frames voted right, averaged over the classes so every phase
counts equally (0.5 = guessing for two classes, 0.17 for six phases). The **control** is
the same vote on labels borrowed from another episode at the same point in time; it shows
how much a head gets right just from how far into the episode a frame is. We report
**score above control**.

**Two runs.**
- **Normal run:** each frame with its own task name and state in the prompt.
- **Images-only run:** the same prompt text ("ring") and the same average state for every
  frame, so the cameras are the only thing that differs between frames.

## Results

| label | normal run: best score (control) | images-only run: best score (control) |
|---|---|---|
| task | 1.00 (0.46) | 0.82 (0.46) |
| phase, hook ring | 0.69 (0.46) | 0.69 (0.48) |
| phase, unhook ring | 0.67 (0.37) | 0.68 (0.38) |
| ring on pipe | 0.90 (0.57) | 0.90 (0.57) |
| ring grasped | 0.95 (0.78) | 0.96 (0.78) |

(Best head of either part; full maps in `pi05_base*/figures/heatmaps.png`.)

## Conclusions

1. **The phase and the ring's state come from the images.** Removing the task name and the
   state from the prompt does not lower the best scores, and the typical head scores higher.
2. **The perfect task score came from the prompt text.** From the images alone, heads still
   tell hook ring from unhook ring well (0.82 against 0.46 for the control), but not perfectly.
3. **"Ring grasped" is mostly timing.** Its control is already 0.78, because grasping always
   happens in the middle of an episode; heads add little beyond that.
4. **The information is spread over nearly all heads.** In the images-only run almost every
   head scores well above control, and the best head is only a little better than a typical
   one. The 20 best heads change noticeably between the two runs (5 to 13 of 20 in common).
   The base model does not show a small set of specialized heads.
5. **A few heads stand out consistently, in the early layers (1-4).** Action-expert head
   L2H3 is the best phase head for both tasks in both runs, and action-expert head L1H5 is the
   best ring-on-pipe head in both runs.
6. **The heads look at the cameras only a little at these positions.** About 9% of the VLM
   heads' attention and 19% of the action-expert heads' attention goes to the image tokens;
   most goes to the start of the prompt and to the state numbers. Because the VLM mixes
   image and text information from the first layer on, this does not mean the heads ignore
   the images (conclusion 1 shows they don't).

## Limits

- This is the base model, which has never seen our robot; a fine-tuned model may organize
  this information differently.
- Head outputs are measured at one position per part and at the first step of producing the
  actions.
- With 40 episodes, small differences between heads may not be reliable.

## Next steps

1. Check how stable the 20 best heads are (different subsets of episodes, 10 to 40 neighbours).
2. Repeat on the fine-tuned unhook-ring model: does fine-tuning concentrate this information
   in fewer heads?
3. Score heads on predicting the action, as the paper does, to compare with its maps.
