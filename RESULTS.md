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

## The paper's score: predicting the action

**Note.** This section follows the paper's text. Its released code differs: it feeds the action
tokens the model's own predicted actions, and differs in frames (it uses keyframes), actions and
how k is chosen. So
these numbers are not a like-for-like comparison with the paper. See `analysis/select_heads.py`.

**How we scored a head.** Following the paper's text, per task: for each frame, take the k frames with the
most similar head output from the task's other episodes, and predict the action as the mean of
their actions. The **action error** of a head is the squared error of that prediction, so lower
is better. k is 10, 20, 30 or 40, chosen by the paper's rule. The **top 20** are the 20 heads with
the lowest action error.

**What we predicted.** The paper uses joint velocities; our data has joint position commands. So
we predicted two things: **position** (the command itself) and **motion** (how far the command
moves the joints in the next 0.1 s, the closest we have to velocities).

**The test.** Before seeing results, we fixed five conditions that all had to hold. Tests 3 and
4 split a task's episodes into two halves of 10 with no episode in common, 10 different ways.

| test | normal run: position | normal run: motion | images-only: position | images-only: motion | pure noise |
|---|---|---|---|---|---|
| 1. Top 20 below their control | 60-75% | 17-24% | 59-69% | 18-23% | under 1% |
| 2. Top 20 below the median head | 15-30% | 7-11% | 16-34% | 4-7% | under 0.7% |
| 3. Top 20 shared between the two halves | 34-56% | 33-46% | 47-74% | 17-27% | 8-18% |
| 4. Heads picked on one half, below the median head on the other | 6-17% | 3-6% | 13-23% | 0.3-1.3% | under 0.1% |

**Test 5, tasks: not met.** A task's top 20 should repeat better between its own two halves
than against the other task. On average it does by only 0.9 points in the normal run and 0.5 in
the images-only run, against 0.5 for pure noise and 20-49 for synthetic heads we built to prefer
one task.

**Test 5, phases: met, but the effect is small.** A phase's top 20 repeats better than it matches
other phases by 4.5 points in the normal run and 3.1 in the images-only run, against 0.2 for pure
noise. Few heads belong to a single phase: 11 in the normal run and 9 in the images-only run,
over 8 comparisons each, against 1 for pure noise. 17 of those 20 are for retract.

### Conclusions from the action error

(Conclusions 9 and 10 use `action_specificity.py`, added after seeing the five tests.)

7. **Some heads are reliably better than others, but they are not specialized by task.** Heads
   picked on some episodes stay better on new ones, except for motion in the images-only run.
   The same heads mostly work for both tasks: 9-17 of the 20 best are shared between hook
   ring and unhook ring.
8. **Motion ranking depends on the prompt.** Tests 3 and 4 for motion nearly vanish when the
   task text and state are removed. For position they hold from the images alone.
9. **Part of a head's rank comes from its layer.** Layers explain 32-64% of the differences
   between heads (about 10% for pure noise), and most top heads are in layers 0-9. Rank still
   repeats within a layer, so individual heads matter too, except for images-only motion, where
   this is weak.
10. **Nearly every head beats the control** (140-144 of 144), so test 1 shows that heads carry
    action information, not that a few do.
11. **A few heads stand out everywhere.** The action expert's L6H3 and L7H3, and the VLM's L2H7,
    L3H0 and L3H3, are in the top 20 for both tasks, both predictions and both runs. These are
    general heads, not specialized ones. These lists were picked after seeing the results, so
    they still need checking on new episodes.
12. **How well heads predict depends on the phase.** Median motion error below control runs from
    0-1% during carry in unhook ring to 45% during align in hook ring (images-only). Which heads
    are best changes far less.

### Compared with the paper

| measure | paper | ours |
|---|---|---|
| Top 20 shared between subsets of a task's demonstrations | 54-82% (20 each) | normal run: 33-56% (10 each) |
| Top 20 shared between tasks | 5-74% (several task pairs) | normal run: 39-48% (hook ring vs unhook ring) |
| Score spread (std / mean), first action token | 0.47 | 0.05-0.07 motion, 0.18-0.32 position |
| Score spread (std / mean), last prompt token | 0.05 | 0.04-0.07 motion, 0.21-0.31 position |

Our overlap is lower, but each of our subsets has half the episodes, so this does not show
that the paper's result fails here. The score spread is the clearest difference: the paper's
first action token spreads far more than its last prompt token, and ours does not. The two are
not strictly comparable, since we predict different actions on a different robot. The paper also
says that its best and worst heads are not clearly separated in task classification.

## The paper's selection, following its released code

**What changed from the section above.** `select_heads.py` follows the paper's code, and we checked
that it picks the same heads as that code on synthetic data. The first action token now receives
the model's own predicted actions instead of noise. The target is the active arm's 7 joint
velocities plus its gripper. We use all frames, not the paper's keyframes.

**Selected heads (action stream).**

- Hook ring (k=40): L0H2 L0H6 L0H3 L6H6 L2H5 L2H7 L1H3 L6H4 L0H4 L3H1 L4H5 L3H3 L3H0 L8H4 L14H4
  L14H2 L0H1 L0H7 L12H5 L6H7
- Unhook ring (k=20): L0H2 L0H6 L0H3 L2H7 L0H7 L2H5 L6H6 L0H4 L1H3 L6H4 L0H1 L3H0 L3H1 L1H1 L3H3
  L4H5 L0H5 L1H5 L2H6 L1H0

**Action error (lower is better).**

| prediction | hook ring | unhook ring |
|---|---|---|
| Average action of the task's other episodes (no head) | 0.137 | 0.133 |
| Median head | 0.130 | 0.124 |
| The 20 selected heads, joined | 0.113 | 0.110 |
| Time only: other episodes' frames at the same point in the episode (no head) | 0.100 | 0.086 |
| Best head (L0H2) | 0.095 | 0.089 |

The two no-head rows come from a one-off check that is not in the repository. The time-only row
uses a plain average, and it knows the episode's length, which a head does not.

| measure | paper | ours |
|---|---|---|
| Score spread (std / mean), first action token | 0.47 | 0.058 hook ring, 0.078 unhook ring |
| Score spread (std / mean), last prompt token | 0.05 | 0.32 hook ring, 0.27 unhook ring |

### Conclusions from the selection

13. **Feeding predicted actions does not give the paper's spread.** The first action token stays
    near 0.06-0.08 against the paper's 0.47, and the paper's ordering is reversed: our last prompt
    token spreads more. Much of that spread comes from its layer 0 heads, which are the worst;
    without layer 0 it is 0.12-0.14. Our robot, target and frames still differ from the paper's.
14. **The selected heads predict actions worse than timing alone.** Joined, they score 0.113 and
    0.110 against 0.100 and 0.086 for time only. Only two single heads beat time only in hook ring
    (L0H2, L0H6), and none do in unhook ring. Joining the 20 is also worse than the best head alone.
15. **Most heads add little.** The median head is only 5-7% better than always predicting the
    average action.
16. **The selection barely separates the tasks.** 15 of 20 heads are shared, against about 3 by
    chance and about 8.5 if only each head's layer mattered. So individual heads repeat across
    tasks, but they look general rather than task-specific.
17. **The selection favours early layers.** 12 (hook ring) and 17 (unhook ring) of the 20 are in
    layers 0-3, and layers explain 46% and 79% of the differences between heads.
18. **The input changes which action-stream heads win.** Only 4-5 of the 20 match the earlier
    motion top 20 (none match position). The target changed too, but the last prompt token, which
    the input cannot affect, keeps 16-18 of 20 for motion. So most of the change comes from the
    input.
19. **k matters little.** The unhook-ring choice of k=20 wins by under 0.5%, and the top 20 stays
    18-20 of 20 the same across k in the action stream (16-20 in the VLM).

We have not tested these selected heads on new episodes, so we don't yet know that they repeat.

## Limits

- This is the base model, which has never seen our robot; a fine-tuned model may organize
  this information differently.
- Head outputs are measured at one position per part and at the first step of producing the
  actions.
- With 40 episodes, small differences between heads may not be reliable. The 10 splits reuse
  the same 20 episodes per task, so they are not independent.
- The action has 25 dimensions, all weighted equally after scaling, so position may partly
  reflect which arm is active (both occur in each task), and motion may include small movements
  of the idle arm. We did not test either.
- The chosen k was 40, the largest allowed, in 6 of 8 cases, so larger values might score better.

## Next steps

1. Check how much of the position and motion rankings come from the active arm alone, by scoring
   only the active arm's joints.
2. Repeat on the fine-tuned unhook-ring model: does fine-tuning concentrate this information
   in fewer heads?
