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

## Second question: how well does each head predict the robot's action?

The sections above asked whether a head can tell labels apart, such as the task or the phase. The
paper ranks heads differently: by how well they predict what the robot does next. This section
applies that idea, following the paper's text.

The paper's released code differs from its text, so the next section follows the code. Neither
section is an exact comparison with the paper, because our robot, tasks and actions are different.

Heads are written as **L2H3**, meaning layer 2, head 3.

### How the score works

Each task is scored separately, in four steps:

1. Take one frame and one head's output on that frame.
2. In the task's other episodes, find the k frames where this head's output is most similar.
3. Average the actions of those k frames. That average is the prediction.
4. Compare the prediction with the frame's real action. The difference is the **action error**.

If a head's output really reflects what the robot is doing, similar outputs will go with similar
actions, and the error will be low. Lower is better, and 0 would be perfect.

- **k** is the number of similar frames. We tried 10, 20, 30 and 40 and used the paper's rule: the
  k that gives the 20 best heads the lowest average error.
- **Top 20** means the 20 heads with the lowest error, out of 144.

### What we predicted

The paper predicts joint velocities, but our data has joint position commands, so we tried two
targets:

- **Position:** the joint positions commanded at this frame.
- **Motion:** how far the command moves the joints over the next 0.1 s, which is the closest we
  have to the paper's velocities.

Each of the 25 action dimensions is scaled so that its 1st percentile becomes -1 and its 99th
becomes 1, so no single joint dominates. The error adds up the squared differences across the 25
dimensions.

**Example of what an error means.** The top 20 heads for position in hook ring have an error of
3.29. That is 3.29 / 25 = 0.13 per dimension, or a typical miss of 0.36 (the square root of 0.13).
The scaled range is 2 wide, so that is about 18% of a joint's usual range of movement.

### What we compared against

- **Control:** the same similar frames, but predicting the actions of a different episode at the
  same point in time. A head that only tracks how far along the episode is does just as well on
  the control as on the real score. So beating the control shows the head knows more than timing.
- **Median head:** the middle head of the 144, which stands for a typical head.
- **Pure noise:** the same tests on random numbers instead of head outputs. A result close to the
  noise value is not evidence of anything.

### The test

Before seeing any results, we fixed five tests that heads had to pass to count as specialized.
Tests 3 and 4 split a task's 20 episodes into two halves of 10 with no episode in common, 10
different ways.

Each cell below is a range over four cases: VLM or action expert, in hook ring or unhook ring.

| test | what the percentage means | normal run: position | normal run: motion | images-only: position | images-only: motion | pure noise |
|---|---|---|---|---|---|---|
| 1. Top 20 beat their control | how much lower their error is than their control | 60-75% | 17-24% | 59-69% | 18-23% | under 1% |
| 2. Top 20 beat the median head | how much lower their error is than the median head's | 15-30% | 7-11% | 16-34% | 4-7% | under 0.7% |
| 3. Top 20 repeat between halves | how many of the top 20 both halves agree on (14% by chance) | 34-56% | 33-46% | 47-74% | 17-27% | 8-18% |
| 4. Top 20 hold up on new episodes | heads picked on one half, scored on the other: how much lower their error is than the median head's | 6-17% | 3-6% | 13-23% | 0.3-1.3% | under 0.1% |

Test 2 flatters the top 20, because the same episodes are used to pick and to score them. Test 4 is
the fair version.

**Test 5 asks whether the top 20 are specific to a task or a phase.** If they are, the top 20 should
agree more with itself (between the two halves) than with a different task or phase. The results
below are gaps in that agreement, in percentage points. One point is 1% of 20 heads, or 0.2 of a
head.

- **Tasks: not met.** The gap is 0.9 points in the normal run and 0.5 in the images-only run, which
  is less than one head out of 20. Pure noise gives 0.5. Synthetic heads that we built to prefer
  one task give 20-49 points, or 4-10 heads.
- **Phases: met, but the effect is small.** The gap is 4.5 points in the normal run and 3.1 in the
  images-only run, or about one head out of 20. Pure noise gives 0.2.

**Very few heads belong to one phase.** We counted a head as belonging to one phase if it was in
the top 20 of both halves in at least 7 of 10 splits for that phase, and in at most 2 of 10 for
every other phase. Counting across the 8 cases of each run (2 parts x 2 targets x 2 tasks), that
gives 11 heads in the normal run and 9 in the images-only run. Pure noise gives 1. Of those 20
heads, 17 are for the retract phase.

### Conclusions from the action error

(Conclusions 9 and 10 use `action_specificity.py`, which we added after seeing the five tests.)

7. **Some heads are reliably better than others, but they are not specific to a task.** Heads picked
   on some episodes stay better on new ones (test 4). The exception is motion in the images-only
   run, where the gain is only 0.3-1.3%. The same heads mostly work for both tasks: on all episodes,
   hook ring and unhook ring share 9-17 of their top 20.
8. **The motion ranking depends on the prompt.** Without the task name and state in the prompt,
   agreement between halves for motion falls from 33-46% to 17-27%, close to the 14% expected by
   chance, and the gain on new episodes nearly disappears. The position ranking holds from the
   images alone (47-74% agreement).
9. **Part of a head's rank comes from its layer.** A head's layer alone accounts for 32-64% of the
   differences between heads, against about 10% for pure noise. 85-98% of the places in the top 20
   lists are in layers 0-9. Individual heads still matter: with each layer's average removed, the
   ranking still repeats between halves (rank correlation 0.37-0.67, where 1 is identical order).
   The exception is motion in the images-only run, where it falls to 0.15-0.29.
10. **Nearly every head beats the control.** 140-144 of the 144 heads have an error at least 1% below
    their control, so test 1 shows that heads in general carry action information, not that a
    special few do.
11. **A few heads do well everywhere.** The action expert's L6H3 and L7H3, and the VLM's L2H7, L3H0 and
    L3H3, are in the top 20 for both tasks, both targets and both runs. That makes them general
    heads, not specialized ones. We picked these names after seeing the results, so they still need
    checking on new episodes.
12. **How well heads predict depends on the phase, but which heads are best changes little.** In the
    images-only run, the median head's motion error beats its control by 0-1% during carry in unhook
    ring, but by 44-47% during align in hook ring. Yet a phase's top 20 agrees with other phases'
    almost as much as with itself (the 3-4.5 point gap in test 5).

### Compared with the paper

| measure | what it means | paper | ours |
|---|---|---|---|
| Top 20 shared between subsets of a task's episodes | agreement on the best heads when the episodes change | 54-82% (20 episodes each) | normal run: 33-56% (10 each) |
| Top 20 shared between tasks | agreement on the best heads when the task changes | 5-74% (several task pairs) | normal run: 39-48% (hook ring vs unhook ring) |
| Score spread at the first action token | how much heads differ from each other (explained below) | 0.47 | 0.05-0.07 motion, 0.18-0.32 position |
| Score spread at the last prompt token | the same, in the VLM | 0.05 | 0.04-0.07 motion, 0.21-0.31 position |

**Score spread** is the standard deviation of the 144 heads' errors divided by their average. At
0.47, heads' errors typically differ from the average by about half of it, so some heads are far
better than others. At 0.05, they differ by about 5%, so the heads are nearly the same.

Our agreement between subsets is lower, but each of our subsets has half as many episodes, so this
does not show that the paper's result fails here. Score spread is the clearest difference: in the
paper, heads differ far more at the first action token than at the last prompt token, and ours do
not show that. We predict different actions on a different robot, so the two aren't strictly
comparable. The paper also says that its best and worst heads are not clearly separated when
classifying tasks.

## Third question: which heads does the paper's released code select?

`select_heads.py` follows the paper's released code, and it picks the same heads as that code on
synthetic data. Three things change from the section above:

- **Input:** the first action token now receives the model's own predicted actions instead of noise.
- **Target:** 8 numbers, which are the active arm's 7 joint speeds plus its gripper. They are not
  scaled.
- **Frames:** we use all frames rather than the paper's keyframes, because the paper's rule would
  leave almost no frames in the contact phases.

For each k, the code joins the outputs of that k's 20 best heads into one long vector and scores it
the same way. The k with the lowest error wins, and its 20 heads are the **selected heads**. The
paper selects at the first action token, so these are the action expert's heads.

- Hook ring (k=40): L0H2 L0H6 L0H3 L6H6 L2H5 L2H7 L1H3 L6H4 L0H4 L3H1 L4H5 L3H3 L3H0 L8H4 L14H4
  L14H2 L0H1 L0H7 L12H5 L6H7
- Unhook ring (k=20): L0H2 L0H6 L0H3 L2H7 L0H7 L2H5 L6H6 L0H4 L1H3 L6H4 L0H1 L3H0 L3H1 L1H1 L3H3
  L4H5 L0H5 L1H5 L2H6 L1H0

Figures are in `pi05_base_denoised/figures/`: `selected_heads.png` shows these heads, and
`selection_heatmaps.png`, `selection_layers.png` and `selection_k.png` show the errors behind them.

### How well they predict

**What the error means here.** It is the squared difference averaged over the 8 numbers, with joint
speeds in rad/s and the gripper from 0 to 1. So it is on a different scale from the 25-dimension
errors in the section above, and the two can't be compared directly. An error of 0.113 means a
typical miss of about 0.34 (its square root) per number. The gripper counts about as much as a
fast-moving joint.

The easiest way to read these values is the last two columns. They show how much lower each error is
than the first row, which is the error from always guessing the average action.

| prediction | error: hook ring | error: unhook ring | better than guessing the average: hook ring | better than guessing the average: unhook ring |
|---|---|---|---|---|
| Guess the average action of the task's other episodes (no head) | 0.137 | 0.133 | - | - |
| Median head | 0.130 | 0.124 | 5% | 7% |
| The 20 selected heads, joined | 0.113 | 0.110 | 18% | 17% |
| Best head (L0H2) | 0.095 | 0.089 | 31% | 33% |
| Time only (no head; see below) | 0.100 | 0.086 | 27% | 35% |

**The two no-head rows are extra checks we added.** They are not part of the paper's method, and
they come from a one-off script that is not in the repository. The **time-only** row predicts a
frame's action by averaging other episodes' frames at the same point in the episode, for example
40% of the way through. It uses a plain average and knows each episode's total length, which a head
doesn't and which isn't known during a real run. So it has an advantage over the heads.

### How much heads differ

| score spread | paper | ours |
|---|---|---|
| First action token (action expert) | 0.47 | 0.058 hook ring, 0.078 unhook ring |
| Last prompt token (VLM) | 0.05 | 0.32 hook ring, 0.27 unhook ring |

### Conclusions from the selection

13. **Following the paper's code does not make heads differ the way they do in the paper.** At the
    first action token, the spread is 0.06-0.08 against the paper's 0.47, so our action-expert heads
    are nearly alike. The paper's ordering is also reversed, as ours spreads more at the last prompt
    token. Most of that comes from the VLM's layer 0 heads, which are clearly the worst (average
    error 0.21-0.24 against 0.11-0.12 for other layers). Without layer 0 the spread is 0.12-0.14.
    Our robot, target and frames still differ from the paper's.
14. **Joining the 20 selected heads does not beat the best single head, or the time-only check.**
    Joined, they reach 0.113 and 0.110, against 0.095 and 0.089 for the best head, and 0.100 and 0.086
    for time only. Only two single heads beat time only in hook ring (L0H2, L0H6), and none do in
    unhook ring. Time only has the advantage described above, so this suggests, but does not prove,
    that the heads add little beyond timing.
15. **A typical head adds little.** The median head's error is only 5-7% lower than always guessing the
    average action.
16. **The selection barely separates the tasks.** The two tasks share 15 of their 20 selected heads. About
    3 would be shared by chance, and about 8.5 if only each head's layer mattered. So the same
    individual heads win in both tasks, which makes them look general rather than task-specific.
17. **The selection favours early layers.** 12 of 20 selected heads in hook ring and 17 of 20 in unhook
    ring are in layers 0-3. A head's layer alone accounts for 46% and 79% of the differences between
    heads.
18. **Changing what the action tokens receive changes which heads win.** Only 4-5 of the 20 selected
    heads were also in the earlier motion top 20, and none in the earlier position top 20. The target
    changed too, so to separate the two causes we looked at the last prompt token, where the action
    tokens' input has no effect: there, 16-18 of 20 stay the same for motion. So most of the change
    comes from the input.
19. **k matters little.** In unhook ring, k=20 wins by under 0.5%. Changing k keeps 18-20 of the
    selected heads in the top 20 at the first action token (16-20 at the last prompt token).

We have not checked whether the same heads would be selected on new episodes.

## Limits

- This is the base model, which has never seen our robot. A fine-tuned model may organize this
  information differently.
- Head outputs are recorded at one token in each part of the model, and at one step of producing
  the actions.
- With 40 episodes, small differences between heads may not be reliable. The 10 splits reuse the same
  20 episodes per task, so they are not independent evidence.
- In the second section, all 25 action dimensions count equally after scaling. So position may partly
  reflect which arm is active (both arms are used in each task), and motion may include small
  movements of the idle arm. We did not test either.
- In the second section, the chosen k was 40, the largest we tried, in 6 of 8 cases, so larger values
  might score better.

## Next steps

1. Check how much of the position and motion rankings come from the active arm alone, by scoring only
   the active arm's joints.
2. Repeat on the fine-tuned unhook-ring model: does fine-tuning concentrate this information in fewer
   heads?
