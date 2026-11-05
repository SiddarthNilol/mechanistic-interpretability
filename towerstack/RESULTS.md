# Head selection on tower stack

The paper's selection, following its released code (`analysis/select_heads.py`), on the base pi0.5.
The data is 20 random `stack_rings` episodes, every 5th frame: 5,245 frames. Figures are in
`towerstack_denoised/figures/`.

**These results are provisional.** `towerstack/prepared/episodes.csv` has the old gripper-based arm
labels, which are wrong for many of these episodes. Each episode was scored against the arm in that
file, so some were scored against the idle arm. The selection needs rerunning with corrected labels
before the numbers below are used.

## What the numbers mean

The action error is the squared difference between the predicted and real action, averaged over 8
numbers: one arm's 7 joint speeds (rad/s) and its gripper (0 to 1). Lower is better. The prediction
is built from the frames, in other episodes of the task, where the head's output is most similar.

## Selected heads (action expert, first action token, k=40)

L0H2 L0H6 L0H3 L6H6 L2H7 L6H4 L3H0 L2H5 L12H4 L4H5 L12H0 L14H7 L0H0 L3H1 L2H3 L12H5 L0H7 L14H6 L6H3 L10H0

## Results

| action expert | tower stack | hook ring | unhook ring |
|---|---|---|---|
| Best head | L0H2, 0.127 | L0H2, 0.095 | L0H2, 0.089 |
| Median head | 0.145 | 0.130 | 0.124 |
| The 20 selected heads, joined | 0.137 | 0.113 | 0.110 |
| Score spread (paper: 0.47) | 0.024 | 0.058 | 0.078 |
| Selected heads in layers 0-3 | 10 of 20 | 12 of 20 | 17 of 20 |

**Score spread** is the standard deviation of the 144 heads' errors divided by their average. At
0.024, heads' errors differ from the average by only about 2%. At the VLM's last prompt token it is
0.178, against the paper's 0.05, and it falls to 0.062 without layer 0.

| shared with tower stack | hook ring | unhook ring | by chance | if only the layer mattered |
|---|---|---|---|---|
| Selected action-expert heads (of 20) | 12 | 11 | about 3 | about 7 |
| Rank correlation of all 144 heads, action expert | 0.63 | 0.46 | | |
| Rank correlation of all 144 heads, VLM | 0.91 | 0.92 | | |

## Conclusions

1. **Heads barely differ on this task.** The action-expert spread of 0.024 is below both ring tasks and
   far below the paper's 0.47.
2. **The same heads win across tasks.** Eleven heads are selected on all three tasks: L0H2 L0H3 L0H6
   L0H7 L2H5 L2H7 L3H0 L3H1 L4H5 L6H4 L6H6. The best three are L0H2, L0H6 and L0H3, in that order, on all
   three tasks. This is another sign that the
   selection finds general heads rather than task-specific ones, and this time on a more different
   task.
3. **Joining the selected heads doesn't beat the best one.** The joined heads reach 0.137, against 0.127
   for L0H2 alone, as on the ring tasks.
4. **k has more effect here than on unhook ring,** but 40, the largest value tried, wins, so larger
   values might do better.

## What these results can't show

- **The higher errors on tower stack (0.145 vs 0.124-0.130) aren't explained.** They could come from the
  task, or from the wrong arm labels. No baseline has been computed for this task.
- **The VLM's correlation may be the most reliable comparison.** Wrong labels change only the
  target, so they should make the tasks agree less, not more. They could still change these values.
- **Selection isn't performance.** No fine-tuning has been run on tower stack.
