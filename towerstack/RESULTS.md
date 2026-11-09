# Head selection on tower stack

The paper's selection, following its released code (`analysis/select_heads.py`), on the base pi0.5.
The data is 20 random `stack_rings` episodes, every 5th frame: 5,245 frames. Hook and unhook ring
were rerun with the same scoring, so all three tasks can be compared. Figures are in
`towerstack_denoised/figures/`.

## What the numbers mean

The **action error** is the squared difference between the predicted and real action, averaged over
the robot's 25 action numbers. For the 23 joints (both arms, chest and neck) that is the speed over
the next 0.1 s in rad/s; for the 2 grippers, the command from 0 to 1. Lower is better. A head's
prediction is built from the frames, in other episodes of the task, where its output is most
similar.

The error depends on how much the robot moves in a task, so errors can be compared between heads on
one task, but not between tasks. Tower stack's errors are about twice the ring tasks'. That is most
likely because its actions vary more, but we didn't measure it, so it doesn't show that heads do
worse on tower stack.

## Selected heads (action expert, first action token, k=40)

L0H2 L0H3 L0H6 L2H7 L6H6 L2H5 L6H4 L3H0 L12H4 L0H7 L3H1 L0H1 L0H4 L4H5 L14H6 L0H0 L2H3 L12H5 L1H3 L14H4

## Results

| action expert | tower stack | hook ring | unhook ring |
|---|---|---|---|
| Best head | L0H2, 0.080 | L0H2, 0.039 | L0H2, 0.041 |
| Median head | 0.102 | 0.051 | 0.051 |
| Best head, lower than the median head by | 21% | 24% | 19% |
| The 20 selected heads joined, lower than the median head by | 7% | 12% | 8% |
| Score spread (paper: 0.47) | 0.043 | 0.053 | 0.062 |
| Selected heads in layers 0-3 | 13 of 20 | 11 of 20 | 17 of 20 |

**Score spread** is the standard deviation of the 144 heads' errors divided by their average. At
0.043, heads' errors differ from the average by only about 4%, against about half the average in the
paper. At the VLM's last prompt token the spread is 0.28, against the paper's 0.05; it falls to 0.105
without layer 0, whose heads are the worst (average error 0.19 against 0.10 for other layers).

| shared with tower stack | hook ring | unhook ring | by chance | if only the layer mattered |
|---|---|---|---|---|
| Selected action-expert heads (of 20) | 15 | 11 | about 3 | about 9 |
| Order of all 144 heads, action expert (rank correlation) | 0.85 | 0.87 | | |
| Order of all 144 heads, VLM (rank correlation) | 0.93 | 0.91 | | |

A rank correlation of 1 means the heads are in the same order on both tasks.

## Conclusions

1. **The same heads do well on all three tasks.** L0H2 is the best head on all three. Eleven heads are
   selected on all three: L0H1 L0H2 L0H3 L0H4 L0H6 L0H7 L1H3 L2H5 L2H7 L6H4 L6H6. Only five are
   selected for tower stack alone: L0H0 L2H3 L3H0 L12H4 L12H5. The order of all 144 heads is also
   close (0.85-0.87 in the action expert, 0.91-0.93 in the VLM). Tower stack is a different task from
   the ring tasks, so this is the clearest sign yet that the selection finds general heads rather
   than task-specific ones.
2. **Heads barely differ.** The action-expert spread of 0.043 is in line with the ring tasks and far
   below the paper's 0.47.
3. **Joining the selected heads doesn't beat the best one.** Joined, the 20 heads are 7% below the
   median head; L0H2 alone is 21% below it. The same holds on the ring tasks.
4. **Early layers dominate.** 13 of the 20 selected heads are in layers 0-3, and a head's layer alone
   accounts for 56% of the differences between heads.
5. **k matters little.** k=40 wins by 0.3%, and changing k keeps 18-20 of the 20 selected heads.

## Earlier version

A first run scored only one arm per episode, using arm labels that were wrong for many episodes.
Scoring the whole robot kept 16 of its 20 selected heads.

## What these results can't show

- **Selection isn't performance.** No fine-tuning has been run on tower stack. On unhook ring,
  fine-tuning the selected heads did no better than random heads.
- **One random set of 20 episodes.** We have not checked whether another set selects the same heads.
