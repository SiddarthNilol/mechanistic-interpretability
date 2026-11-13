"""Module 3: Flow Control's scripted controller (paper Appendix G.1). numpy only.

The paper replaces a human at the keyboard with a script that, at every replan, "holds the key"
for the dominant axis of the vector from the end effector to a target:

    before the grasp    target = the task object's position
    after the grasp     target = a point 12 cm above the goal region

    command = (axis, sign) with axis = argmax |target - end effector| over x, y, z, and sign the
              sign of that component. It is sent to the policy server, which writes it into the
              starting noise (src/policy/injection.py).

It reads privileged simulator state (libero_env.state), which is allowed only for this oracle,
for labels and for scoring. In Stage 5 the same rule is fed by the learned tracker instead.

Assumes the action axes are the world axes, which src/sim/probe.py checks ("action frame").
"""

from __future__ import annotations

import numpy as np

ABOVE_GOAL = np.array([0.0, 0.0, 0.12])  # meters


def target_point(state: dict) -> np.ndarray:
    return state["goal"] + ABOVE_GOAL if state["held"] else state["target"]


def command(state: dict) -> tuple[int, int]:
    """(axis, sign) of the dominant component of the vector from end effector to target."""
    vector = target_point(state) - state["eef"]
    axis = int(np.argmax(np.abs(vector)))
    return axis, 1 if vector[axis] >= 0 else -1


if __name__ == "__main__":  # self-test: python src/sim/controller.py
    s = {"eef": np.zeros(3), "target": np.array([0.02, -0.30, 0.01]), "goal": np.array([0.4, 0.0, 0.0]), "held": False}
    assert command(s) == (1, -1), "before the grasp: toward the object, dominant -y"
    s["held"] = True
    assert command(s) == (0, 1), "after the grasp: toward 12 cm above the goal, dominant +x"
    s["eef"] = np.array([0.4, 0.0, -0.2])
    assert command(s) == (2, 1), "above the goal: up"
    print("controller.py: ok")
