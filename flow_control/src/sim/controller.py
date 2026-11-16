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
GATE = 0.08  # Stage 3: horizontal distance from the target beyond which the extensions act (m)


def target_point(state: dict) -> np.ndarray:
    return state["goal"] + ABOVE_GOAL if state["held"] else state["target"]


def command(state: dict) -> tuple[int, int]:
    """(axis, sign) of the dominant component of the vector from end effector to target."""
    vector = target_point(state) - state["eef"]
    axis = int(np.argmax(np.abs(vector)))
    return axis, 1 if vector[axis] >= 0 else -1


def far_from_target(state: dict) -> bool:
    """Stage 3 gate for the gripper veto and adaptive strength: the target is not held and the
    end effector is more than GATE from it horizontally. Offline (Stage 3 diagnostic), the replans
    before a correct grasp were within 7.4 cm of the target and those before a wrong grasp at
    9.6 cm or more, so inside the gate the policy is left to grasp as plain Flow Control."""
    return (not state["held"]) and float(np.linalg.norm((state["target"] - state["eef"])[:2])) > GATE


if __name__ == "__main__":  # self-test: python src/sim/controller.py
    s = {"eef": np.zeros(3), "target": np.array([0.02, -0.30, 0.01]), "goal": np.array([0.4, 0.0, 0.0]), "held": False}
    assert command(s) == (1, -1), "before the grasp: toward the object, dominant -y"
    s["held"] = True
    assert command(s) == (0, 1), "after the grasp: toward 12 cm above the goal, dominant +x"
    s["eef"] = np.array([0.4, 0.0, -0.2])
    assert command(s) == (2, 1), "above the goal: up"
    assert not far_from_target(s), "held: gate off"
    s = {"eef": np.zeros(3), "target": np.array([0.05, 0.05, -0.1]), "held": False}
    assert not far_from_target(s) and far_from_target({**s, "target": np.array([0.1, 0.0, 0.0])}), "gate"
    print("controller.py: ok")
