"""Module 2a: Flow Control's starting noise. numpy only, so it can be tested anywhere.

pi0.5 makes an action chunk by starting from noise x0 (shape 10 x 32, N(0, I), in the
normalized, zero-padded action space) and integrating the action expert's velocity for 10 Euler
steps. The flow is deterministic given x0, so x0 decides which of the many plausible chunks comes
out. Flow Control (arXiv 2606.10180, Appendix G.1 for LIBERO) steers by writing the command into
x0 before integration:

    chunk steps 0 .. tau-1:   translation dim `axis` = sign * sigma
                              the other 5 translation and rotation dims = 0
    chunk steps tau .. 9:     unchanged noise
    gripper (dim 6) and padding (dims 7 to 31): unchanged noise, at every step

with tau = 5 and sigma = 1.5. LIBERO's action dims are (dx, dy, dz, droll, dpitch, dyaw,
gripper), so axis 0, 1, 2 = x, y, z of the end effector in the robot base frame.

Pairing: the noise is a function of a seed only, so an unsteered and a steered episode that use
the same seed at the same replan start from the same noise (the steered one with the command
written over part of it).
"""

from __future__ import annotations

import numpy as np

HORIZON = 10  # pi05_libero's chunk length
MODEL_ACTION_DIM = 32  # the model's padded action dim
TAU = 5  # chunk steps that carry the command
SIGMA = 1.5  # command magnitude, in normalized action units
MOTION_DIMS = 6  # dx, dy, dz, droll, dpitch, dyaw


def base_noise(seed: int) -> np.ndarray:
    """The unsteered starting noise for one replan: N(0, I), (10, 32), float32."""
    return np.random.default_rng(seed).standard_normal((HORIZON, MODEL_ACTION_DIM)).astype(np.float32)


def inject(noise: np.ndarray, axis: int, sign: int, *, tau: int = TAU, sigma: float = SIGMA) -> np.ndarray:
    """Flow Control's "set" injection: a copy of `noise` with the command written into it."""
    if axis not in (0, 1, 2) or sign not in (-1, 1):
        raise ValueError(f"command must be a translation axis 0-2 and a sign +-1, got {axis}, {sign}")
    steered = np.array(noise, copy=True)
    steered[:tau, :MOTION_DIMS] = 0.0
    steered[:tau, axis] = sign * sigma
    return steered


def episode_seed(task_index: int, initial_state: int, replan: int, stream: int = 0) -> int:
    """One seed per (task, initial state, replan index, noise stream), the same for every
    condition, so conditions are paired."""
    return int(np.random.SeedSequence([stream, task_index, initial_state, replan]).generate_state(1)[0])


if __name__ == "__main__":  # self-test: python src/policy/injection.py
    x = base_noise(1)
    assert np.array_equal(x, base_noise(1)) and not np.array_equal(x, base_noise(2)), "seeding"
    y = inject(x, axis=2, sign=-1)
    assert np.all(y[:TAU, 2] == -SIGMA) and np.all(y[:TAU, [0, 1, 3, 4, 5]] == 0), "command rows"
    assert np.array_equal(y[TAU:], x[TAU:]), "later chunk steps untouched"
    assert np.array_equal(y[:, MOTION_DIMS:], x[:, MOTION_DIMS:]), "gripper and padding untouched"
    assert episode_seed(12, 3, 7) == episode_seed(12, 3, 7) != episode_seed(12, 3, 8), "episode seeds"
    print("injection.py: ok")
