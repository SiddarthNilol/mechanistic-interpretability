"""Stage 2 module 2: automatic disturbances during a rollout. LIBERO env.

Five kinds, all applied by the simulator, never by the policy:

    none         no disturbance (the reference)
    drop         while the target is carried (held, lifted LIFT m above where it started) and
                 not yet near the goal, at a random time: force the gripper open for OPEN_STEPS
                 steps so the object falls, then return control
    shift        before the grasp, when the gripper first comes within APPROACH m (horizontal)
                 of the target, or at step SHIFT_LATEST if it never does: teleport the target a
                 few cm in a random horizontal direction
    distractor   at the start of the episode: put the nearest other bowl next to the target, on
                 the side that keeps the instruction true (task 60: the target stays the left
                 bowl; task 12: it stays the bowl at the back) and puts the distractor between the
                 gripper's start and the target. Object identities are kept, because LIBERO scores
                 the named object.
    moved_goal   while the target is carried, with the same trigger as the drop: teleport the
                 goal object (tray or plate) a few cm in a random horizontal direction

Randomness (timing, size, direction) comes from a seed per (task, initial state, kind, stream),
drawn independently of the policy's noise seeds. So the unsteered and steered episodes on one
initial state get the same disturbance parameters. They are still not identical: the trigger is
a state of the episode (for example "carried"), which each condition reaches at its own time.

Each disturbance records what it did in `info` (step, offsets, what the scene looked like then).
"""

from __future__ import annotations

import numpy as np

import libero_env as L

KINDS = ("none", "drop", "shift", "distractor", "moved_goal")

LIFT = 0.03  # m above the target's starting height: "carried", not just grasped
OPEN_STEPS = 10  # drop: steps with the gripper forced open (0.5 s at 20 Hz)
DELAY = (0, 20)  # drop, moved_goal: steps after the carry starts, uniform, inclusive
PROGRESS = 0.5  # ... or earlier, once the target has covered half its way to the goal
SHIFT_SIZE = (0.03, 0.06)  # m
APPROACH = 0.10  # m, horizontal: shift fires when the gripper first gets this close
SHIFT_LATEST = 60  # steps: shift fires here if the gripper never got close
GOAL_SIZE = (0.05, 0.08)  # m
DISTRACTOR = {  # task -> (direction from target in the world xy plane, center-to-center distance, m)
    60: ((0.0, 1.0), 0.10),  # PLACEHOLDER, set from results/probe/stage2_probe.json
    12: ((1.0, 0.0), 0.10),  # PLACEHOLDER
}


def draw(task_index: int, state_index: int, kind: str, stream: int = 0) -> dict:
    """The disturbance's random parameters, a function of the seed only."""
    seed = np.random.SeedSequence([stream, task_index, state_index, 10_000 + KINDS.index(kind)])
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0, 2 * np.pi))
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    if kind == "drop":
        return {"delay": int(rng.integers(DELAY[0], DELAY[1] + 1))}
    if kind == "shift":
        return {"offset": (float(rng.uniform(*SHIFT_SIZE)) * direction).tolist()}
    if kind == "moved_goal":
        return {"delay": int(rng.integers(DELAY[0], DELAY[1] + 1)),
                "offset": (float(rng.uniform(*GOAL_SIZE)) * direction).tolist()}
    return {}


def _horizontal(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b))[:2]))


class Disturbance:
    """Call order in the episode loop: `at_reset` once after the reset, then at every step
    `before_step` (may teleport an object) and `gripper` (a forced gripper value or None)."""

    def __init__(self, kind: str, task_index: int, state_index: int, stream: int = 0):
        if kind not in KINDS:
            raise ValueError(f"unknown disturbance {kind!r}")
        self.kind = kind
        self.task = L.TASKS[task_index]
        self.params = draw(task_index, state_index, kind, stream)
        self.info: dict = {"kind": kind, "params": self.params, "fired": False}
        self._carry_start = None
        self._open_until = -1

    # -- hooks -----------------------------------------------------------------------------

    def at_reset(self, env, obs):
        self._start = L.body_position(env, self.task.target)
        if self.kind != "distractor":
            return obs
        direction, distance = DISTRACTOR[self.task.index]
        others = [n for n in L._inner(env).objects_dict if "bowl" in n and n != self.task.target]
        nearest = min(others, key=lambda n: _horizontal(L.body_position(env, n), self._start))
        spot = self._start + distance * np.array([direction[0], direction[1], 0.0])
        before = L.body_position(env, nearest)
        spot[2] = before[2]
        L.move_object(env, nearest, spot - before)
        for _ in range(L.SETTLE_STEPS):
            obs, _, _, _ = env.step(L.SETTLE_ACTION)
        self._start = L.body_position(env, self.task.target)
        self.info.update(fired=True, step=0, distractor=nearest, placed_at=spot.round(4).tolist(),
                         distractor_after_settle=L.body_position(env, nearest).round(4).tolist())
        return obs

    def before_step(self, env, obs, step: int) -> None:
        if self.info["fired"] or self.kind in ("none", "distractor"):
            return
        target = L.body_position(env, self.task.target)
        goal = L.body_position(env, self.task.goal)
        eef = np.asarray(obs["robot0_eef_pos"])
        if self.kind == "shift":
            if L.is_held(env, self.task.target):
                self.info.update(fired=True, step=step, skipped="held before the shift fired")
                return
            close = _horizontal(eef, target) <= APPROACH
            if close or step >= SHIFT_LATEST:
                L.move_object(env, self.task.target, self.params["offset"])
                self.info.update(fired=True, step=step, why="approach" if close else "latest",
                                 target_before=target.round(4).tolist(), eef=eef.round(4).tolist())
            return
        # drop and moved_goal share the "carried" trigger
        carried = L.is_held(env, self.task.target) and target[2] > self._start[2] + LIFT
        if not carried:
            return
        if self._carry_start is None:
            self._carry_start = step
            self._carry_distance = _horizontal(target, goal)
        progressed = _horizontal(target, goal) <= PROGRESS * self._carry_distance
        if step - self._carry_start < self.params["delay"] and not progressed:
            return
        self.info.update(fired=True, step=step, carry_start=self._carry_start,
                         why="delay" if not progressed else "progress", target_before=target.round(4).tolist(),
                         goal_before=goal.round(4).tolist(), eef=eef.round(4).tolist())
        if self.kind == "drop":
            self._open_until = step + OPEN_STEPS
            self.info["open_until"] = self._open_until
        else:
            L.move_object(env, self.task.goal, self.params["offset"])

    def gripper(self, step: int):
        """-1 (open) while a drop is forcing the gripper, else None (the policy's own value)."""
        return -1.0 if step < self._open_until else None
