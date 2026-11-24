"""Multi-stage experiment: task specs and the subgoal oracle for LIBERO-90 tasks 3, 4, 63, 64.
numpy only (plus the LIBERO env object passed in). Kept apart from src/sim so Stage 4 code is
untouched; it reuses libero_env's readers unchanged.

Each task's goal (LIBERO's BDDL) is two predicates that must hold together; the oracle works
through them in order, re-deciding the subgoal from privileged state at every replan:

    tasks 3, 4    1. In(butter, top drawer region)    grasp the butter, carry it above the open
                                                      drawer (the policy releases it)
                  2. Close(top drawer)                go to a point in front of the drawer face,
                                                      then push it shut (-y)
    tasks 63, 64  1. On(upper bowl, lower bowl)       grasp the upper bowl, carry it above the
                                                      lower one
                  2. In(lower bowl, tray)             grasp the stack (the lower bowl), carry it
                                                      above the tray

Command = the dominant axis of end effector -> subgoal point, as in src/sim/controller.py.
`gate` marks replans where the Stage 3 extensions may act: nothing held and the end effector
more than GATE from the point it is reaching for (horizontally), as in Stage 3.
Probe (src/multistage/probe.py): the drawer opens along +y, its top region site moves 0.145 to
0.149 m in -y when closed.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
import libero_env as L  # noqa: E402

GATE = 0.08  # m, Stage 3's gate
ABOVE_DRAWER = np.array([0.0, 0.0, 0.06])  # carry point over the open drawer's region site
ABOVE_BOWL = np.array([0.0, 0.0, 0.10])  # carry point over the lower bowl
ABOVE_TRAY = np.array([0.0, 0.0, 0.12])  # as the Stage 1 controller's goal offset
DRAWER_FRONT = np.array([0.0, 0.11, 0.0])  # from the region site to just in front of the drawer face
PUSH_RADIUS = 0.04  # within this of the front point: push the drawer shut


@dataclasses.dataclass(frozen=True)
class Spec:
    index: int
    name: str
    kind: str  # "drawer" or "stack"
    first: str  # object picked first (butter, or the upper bowl)
    second: str  # drawer region site, or the lower bowl


SPECS = {
    3: Spec(3, "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
            "drawer", "butter_2", "wooden_cabinet_1_top_region"),
    4: Spec(4, "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
            "drawer", "butter_1", "wooden_cabinet_1_top_region"),
    63: Spec(63, "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
             "stack", "akita_black_bowl_1", "akita_black_bowl_2"),
    64: Spec(64, "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
             "stack", "akita_black_bowl_2", "akita_black_bowl_1"),
}
TRAY = "wooden_tray_1_contain_region"


def make_env(task_index: int, seed: int = 0):
    """(env, instruction, initial_states), as libero_env.make_env but for these tasks."""
    import os

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(task_index)
    if task.name != SPECS[task_index].name:
        raise RuntimeError(f"LIBERO-90 task {task_index} is {task.name}, expected {SPECS[task_index].name}")
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=L.IMAGE_SIZE, camera_widths=L.IMAGE_SIZE)
    env.seed(seed)
    return env, task.language, suite.get_task_init_states(task_index)


def site_position(env, name: str) -> np.ndarray:
    return np.array(L._inner(env).object_states_dict[name].get_geom_state()["pos"])


def goal_status(env) -> list[bool]:
    """LIBERO's goal predicates, each on its own, in BDDL order."""
    inner = L._inner(env)
    return [bool(inner._eval_predicate(g)) for g in inner.parsed_problem["goal_state"]]


def observe(env, obs, spec: Spec) -> dict:
    """Privileged state for the oracle and the labels."""
    s = {"eef": np.array(obs["robot0_eef_pos"]), "goals": goal_status(env),
         "first": L.body_position(env, spec.first), "held_first": L.is_held(env, spec.first)}
    if spec.kind == "drawer":
        s["region"] = site_position(env, spec.second)
        s["held_second"] = False
    else:
        s["second"] = L.body_position(env, spec.second)
        s["held_second"] = L.is_held(env, spec.second)
        s["tray"] = site_position(env, TRAY)
    return s


def subgoal(s: dict, spec: Spec) -> tuple[str, np.ndarray, bool]:
    """(phase name, point to move toward, reaching = nothing held and heading to an object or
    the drawer face)."""
    done1, _ = s["goals"][:2]
    if spec.kind == "drawer":
        if not done1:
            if s["held_first"]:
                return "carry_butter", s["region"] + ABOVE_DRAWER, False
            return "grasp_butter", s["first"], True
        return "close_drawer", s["region"] + DRAWER_FRONT, True
    if not done1:
        if s["held_first"]:
            return "carry_upper", s["second"] + ABOVE_BOWL, False
        return "grasp_upper", s["first"], True
    if s["held_second"] or s["held_first"]:
        return "carry_stack", s["tray"] + ABOVE_TRAY, False
    return "grasp_stack", s["second"], True


def command(s: dict, spec: Spec) -> tuple[str, tuple[int, int], bool]:
    """(phase, (axis, sign), gate)."""
    phase, point, reaching = subgoal(s, spec)
    vector = point - s["eef"]
    if phase == "close_drawer" and np.linalg.norm(vector) < PUSH_RADIUS:
        return "push_drawer", (1, -1), False
    axis = int(np.argmax(np.abs(vector)))
    gate = reaching and float(np.linalg.norm(vector[:2])) > GATE
    return phase, (axis, 1 if vector[axis] >= 0 else -1), gate


if __name__ == "__main__":  # self-test: python src/multistage/oracle.py
    d = SPECS[3]
    s = {"eef": np.zeros(3), "goals": [False, False], "first": np.array([0.0, 0.3, 0.0]), "held_first": False,
         "region": np.array([0.0, -0.2, 0.1]), "held_second": False}
    assert command(s, d) == ("grasp_butter", (1, 1), True)
    assert command({**s, "held_first": True}, d)[:2] == ("carry_butter", (1, -1))
    s2 = {**s, "goals": [True, False], "eef": s["region"] + DRAWER_FRONT + 0.01}
    assert command(s2, d) == ("push_drawer", (1, -1), False)
    k = SPECS[63]
    s = {"eef": np.zeros(3), "goals": [True, False], "first": np.zeros(3), "held_first": False,
         "second": np.array([0.0, 0.0, -0.2]), "held_second": False, "tray": np.array([0.0, 0.3, 0.0])}
    assert command(s, k) == ("grasp_stack", (2, -1), False)
    assert command({**s, "held_second": True}, k)[:2] == ("carry_stack", (1, 1))
    print("oracle.py: ok")
