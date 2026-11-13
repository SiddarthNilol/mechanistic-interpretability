"""Module 1: the two LIBERO-90 tasks, and everything we read from or do to the simulator.

Runs in the LIBERO environment (Python 3.8, `source jobs/sim_env.sh`), not in openpi's.

    TASKS              task 12 and task 60 of LIBERO-90: instruction, target object, goal object
    make_env           the simulator for one task, set up as openpi's examples/libero/main.py does
    reset              one of the task's 50 fixed initial states, plus 10 settling steps
    policy_input       the observation in the format the pi05_libero server expects
    state              privileged simulator state: end effector, gripper, target, goal, held.
                       Allowed only for the scripted controller (oracle), labels and scoring.
    move_object        Stage 2 "shift": teleport the target object
    is_success         LIBERO's own goal check

LIBERO's gripper command is the 7th action dim: -1 opens, +1 closes (openpi's settling action
is [0, 0, 0, 0, 0, 0, -1]).
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

SETTLE_STEPS = 10  # openpi: let objects fall into place before the policy acts
MAX_STEPS = 400  # openpi's limit for LIBERO-90 (longest training demo 373 steps)
SETTLE_ACTION = [0.0] * 6 + [-1.0]
IMAGE_SIZE = 256  # LIBERO renders 256 x 256
POLICY_IMAGE_SIZE = 224  # openpi's client resizes (with padding) to 224 before sending


@dataclasses.dataclass(frozen=True)
class Task:
    index: int  # in LIBERO-90
    name: str
    target: str  # the object to pick up
    goal: str  # the object it must end up on or in


TASKS = {
    12: Task(12, "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate", "akita_black_bowl_3", "plate_1"),
    60: Task(60, "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
             "akita_black_bowl_1", "wooden_tray_1"),
}


def make_env(task_index: int, seed: int = 0):
    """(env, instruction, initial_states) for one LIBERO-90 task."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(task_index)
    if task.name != TASKS[task_index].name:
        raise RuntimeError(f"LIBERO-90 task {task_index} is {task.name}, expected {TASKS[task_index].name}")
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE)
    env.seed(seed)  # openpi: the seed changes object positions even with a fixed initial state
    return env, task.language, suite.get_task_init_states(task_index)


def reset(env, initial_state):
    """Puts the scene in a fixed initial state and lets it settle. Returns the observation."""
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(SETTLE_ACTION)
    return obs


def quat2axisangle(quat):
    """From openpi's examples/libero/main.py (itself from robosuite)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def policy_input(obs, instruction: str) -> dict:
    """The request openpi's LIBERO client sends (examples/libero/main.py): both images rotated
    180 degrees (as in the training data) and resized with padding to 224, the 8-dim robot
    state, and the prompt."""
    from openpi_client import image_tools

    def image(key):
        rotated = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE))

    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32),
        "prompt": instruction,
    }


# -- privileged state -------------------------------------------------------------------


def _inner(env):
    return env.env  # OffScreenRenderEnv wraps the robosuite environment


def body_position(env, name: str) -> np.ndarray:
    inner = _inner(env)
    return np.array(inner.sim.data.body_xpos[inner.obj_body_id[name]])


def is_held(env, name: str) -> bool:
    """robosuite's grasp check: both fingers' pads touch the object's geoms."""
    inner = _inner(env)
    return bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=inner.objects_dict[name]))


def state(env, obs, task: Task) -> dict:
    return {
        "eef": np.array(obs["robot0_eef_pos"]),
        "gripper": np.array(obs["robot0_gripper_qpos"]),  # two finger joint positions
        "target": body_position(env, task.target),
        "goal": body_position(env, task.goal),
        "held": is_held(env, task.target),
    }


def move_object(env, name: str, offset) -> None:
    """Teleports an object by `offset` (meters, world frame), keeping its orientation."""
    inner = _inner(env)
    joint = inner.objects_dict[name].joints[0]  # a free joint: 3 position + 4 quaternion
    qpos = np.array(inner.sim.data.get_joint_qpos(joint))
    qpos[:3] += np.asarray(offset, dtype=np.float64)
    inner.sim.data.set_joint_qpos(joint, qpos)
    inner.sim.data.set_joint_qvel(joint, np.zeros(6))
    inner.sim.forward()


def is_success(env) -> bool:
    return bool(_inner(env)._check_success())
