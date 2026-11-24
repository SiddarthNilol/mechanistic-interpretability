"""Multi-stage experiment, step 1: what the simulator exposes for LIBERO-90 tasks 3, 4, 63, 64.
No policy. LIBERO env:

    sbatch jobs/sim_job.sh python src/multistage/probe.py

Per task, initial state 0: task name and instruction, object and site names, the goal's
predicates and their value now, object and site positions, the end effector. For the cabinet
tasks also the drawer joints, and where the top region site moves when the drawer is closed
(the closing direction the oracle needs).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
import libero_env as L  # noqa: E402

TASKS = (3, 4, 63, 64)


def make_env(task_index: int):
    import os

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(task_index)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=L.IMAGE_SIZE, camera_widths=L.IMAGE_SIZE)
    env.seed(0)
    return env, task, suite.get_task_init_states(task_index)


def site_pos(inner, name):
    return np.array(inner.object_states_dict[name].get_geom_state()["pos"])


def main():
    for t in TASKS:
        env, task, init = make_env(t)
        obs = L.reset(env, init[0])
        inner = env.env
        print(f"\n== task {t}: {task.name}\n   {task.language!r}, {len(init)} initial states")
        goal = inner.parsed_problem["goal_state"]
        print("goal:", goal, "now:", [bool(inner._eval_predicate(g)) for g in goal])
        print("eef:", np.round(obs["robot0_eef_pos"], 3).tolist())
        for name in inner.objects_dict:
            print(f"  object {name}: {np.round(L.body_position(env, name), 3).tolist()}")
        for name in inner.object_states_dict:
            if name not in inner.objects_dict:
                try:
                    print(f"  state {name}: {np.round(site_pos(inner, name), 3).tolist()}")
                except Exception as e:  # noqa: BLE001
                    print(f"  state {name}: ({type(e).__name__})")
        joints = [inner.sim.model.joint_id2name(i) for i in range(inner.sim.model.njnt)]
        cab = [j for j in joints if j and "cabinet" in j]
        if cab:
            print("cabinet joints:", {j: round(float(inner.sim.data.get_joint_qpos(j)), 4) for j in cab})
            region = "wooden_cabinet_1_top_region"
            before = site_pos(inner, region)
            top = [j for j in cab if "top" in j][0]
            inner.sim.data.set_joint_qpos(top, 0.0)
            inner.sim.forward()
            after = site_pos(inner, region)
            print(f"top region site: open {np.round(before, 3).tolist()} -> closed {np.round(after, 3).tolist()}, "
                  f"closing moves it by {np.round(after - before, 3).tolist()}")
            print("goal now (drawer forced closed):", [bool(inner._eval_predicate(g)) for g in goal])
        env.close()


if __name__ == "__main__":
    main()
