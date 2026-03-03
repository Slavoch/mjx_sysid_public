"""
Replay a dataset in real time: 1 second of data = 1 second wall clock.
No simulation — only sets qpos/qvel from the dataset and renders.
Use for viewing recorded motions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Run from repo root so task config paths resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer

from config.registry import available_tasks, get_task_config
from sysidmjx.assets.single_motor.dataloader.single_joint import data_load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play dataset in real time (replay states only). 1 sec data = 1 sec wall clock."
    )
    parser.add_argument("--task", type=str, help="Task key (e.g. roki2.torque).")
    parser.add_argument("--list-tasks", action="store_true", help="List registered task keys and exit.")
    return parser.parse_args()


def _get_config(task: str) -> tuple[str, str, float, int, int]:
    """Resolve dataset path, model path, dt, num_lags, motor_id from task config."""
    cfg = get_task_config(task)
    dataset_path = cfg.data.dataset_path
    model_xml = cfg.sim.model_xml
    if not Path(dataset_path).is_absolute():
        dataset_path = str(REPO_ROOT / dataset_path)
    if not Path(model_xml).is_absolute():
        model_xml = str(REPO_ROOT / model_xml)
    return dataset_path, model_xml, cfg.data.dt, cfg.data.num_lags, cfg.data.motor_id


def run_replay(
    *,
    dataset_path: str,
    model_xml: str,
    dt: float,
    motor_id: int,
    num_lags: int,
) -> None:
    _, df = data_load(
        data_path=dataset_path,
        motor_id=motor_id,
        num_lags=num_lags,
        dtype=np.float32,
    )

    qpos = np.asarray(df["qpos"].values)
    qvel = np.asarray(df["qvel"].values)
    if qpos.ndim > 1:
        qpos = qpos.squeeze(axis=1)
    if qvel.ndim > 1:
        qvel = qvel.squeeze(axis=1)
    n_steps = len(qpos)

    mj_model = mujoco.MjModel.from_xml_path(model_xml)
    mj_model.opt.timestep = dt
    mj_data = mujoco.MjData(mj_model)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while True:
            t0 = time.perf_counter()
            for i in range(n_steps):
                mj_data.qpos[:] = qpos[i]
                mj_data.qvel[:] = qvel[i]
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                elapsed = time.perf_counter() - t0
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Dataset ended — viewer stays on last frame
            input("Press Enter to replay.")


def main() -> int:
    args = _parse_args()

    if args.list_tasks:
        for key in available_tasks():
            print(key)
        return 0

    if not args.task:
        print("Missing --task. Use --list-tasks to see available tasks.", file=sys.stderr)
        return 1

    dataset_path, model_xml, dt, num_lags, motor_id = _get_config(args.task)

    run_replay(
        dataset_path=dataset_path,
        model_xml=model_xml,
        dt=dt,
        motor_id=motor_id,
        num_lags=num_lags,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
