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
from sysidmjx.assets.single_motor.dataloader.single_joint import SingleMotorDataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play dataset in real time (replay states only). 1 sec data = 1 sec wall clock."
    )
    parser.add_argument("--task", type=str, help="Task key (e.g. encos8112.pd_then_changers).")
    parser.add_argument("--list-tasks", action="store_true", help="List registered task keys and exit.")
    return parser.parse_args()


def _get_config(task: str):
    """Resolve task config for dataset replay."""
    cfg = get_task_config(task)
    model_xml = cfg.data.dataset.model_xml
    if not Path(model_xml).is_absolute():
        model_xml = str(REPO_ROOT / model_xml)
    return cfg, model_xml


def run_replay(
    *,
    cfg,
    model_xml: str,
) -> None:
    loader = SingleMotorDataLoader.from_spec(
        spec=cfg.data.dataset,
        history_len=cfg.data.history_len,
        prediction_horizon=cfg.data.prediction_horizon,
        dtype=np.float32,
        repo_root=REPO_ROOT,
    )
    run_frame = loader.frames_by_run[0]
    df = run_frame.frame

    qpos = df["qpos"].to_numpy(dtype=np.float64)
    qvel = df["qvel"].to_numpy(dtype=np.float64)
    n_steps = len(qpos)

    mj_model = mujoco.MjModel.from_xml_path(model_xml)
    mj_model.opt.timestep = cfg.data.dt
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
                sleep_time = (i + 1) * cfg.data.dt - elapsed
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

    cfg, model_xml = _get_config(args.task)

    run_replay(
        cfg=cfg,
        model_xml=model_xml,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
