"""
Generate and visualize dataset vs learned-model trajectory.

Core API:
    play(task, checkpoint) -> (generated_trajectory, ground_truth_trajectory)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer
from mujoco import mjx

from config.registry import available_tasks, get_task_config
from sysidmjx.assets.single_motor.dataloader.single_joint import data_load
from sysidmjx.models import compose_action_model
from sysidmjx.optimizers.single_motor import _apply_jax_config_defaults, _build_component_factory
from sysidmjx.utils.xml_clone import make_twin_xml

# Validation playback settings (moderate, not overly strict).
# We keep dataset-level dt unchanged by running a few smaller substeps per sample.
_VAL_SUBSTEPS_PER_SAMPLE = 4
_VAL_SOLVER_ITERATIONS = 20
_VAL_LS_ITERATIONS = 20
_VAL_TOLERANCE = 1e-9
_VAL_INTEGRATOR = mujoco.mjtIntegrator.mjINT_RK4

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory from checkpoint and visualize."
    )
    parser.add_argument("--task", type=str, help="Task key (e.g. roki2.torque).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to .pkl checkpoint file.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List tasks and exit.")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _resolve_checkpoint_path(checkpoint: str) -> Path:
    cp = _resolve(checkpoint)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp}")
    if not cp.is_file() or cp.suffix != ".pkl":
        raise ValueError(f"--checkpoint must be a .pkl file: {cp}")
    return cp


def _twin_model_path(single_model_xml: Path) -> Path:
    """Path to twin XML in same dir, e.g. motor_roki.xml -> motor_roki_twin.xml."""
    return single_model_xml.parent / (single_model_xml.stem + "_twin.xml")


def _get_or_create_twin_xml(single_model_xml: Path) -> Path:
    """Regenerate twin XML next to the single model and return its path."""
    twin_path = _twin_model_path(single_model_xml)
    make_twin_xml(
        single_model_xml,
        twin_path,
        body_suffixes=("_dataset", "_sim"),
        clone_positions=("0.5 0 0.5", "0 0 0.5"),
        clone_geom_rgba=("0.2 0.9 0.3 1", None),
        twin_model_name="twin_motor",
    )
    return twin_path


def _apply_changers_to_model(model: mujoco.MjModel, params: dict | None) -> None:
    if not params:
        return
    changers = params.get("changers", {})
    for ch in changers.values():
        if "armature" in ch:
            model.dof_armature[:] = np.abs(np.asarray(ch["armature"]).reshape(-1))
        if "damping" in ch:
            model.dof_damping[:] = np.abs(np.asarray(ch["damping"]).reshape(-1))
        if "frictionloss" in ch:
            model.dof_frictionloss[:] = np.abs(np.asarray(ch["frictionloss"]).reshape(-1))


def _set_high_precision_validation_options(model: mujoco.MjModel, dt: float) -> None:
    model.opt.timestep = dt / float(_VAL_SUBSTEPS_PER_SAMPLE)
    model.opt.integrator = _VAL_INTEGRATOR
    model.opt.iterations = int(_VAL_SOLVER_ITERATIONS)
    if hasattr(model.opt, "ls_iterations"):
        model.opt.ls_iterations = int(_VAL_LS_ITERATIONS)
    if hasattr(model.opt, "tolerance"):
        model.opt.tolerance = float(_VAL_TOLERANCE)


def _prepare_rollout(
    task: str,
    checkpoint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
    _apply_jax_config_defaults()
    print(f"[play] Loading task config: {task}")
    cfg = get_task_config(task)
    best_params_path = _resolve_checkpoint_path(checkpoint)
    print(f"[play] Using checkpoint: {best_params_path}")

    dataset_path = _resolve(cfg.data.dataset_path)
    model_xml = _resolve(cfg.sim.model_xml)
    print(f"[play] Loading dataset: {dataset_path}")

    dataset, df = data_load(
        data_path=str(dataset_path),
        motor_id=cfg.data.motor_id,
        num_lags=cfg.data.num_lags,
        dtype=jnp.float64,
    )
    print(f"[play] Dataset loaded: {len(df)} rows")
    qpos_meas = np.asarray(df["qpos"].values, dtype=np.float64).reshape(-1)
    qvel_meas = np.asarray(df["qvel"].values, dtype=np.float64).reshape(-1)
    ctrl_target = np.asarray(df["qact"].values, dtype=np.float64).reshape(-1)
    if qpos_meas.ndim > 1:
        qpos_meas = qpos_meas.squeeze()
    if qvel_meas.ndim > 1:
        qvel_meas = qvel_meas.squeeze()
    if ctrl_target.ndim > 1:
        ctrl_target = ctrl_target.squeeze()
    qpos_meas = np.atleast_1d(qpos_meas)
    qvel_meas = np.atleast_1d(qvel_meas)
    ctrl_target = np.atleast_1d(ctrl_target)

    print("[play] Loading model parameters...")
    with best_params_path.open("rb") as f:
        best_params = pickle.load(f)
    print("[play] Model parameters loaded")

    # Single-joint model for running learned rollout
    print(f"[play] Building MuJoCo model: {model_xml}")
    model_single = mujoco.MjModel.from_xml_path(str(model_xml))
    _set_high_precision_validation_options(model_single, cfg.data.dt)
    _apply_changers_to_model(model_single, best_params)
    data_single = mujoco.MjData(model_single)

    controller_factories = [
        _build_component_factory(c, dataset) for c in cfg.experiment.controllers
    ]
    friction_factories = [
        _build_component_factory(c, dataset) for c in cfg.experiment.frictions
    ]
    changer_factories = [
        _build_component_factory(c, dataset) for c in cfg.experiment.changers
    ]
    make_action, _, _ = compose_action_model(
        controllers=controller_factories,
        frictions=friction_factories,
        changers=changer_factories,
        seed=jax.random.PRNGKey(cfg.train.seed),
    )

    mjx_data0 = mjx.put_data(model_single, data_single)
    data_single.qpos[:] = qpos_meas[0]
    data_single.qvel[:] = qvel_meas[0]
    mujoco.mj_forward(model_single, data_single)

    T = min(len(ctrl_target), len(qpos_meas))
    print(f"[play] Running rollout for {T} steps...")
    progress_every = max(1, T // 10)
    sim_qpos = np.zeros(T)
    for t in range(T):
        sim_qpos[t] = float(data_single.qpos[0])
        for _ in range(_VAL_SUBSTEPS_PER_SAMPLE):
            d = mjx_data0.replace(
                qpos=jnp.asarray(data_single.qpos),
                qvel=jnp.asarray(data_single.qvel),
                qfrc_bias=jnp.asarray(data_single.qfrc_bias),
            )
            ctrl_t = jnp.asarray([float(ctrl_target[t])], dtype=jnp.float32)
            time_index = jnp.asarray(t, dtype=jnp.int32)
            tau = make_action(best_params, d, ctrl_t, time_index)
            tau_np = np.asarray(tau).reshape(-1)
            data_single.ctrl[:] = tau_np[: model_single.nu]
            mujoco.mj_step(model_single, data_single)
        if (t + 1) % progress_every == 0 or (t + 1) == T:
            pct = 100.0 * float(t + 1) / float(T)
            print(f"[play] Rollout progress: {t + 1}/{T} ({pct:.0f}%)")
    gt_qpos = qpos_meas[:T].copy()
    print("[play] Rollout finished")
    return sim_qpos, gt_qpos, qvel_meas[:T].copy(), cfg.data.dt, model_xml


def play(task: str, checkpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run learned rollout for a task/checkpoint without visualization.

    Returns:
        generated_trajectory: simulated qpos trajectory from learned model
        ground_truth_trajectory: measured dataset qpos trajectory (same length)
    """
    sim_qpos, gt_qpos, _, _, _ = _prepare_rollout(task=task, checkpoint=checkpoint)
    return sim_qpos, gt_qpos


def _visualize(
    generated_qpos: np.ndarray,
    ground_truth_qpos: np.ndarray,
    ground_truth_qvel: np.ndarray,
    dt: float,
    model_xml: Path,
) -> None:
    twin_xml = _get_or_create_twin_xml(model_xml)
    model_twin = mujoco.MjModel.from_xml_path(str(twin_xml))
    model_twin.opt.timestep = dt
    data_twin = mujoco.MjData(model_twin)
    T = min(len(generated_qpos), len(ground_truth_qpos), len(ground_truth_qvel))

    with mujoco.viewer.launch_passive(model_twin, data_twin) as viewer:
        while True:
            t0 = time.perf_counter()
            for i in range(T):
                # Dataset replay (green, +50 cm x): qpos[0], qvel[0]
                data_twin.qpos[0] = ground_truth_qpos[i]
                data_twin.qvel[0] = ground_truth_qvel[i]
                # Generated trajectory (origin): qpos[1], qvel[1]
                data_twin.qpos[1] = generated_qpos[i]
                data_twin.qvel[1] = 0.0
                mujoco.mj_forward(model_twin, data_twin)
                viewer.sync()
                elapsed = time.perf_counter() - t0
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            input("Press Enter to replay.")


def run() -> None:
    args = _parse_args()

    if args.list_tasks:
        for key in available_tasks():
            print(key)
        return

    if not args.task or not args.checkpoint:
        raise ValueError("Missing --task or --checkpoint. Use --list-tasks to see tasks.")

    generated_qpos, ground_truth_qpos, ground_truth_qvel, dt, model_xml = _prepare_rollout(
        task=args.task,
        checkpoint=args.checkpoint,
    )
    print(f"Generated trajectory length: {len(generated_qpos)}")
    print(f"Ground-truth trajectory length: {len(ground_truth_qpos)}")

    _visualize(
        generated_qpos=generated_qpos,
        ground_truth_qpos=ground_truth_qpos,
        ground_truth_qvel=ground_truth_qvel,
        dt=dt,
        model_xml=model_xml,
    )


def main() -> int:
    try:
        run()
        return 0
    except Exception as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
