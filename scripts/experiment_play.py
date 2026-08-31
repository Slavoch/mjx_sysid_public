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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mujoco.viewer
from mujoco import mjx

from config.registry import available_tasks, get_task_config
from sysidmjx.core import generate_loss_train_functions
from sysidmjx.models import compose_action_model
from sysidmjx.optimizers.single_motor import _build_component_factory
from sysidmjx.utils.xml_clone import make_twin_xml

# Validation playback settings (moderate, not overly strict).
# We keep dataset-level dt unchanged by running a few smaller substeps per sample.
_VAL_SUBSTEPS_PER_SAMPLE = 1
_VAL_SOLVER_ITERATIONS = 1
_VAL_LS_ITERATIONS = 20
_VAL_TOLERANCE = 1e-9
_VAL_INTEGRATOR = mujoco.mjtIntegrator.mjINT_EULER


def _apply_jax_playback_config() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory from checkpoint and visualize."
    )
    parser.add_argument("--task", type=str, help="Task key (e.g. encos8112.pd_then_changers).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to .pkl checkpoint file.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Dataset run index to replay.",
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
        if "actuator_filter_timeconst" in ch:
            model.actuator_dynprm[0, 0] = float(
                np.abs(np.asarray(ch["actuator_filter_timeconst"]).reshape(-1)[0])
            )
        if "actuator_gain" in ch:
            model.actuator_gainprm[0, 0] = float(
                np.abs(np.asarray(ch["actuator_gain"]).reshape(-1)[0])
            )


def _set_high_precision_validation_options(model: mujoco.MjModel, dt: float) -> None:
    model.opt.timestep = dt / float(_VAL_SUBSTEPS_PER_SAMPLE)
    model.opt.integrator = _VAL_INTEGRATOR
    model.opt.iterations = int(_VAL_SOLVER_ITERATIONS)
    if hasattr(model.opt, "ls_iterations"):
        model.opt.ls_iterations = int(_VAL_LS_ITERATIONS)
    if hasattr(model.opt, "tolerance"):
        model.opt.tolerance = float(_VAL_TOLERANCE)


def _mlp_forward_numpy(mlp_params: list[dict], x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    for layer in mlp_params[:-1]:
        y = np.maximum(y @ np.asarray(layer["w"]) + np.asarray(layer["b"]), 0.0)
    y = y @ np.asarray(mlp_params[-1]["w"]) + np.asarray(mlp_params[-1]["b"])
    return float(np.ravel(y)[0])


def _make_numpy_action(cfg, params: dict):
    model_name = getattr(cfg.experiment.models[0].func, "__name__", "")
    if (
        len(cfg.experiment.models) != 1
        or cfg.experiment.frictions
        or model_name not in {"get_model_pd_residual", "get_model_isotropic_pd_residual"}
        or len(params.get("models", {})) != 1
    ):
        return None

    isotropic = model_name == "get_model_isotropic_pd_residual"
    model_cfg = cfg.experiment.models[0].params
    model_params = next(iter(params["models"].values()))
    input_signals = tuple(model_cfg.get("input_signals", ("tau_cmd", "qpos", "qvel")))
    target_qpos_field = model_cfg.get("target_qpos_field", "target_qpos")
    target_qvel_field = model_cfg.get("target_qvel_field", "target_qvel")
    kp_field = model_cfg.get("kp_field", "kp")
    kd_field = model_cfg.get("kd_field", "kd")

    def action(data: mujoco.MjData, control: dict[str, np.ndarray], i: int) -> np.ndarray:
        tau_pd = (
            control[kp_field][i] * (control[target_qpos_field][i] - data.qpos)
            + control[kd_field][i] * (control[target_qvel_field][i] - data.qvel)
        )
        features = []
        for signal in input_signals:
            if signal in ("tau_cmd", "tau_pd"):
                features.append(tau_pd)
            elif signal == "qpos":
                features.append(data.qpos)
            elif signal == "qvel":
                features.append(data.qvel)
            elif signal == "logged_tau_cmd":
                features.append(control["tau_cmd"][i])
            else:
                features.append(control[signal][i])
        nn_in = np.concatenate([np.ravel(x) for x in features])
        delta_tau = _mlp_forward_numpy(model_params["mlp"], nn_in)
        if isotropic:
            delta_tau -= _mlp_forward_numpy(model_params["mlp"], -nn_in)
        return tau_pd + delta_tau

    return action


class PlaybackRunner:
    def __init__(self, task: str, checkpoint: str) -> None:
        _apply_jax_playback_config()
        print(f"[play] Loading task config: {task}")
        self.cfg = get_task_config(task)
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint)
        print(f"[play] Using checkpoint: {self.checkpoint_path}")

        self.model_xml = _resolve(self.cfg.data.dataset.model_xml)
        print("[play] Loading model parameters...")
        with self.checkpoint_path.open("rb") as f:
            self.best_params = pickle.load(f)
        print("[play] Model parameters loaded")

        print(f"[play] Building MuJoCo model: {self.model_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        _set_high_precision_validation_options(self.model, self.cfg.data.dt)
        _apply_changers_to_model(self.model, self.best_params)
        self.numpy_action = _make_numpy_action(self.cfg, self.best_params)
        if self.numpy_action is not None:
            print("[play] Using native MuJoCo + NumPy action playback")
            return

        dataset_stub = {
            "qpos_hist": jnp.zeros(
                (1, self.cfg.data.history_len, self.model.nq),
                dtype=jnp.float64,
            ),
            "time_index": jnp.arange(1, dtype=jnp.int32),
        }
        model_factories = [
            _build_component_factory(c, dataset_stub)
            for c in self.cfg.experiment.models
        ]
        friction_factories = [
            _build_component_factory(c, dataset_stub)
            for c in self.cfg.experiment.frictions
        ]
        changer_factories = [
            _build_component_factory(c, dataset_stub)
            for c in self.cfg.experiment.changers
        ]
        self.make_action, _, _ = compose_action_model(
            models=model_factories,
            frictions=friction_factories,
            changers=changer_factories,
            seed=jax.random.PRNGKey(self.cfg.train.seed),
        )
        mjx_model = mjx.put_model(self.model)
        _, self.predict_next, _ = generate_loss_train_functions(
            mjx_model=mjx_model,
            mjx_data=mjx.make_data(mjx_model),
            change_model=lambda params, model: model,
            make_action=self.make_action,
            w_qpos=self.cfg.train.loss_w_qpos,
            w_qvel=self.cfg.train.loss_w_qvel,
            qpos_scale=self.cfg.train.loss_qpos_scale,
            qvel_scale=self.cfg.train.loss_qvel_scale,
            huber_delta=self.cfg.train.loss_huber_delta,
        )
        print("[play] Using compiled MJX playback")

    def run_index(
        self,
        run_index: int = 0,
        *,
        df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
        runs = self.cfg.data.dataset.runs
        if not -len(runs) <= run_index < len(runs):
            raise IndexError(
                f"run_index={run_index} is out of range for "
                f"{len(runs)} dataset runs"
            )
        run_path = _resolve(runs[run_index])
        print("[play] Loading dataset")
        if df is None:
            df = pd.read_csv(run_path).dropna().reset_index(drop=True)
        print(
            f"[play] Dataset run {run_index}: {run_path} "
            f"rows={len(df)}"
        )
        qpos_meas = df["qpos"].to_numpy(dtype=np.float64)
        qvel_meas = df["qvel"].to_numpy(dtype=np.float64)
        ctrl_target = df["qact"].to_numpy(dtype=np.float64)

        history_len = int(self.cfg.data.history_len)
        if len(qpos_meas) < history_len:
            raise ValueError(
                f"Dataset is shorter than history_len: rows={len(qpos_meas)} "
                f"history_len={history_len}"
            )

        start_idx = history_len - 1
        qpos_hist = qpos_meas[:history_len, None].astype(np.float64)
        qvel_hist = qvel_meas[:history_len, None].astype(np.float64)
        qact_hist = ctrl_target[:history_len, None].astype(np.float64)
        control_values = {
            name: df[column].to_numpy(dtype=np.float64)
            for name, column in self.cfg.data.dataset.rollout_fields.items()
        }

        rollout_steps = min(len(ctrl_target), len(qpos_meas)) - start_idx - 1
        if rollout_steps <= 0:
            raise ValueError("Dataset is too short for rollout playback.")

        if self.numpy_action is None:
            return self._run_index_mjx(
                qpos_meas=qpos_meas,
                qvel_meas=qvel_meas,
                ctrl_target=ctrl_target,
                qpos_hist=qpos_hist,
                qvel_hist=qvel_hist,
                qact_hist=qact_hist,
                control_values=control_values,
                start_idx=start_idx,
                rollout_steps=rollout_steps,
            )

        return self._run_index_mujoco_np(
            qpos_meas=qpos_meas,
            qvel_meas=qvel_meas,
            ctrl_target=ctrl_target,
            qpos_hist=qpos_hist,
            qvel_hist=qvel_hist,
            qact_hist=qact_hist,
            control_values=control_values,
            start_idx=start_idx,
            rollout_steps=rollout_steps,
        )

    def _run_index_mjx(
        self,
        *,
        qpos_meas: np.ndarray,
        qvel_meas: np.ndarray,
        ctrl_target: np.ndarray,
        qpos_hist: np.ndarray,
        qvel_hist: np.ndarray,
        qact_hist: np.ndarray,
        control_values: dict[str, np.ndarray],
        start_idx: int,
        rollout_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
        control_slice = slice(start_idx, start_idx + rollout_steps)
        print(f"[play] Running compiled MJX rollout for {rollout_steps} steps...")
        traj = self.predict_next(
            self.best_params,
            qpos_hist=jnp.asarray(qpos_hist),
            qvel_hist=jnp.asarray(qvel_hist),
            qact_hist=jnp.asarray(qact_hist),
            qact_ctrl=jnp.asarray(ctrl_target[control_slice, None], dtype=jnp.float64),
            control={
                name: jnp.asarray(values[control_slice, None], dtype=jnp.float64)
                for name, values in control_values.items()
            },
            time_index=jnp.arange(
                start_idx,
                start_idx + rollout_steps,
                dtype=jnp.int32,
            ),
        )
        sim_qpos = np.concatenate([[qpos_meas[start_idx]], np.asarray(traj)[:, 0]])
        print("[play] Rollout finished")
        return (
            sim_qpos,
            qpos_meas[start_idx:start_idx + rollout_steps + 1].copy(),
            qvel_meas[start_idx:start_idx + rollout_steps + 1].copy(),
            self.cfg.data.dt,
            self.model_xml,
        )

    def _run_index_mujoco_np(
        self,
        *,
        qpos_meas: np.ndarray,
        qvel_meas: np.ndarray,
        ctrl_target: np.ndarray,
        qpos_hist: np.ndarray,
        qvel_hist: np.ndarray,
        qact_hist: np.ndarray,
        control_values: dict[str, np.ndarray],
        start_idx: int,
        rollout_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
        assert self.numpy_action is not None
        data = mujoco.MjData(self.model)
        data.qpos[0] = qpos_meas[start_idx]
        data.qvel[0] = qvel_meas[start_idx]
        mujoco.mj_forward(self.model, data)

        sim_qpos = np.empty(rollout_steps + 1, dtype=np.float64)
        print(f"[play] Running native MuJoCo rollout for {rollout_steps} steps...")

        def append_history(hist: np.ndarray, value: np.ndarray) -> np.ndarray:
            return np.concatenate([hist[1:], value.reshape(1, -1)], axis=0)

        for out_i in range(rollout_steps + 1):
            data_i = start_idx + out_i
            sim_qpos[out_i] = data.qpos[0]
            if out_i == rollout_steps:
                break

            tau = np.asarray(self.numpy_action(data, control_values, data_i)).reshape(-1)
            data.ctrl[: self.model.nu] = tau[: self.model.nu]
            for _ in range(_VAL_SUBSTEPS_PER_SAMPLE):
                mujoco.mj_step(self.model, data)
            qpos_hist = append_history(qpos_hist, np.asarray(data.qpos, dtype=np.float64))
            qvel_hist = append_history(qvel_hist, np.asarray(data.qvel, dtype=np.float64))
            qact_hist = append_history(
                qact_hist,
                np.asarray([ctrl_target[data_i + 1]], dtype=np.float64),
            )

        print("[play] Rollout finished")
        return (
            sim_qpos,
            qpos_meas[start_idx:start_idx + rollout_steps + 1].copy(),
            qvel_meas[start_idx:start_idx + rollout_steps + 1].copy(),
            self.cfg.data.dt,
            self.model_xml,
        )


def _prepare_rollout(
    task: str,
    checkpoint: str,
    run_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path]:
    return PlaybackRunner(task, checkpoint).run_index(run_index)


def play(
    task: str,
    checkpoint: str,
    run_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run learned rollout for a task/checkpoint without visualization.

    Returns:
        generated_trajectory: simulated qpos trajectory from learned model
        ground_truth_trajectory: measured dataset qpos trajectory (same length)
    """
    sim_qpos, gt_qpos, _, _, _ = _prepare_rollout(
        task=task,
        checkpoint=checkpoint,
        run_index=run_index,
    )
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
        run_index=args.run_index,
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
