from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from flax.training import train_state
from mujoco import mjx

from config.base import RunConfig
from sysidmjx.assets.single_motor.dataloader.single_joint import data_load
from sysidmjx.core import generate_loss_train_functions, get_batch
from sysidmjx.models import compose_action_model


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).resolve()


def _validate_paths(cfg: RunConfig) -> tuple[Path, Path]:
    dataset_path = _resolve_path(cfg.data.dataset_path)
    model_xml_path = _resolve_path(cfg.sim.model_xml)

    missing = []
    if not dataset_path.exists():
        missing.append(f"dataset path: {dataset_path}")
    if not model_xml_path.exists():
        missing.append(f"model xml: {model_xml_path}")
    if missing:
        raise FileNotFoundError("Missing required input files:\n- " + "\n- ".join(missing))

    return dataset_path, model_xml_path


def _apply_jax_config_defaults() -> None:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")


def _tree_l2_norm(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    sq_sum = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq_sum))


def _params_scalar_entries(params: dict[str, Any], prefix: str = "params") -> list[tuple[str, float]]:
    """Flatten params tree to (key_path, scalar_value) for scalars or size-1 arrays only."""
    out: list[tuple[str, float]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}/{k}" if path else k)
        elif hasattr(obj, "shape") and hasattr(obj, "size"):
            arr = np.asarray(obj)
            if arr.size == 1:
                out.append((path, float(arr.reshape(-1).item())))
            # size > 1: skip (e.g. weight matrices), so we only log small param sets
        elif isinstance(obj, (int, float)):
            out.append((path, float(obj)))

    visit(params, prefix)
    return out


_MAX_PARAMS_FOR_TENSORBOARD = 8


def _build_component_factory(component_cfg: Any, dataset: dict[str, Any]):
    params = dict(component_cfg.params)
    init_kind = params.pop("initialization", None)
    if init_kind is not None:
        if init_kind != "zeros_by_dataset":
            raise ValueError(f"Unknown component initialization '{init_kind}'.")
        torque_dim = int(params.pop("torque_dim", dataset["qpos"].shape[-1]))
        params["optimal_torques"] = jnp.zeros(
            (dataset["qpos"].shape[0], torque_dim),
            dtype=jnp.float64,
        )
    return component_cfg.func(**params)


def run_single_motor_torque_opt(
    cfg: RunConfig,
    run_dir: Path,
    logger: Any,
    task_name: str,
) -> dict[str, Any]:
    _apply_jax_config_defaults()

    dataset_path, model_xml_path = _validate_paths(cfg)

    logger.add_text("paths/model_xml", str(model_xml_path), 0)
    logger.add_text("paths/dataset_path", str(dataset_path), 0)

    dataset, _ = data_load(
        data_path=str(dataset_path),
        motor_id=cfg.data.motor_id,
        num_lags=cfg.data.num_lags,
        dtype=jnp.float64,
    )

    logger.add_scalar("data/num_samples", float(dataset["qpos"].shape[0]), 0)
    logger.add_scalar("data/num_lags", float(cfg.data.num_lags), 0)

    mj_model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    mj_model.opt.timestep = cfg.data.dt
    mj_model.opt.integrator = cfg.sim.integrator
    mj_model.opt.iterations = int(cfg.sim.iterations)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    controller_factories = [
        _build_component_factory(component, dataset)
        for component in cfg.experiment.controllers
    ]
    friction_factories = [
        _build_component_factory(component, dataset)
        for component in cfg.experiment.frictions
    ]
    changer_factories = [
        _build_component_factory(component, dataset)
        for component in cfg.experiment.changers
    ]

    make_action, change_model, init_params = compose_action_model(
        controllers=controller_factories,
        frictions=friction_factories,
        changers=changer_factories,
        seed=jax.random.PRNGKey(cfg.train.seed),
    )

    total_loss, train_step, _ = generate_loss_train_functions(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        change_model=change_model,
        make_action=make_action,
        w_qpos=cfg.train.loss_w_qpos,
        w_qvel=cfg.train.loss_w_qvel,
    )

    state = train_state.TrainState.create(
        apply_fn=None,
        params=init_params,
        tx=optax.adam(cfg.train.learning_rate),
    )

    indices = jnp.arange(dataset["qpos"].shape[0])
    rng = jax.random.PRNGKey(cfg.train.seed)
    patience = cfg.train.early_stop_patience
    min_delta = cfg.train.early_stop_min_delta

    best_loss = float("inf")
    best_params = state.params
    no_improve = 0
    loss_hist: list[float] = []
    grad_norm_hist: list[float] = []
    last_epoch = -1
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Starting optimization | "
        f"task={task_name} samples={dataset['qpos'].shape[0]} "
        f"max_iter={cfg.train.max_iter} batch_size={cfg.train.batch_size} "
        f"lr={cfg.train.learning_rate} log_every={cfg.logging.log_every} "
        f"checkpoint_every={cfg.logging.checkpoint_every}"
    )

    for epoch in range(cfg.train.max_iter):
        last_epoch = epoch
        loss = total_loss(
            state.params,
            qpos=dataset["qpos"],
            qvel=dataset["qvel"],
            ctrl_vec=dataset["qact"],
            qpos_des=dataset["qpos_next"],
            qvel_des=dataset["qvel_next"],
            time_index=dataset["time_index"],
        )
        loss_val = float(loss)
        loss_hist.append(loss_val)

        if epoch % cfg.logging.log_every == 0:
            logger.add_scalar("train/loss", loss_val, epoch)
            param_entries = _params_scalar_entries(state.params, prefix="params")
            if len(param_entries) <= _MAX_PARAMS_FOR_TENSORBOARD:
                for key_path, value in param_entries:
                    logger.add_scalar(key_path, value, epoch)

        improved = loss_val < (best_loss - min_delta)
        if improved:
            best_loss = loss_val
            best_params = state.params
            no_improve = 0
        else:
            no_improve += 1

        if epoch % cfg.logging.checkpoint_every == 0:
            with (checkpoints_dir / f"epoch_{epoch:06d}.pkl").open("wb") as file:
                pickle.dump(state.params, file)
            print(f"[epoch {epoch:06d}] checkpoint saved")

        if patience is not None and no_improve >= patience:
            print(
                f"Early stop at epoch={epoch}: no_improve={no_improve} "
                f"(patience={patience}, min_delta={min_delta}) best_loss={best_loss:.6f}"
            )
            break

        rng, batch_seed = jax.random.split(rng)
        batch, indices = get_batch(
            dataset,
            batch_seed,
            indices,
            cfg.train.batch_size,
        )
        state, metrics = train_step(
            state,
            qpos=batch["qpos"],
            qvel=batch["qvel"],
            ctrl_vec=batch["qact"],
            qpos_des=batch["qpos_next"],
            qvel_des=batch["qvel_next"],
            time_index=batch["time_index"],
        )

        grad_norm = _tree_l2_norm(metrics["grads"])
        grad_norm_hist.append(grad_norm)
        if epoch % cfg.logging.log_every == 0:
            logger.add_scalar("train/grad_norm", grad_norm, epoch)
            logger.add_scalar("train/loss_qpos", float(metrics["loss_qpos"]), epoch)
            logger.add_scalar("train/loss_qvel", float(metrics["loss_qvel"]), epoch)
            print(
                f"[epoch {epoch:06d}] loss={loss_val:.6f} best={best_loss:.6f} "
                f"grad_norm={grad_norm:.6f} no_improve={no_improve}/{patience}"
            )

    np.save(run_dir / "loss_history.npy", np.asarray(loss_hist, dtype=np.float64))
    np.save(run_dir / "grad_norm_history.npy", np.asarray(grad_norm_hist, dtype=np.float64))
    with (run_dir / "best_params.pkl").open("wb") as file:
        pickle.dump(best_params, file)
    with (run_dir / "final_params.pkl").open("wb") as file:
        pickle.dump(state.params, file)
    final_loss_val = loss_hist[-1] if loss_hist else float("nan")
    print(
        "Optimization done | "
        f"epochs_ran={len(loss_hist)} best_loss={best_loss:.6f} "
        f"final_loss={final_loss_val:.6f}"
    )

    return {
        "status": "ok",
        "runner": getattr(cfg.runner, "__name__", str(cfg.runner)),
        "samples": int(dataset["qpos"].shape[0]),
        "last_epoch": last_epoch,
        "epochs_ran": len(loss_hist),
        "best_loss": best_loss if loss_hist else None,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "final_grad_norm": grad_norm_hist[-1] if grad_norm_hist else None,
    }


# Backward-compatible alias for old imports
_run_single_motor_torque_opt = run_single_motor_torque_opt
