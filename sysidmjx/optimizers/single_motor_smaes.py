from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.flatten_util import ravel_pytree
from mujoco import mjx

from cmaes import CMA

from config.base import RunConfig
from sysidmjx.assets.single_motor.dataloader.single_joint import data_load
from sysidmjx.core import generate_loss_train_functions
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


def _params_scalar_entries(params: dict[str, Any], prefix: str = "params") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}/{k}" if path else k)
        elif hasattr(obj, "shape") and hasattr(obj, "size"):
            arr = np.asarray(obj)
            if arr.size == 1:
                out.append((path, float(arr.reshape(-1).item())))
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


def run_single_motor_torque_smaes_opt(
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

    total_loss, _, _ = generate_loss_train_functions(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        change_model=change_model,
        make_action=make_action,
        w_qpos=cfg.train.loss_w_qpos,
        w_qvel=cfg.train.loss_w_qvel,
    )

    init_vec, unravel = ravel_pytree(init_params)
    init_vec_np = np.asarray(init_vec, dtype=np.float64)

    sigma = float(getattr(cfg.train, "smaes_sigma", 0.05))
    population_size = int(getattr(cfg.train, "smaes_population_size", 32))
    lower_bound = float(getattr(cfg.train, "smaes_lower_bound", -5.0))
    upper_bound = float(getattr(cfg.train, "smaes_upper_bound", 5.0))

    bounds = np.repeat(
        np.asarray([[lower_bound, upper_bound]], dtype=np.float64),
        init_vec_np.size,
        axis=0,
    )

    optimizer = CMA(
        mean=init_vec_np,
        sigma=sigma,
        bounds=bounds,
        population_size=population_size,
        seed=int(cfg.train.seed),
    )

    patience = getattr(cfg.train, "early_stop_patience", None)
    min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0))

    best_loss = float("inf")
    best_params = init_params
    final_params = init_params
    no_improve = 0
    best_loss_hist: list[float] = []
    pop_mean_loss_hist: list[float] = []
    pop_std_loss_hist: list[float] = []
    last_generation = -1

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Starting SMA-ES optimization | "
        f"task={task_name} samples={dataset['qpos'].shape[0]} "
        f"max_gen={cfg.train.max_iter} pop_size={population_size} sigma={sigma} "
        f"log_every={cfg.logging.log_every} checkpoint_every={cfg.logging.checkpoint_every}"
    )

    def eval_loss(candidate_vec_np: np.ndarray) -> float:
        candidate_params = unravel(jnp.asarray(candidate_vec_np, dtype=jnp.float64))
        loss = total_loss(
            candidate_params,
            qpos=dataset["qpos"],
            qvel=dataset["qvel"],
            ctrl_vec=dataset["qact"],
            qpos_des=dataset["qpos_next"],
            qvel_des=dataset["qvel_next"],
            time_index=dataset["time_index"],
        )
        return float(loss)

    for generation in range(cfg.train.max_iter):
        last_generation = generation
        solutions: list[tuple[np.ndarray, float]] = []
        generation_losses: list[float] = []

        for _ in range(population_size):
            candidate = optimizer.ask()
            candidate_loss = eval_loss(candidate)
            solutions.append((candidate, candidate_loss))
            generation_losses.append(candidate_loss)

            if candidate_loss < (best_loss - min_delta):
                best_loss = candidate_loss
                best_params = unravel(jnp.asarray(candidate, dtype=jnp.float64))
                no_improve = 0

        if not generation_losses:
            break

        optimizer.tell(solutions)
        best_loss_hist.append(best_loss)

        pop_mean = float(np.mean(generation_losses))
        pop_std = float(np.std(generation_losses))
        pop_mean_loss_hist.append(pop_mean)
        pop_std_loss_hist.append(pop_std)

        final_params = unravel(jnp.asarray(optimizer.mean, dtype=jnp.float64))
        no_improve += 1

        if generation % cfg.logging.log_every == 0:
            logger.add_scalar("train/best_loss", best_loss, generation)
            logger.add_scalar("train/pop_mean_loss", pop_mean, generation)
            logger.add_scalar("train/pop_std_loss", pop_std, generation)
            param_entries = _params_scalar_entries(best_params, prefix="params")
            if len(param_entries) <= _MAX_PARAMS_FOR_TENSORBOARD:
                for key_path, value in param_entries:
                    logger.add_scalar(key_path, value, generation)
            print(
                f"[gen {generation:06d}] best={best_loss:.6f} "
                f"pop_mean={pop_mean:.6f} pop_std={pop_std:.6f} "
                f"no_improve={no_improve}/{patience}"
            )

        if generation % cfg.logging.checkpoint_every == 0:
            with (checkpoints_dir / f"gen_{generation:06d}.pkl").open("wb") as file:
                pickle.dump(best_params, file)
            print(f"[gen {generation:06d}] checkpoint saved")

        if optimizer.should_stop():
            print(f"SMA-ES internal stopping criterion met at generation={generation}.")
            break

        if patience is not None and no_improve >= patience:
            print(
                f"Early stop at generation={generation}: no_improve={no_improve} "
                f"(patience={patience}, min_delta={min_delta}) best_loss={best_loss:.6f}"
            )
            break

    np.save(run_dir / "best_loss_history.npy", np.asarray(best_loss_hist, dtype=np.float64))
    np.save(run_dir / "pop_mean_loss_history.npy", np.asarray(pop_mean_loss_hist, dtype=np.float64))
    np.save(run_dir / "pop_std_loss_history.npy", np.asarray(pop_std_loss_hist, dtype=np.float64))

    with (run_dir / "best_params.pkl").open("wb") as file:
        pickle.dump(best_params, file)
    with (run_dir / "final_params.pkl").open("wb") as file:
        pickle.dump(final_params, file)

    final_best = best_loss_hist[-1] if best_loss_hist else float("nan")
    print(
        "Optimization done | "
        f"gens_ran={len(best_loss_hist)} best_loss={final_best:.6f}"
    )

    return {
        "status": "ok",
        "runner": getattr(cfg.runner, "__name__", str(cfg.runner)),
        "samples": int(dataset["qpos"].shape[0]),
        "last_generation": last_generation,
        "generations_ran": len(best_loss_hist),
        "best_loss": best_loss if best_loss_hist else None,
        "final_loss": pop_mean_loss_hist[-1] if pop_mean_loss_hist else None,
        "final_pop_std_loss": pop_std_loss_hist[-1] if pop_std_loss_hist else None,
    }


# Backward-compatible alias for old imports
_run_single_motor_torque_smaes_opt = run_single_motor_torque_smaes_opt
