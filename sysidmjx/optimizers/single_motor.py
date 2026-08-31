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
from sysidmjx.assets.single_motor.dataloader.single_joint import SingleMotorDataLoader
from sysidmjx.core import generate_loss_train_functions
from sysidmjx.models import compose_action_model


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).resolve()


def _validate_paths(cfg: RunConfig) -> Path:
    model_xml_path = _resolve_path(cfg.data.dataset.model_xml)
    missing = []
    if not model_xml_path.exists():
        missing.append(f"model xml: {model_xml_path}")
    if missing:
        raise FileNotFoundError("Missing required input files:\n- " + "\n- ".join(missing))
    return model_xml_path


def _apply_jax_config_defaults() -> None:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")


def _tree_l2_norm_array(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    sq_sum = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return jnp.sqrt(sq_sum)


def _learning_rate(cfg: RunConfig):
    lr0 = cfg.train.learning_rate
    lr_final = cfg.train.learning_rate_final
    if lr_final is None:
        return optax.constant_schedule(lr0)
    return optax.exponential_decay(
        init_value=lr0,
        transition_steps=cfg.train.max_iter,
        decay_rate=lr_final / lr0,
        end_value=lr_final,
    )


_MAX_TENSORBOARD_PARAM_LEAF_SIZE = 16


def _params_log_entries(params: dict[str, Any], prefix: str = "params") -> list[tuple[str, float]]:
    """Flatten scalar and small-array params to TensorBoard scalar entries."""
    out: list[tuple[str, float]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}/{k}" if path else k)
        elif hasattr(obj, "shape") and hasattr(obj, "size"):
            arr = np.asarray(obj)
            if arr.size == 1:
                out.append((path, float(arr.reshape(-1).item())))
            elif arr.size <= _MAX_TENSORBOARD_PARAM_LEAF_SIZE:
                out.extend(
                    (f"{path}/{i}", float(value))
                    for i, value in enumerate(arr.reshape(-1))
                )
        elif isinstance(obj, (int, float)):
            out.append((path, float(obj)))

    visit(params, prefix)
    return out


def _build_component_factory(component_cfg: Any, dataset: dict[str, Any]):
    params = dict(component_cfg.params)
    checkpoint = params.pop("checkpoint", None)
    init_kind = params.pop("initialization", None)
    if init_kind is not None:
        if init_kind != "zeros_by_dataset":
            raise ValueError(f"Unknown component initialization '{init_kind}'.")
        torque_dim = int(params.pop("torque_dim", dataset["qpos_hist"].shape[-1]))
        n_torques = int(jnp.max(dataset["time_index"])) + 1
        params["optimal_torques"] = jnp.zeros(
            (n_torques, torque_dim),
            dtype=jnp.float64,
        )
    factory = component_cfg.func(**params)
    if checkpoint is None:
        return factory

    def factory_from_checkpoint(seed: jax.Array):
        make_model, init_params = factory(seed)
        del init_params
        checkpoint_path = _resolve_path(str(checkpoint))
        with checkpoint_path.open("rb") as file:
            checkpoint_params = pickle.load(file)

        if "models" in checkpoint_params:
            model_params = checkpoint_params["models"]
        elif "controllers" in checkpoint_params:
            model_params = checkpoint_params["controllers"]
        else:
            loaded_params = checkpoint_params
            model_params = None

        if model_params is not None:
            if len(model_params) != 1:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} must contain exactly one model "
                    "when used as a component checkpoint."
                )
            loaded_params = next(iter(model_params.values()))

        @jax.jit
        def frozen_model(loaded_model_params, data, state):
            frozen_params = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                loaded_model_params,
            )
            return make_model(frozen_params, data, state)

        return frozen_model, loaded_params

    return factory_from_checkpoint


def run_single_motor_torque_opt(
    cfg: RunConfig,
    run_dir: Path,
    logger: Any,
    task_name: str,
) -> dict[str, Any]:
    _apply_jax_config_defaults()

    model_xml_path = _validate_paths(cfg)

    logger.add_text("paths/model_xml", str(model_xml_path), 0)
    loader = SingleMotorDataLoader.from_spec(
        spec=cfg.data.dataset,
        history_len=cfg.data.history_len,
        prediction_horizon=cfg.data.prediction_horizon,
        val_prediction_horizon=cfg.data.val_prediction_horizon,
        dtype=jnp.float64,
        repo_root=Path.cwd(),
    )
    dataset = loader.full

    logger.add_scalar("data/num_samples", float(dataset["qpos_hist"].shape[0]), 0)
    logger.add_scalar("data/history_len", float(dataset["qpos_hist"].shape[1]), 0)
    logger.add_scalar("data/prediction_horizon", float(dataset["qpos_next"].shape[1]), 0)
    logger.add_scalar(
        "data/val_prediction_horizon",
        float(loader.val.data["qpos_next"].shape[1]),
        0,
    )

    model_factories = [
        _build_component_factory(component, dataset)
        for component in cfg.experiment.models
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
        models=model_factories,
        frictions=friction_factories,
        changers=changer_factories,
        seed=jax.random.PRNGKey(cfg.train.seed),
    )

    mj_model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    mj_model.opt.timestep = cfg.data.dt
    mj_model.opt.integrator = cfg.sim.integrator
    mj_model.opt.iterations = int(cfg.sim.iterations)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    total_loss, _, loss_components = generate_loss_train_functions(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        change_model=change_model,
        make_action=make_action,
        w_qpos=cfg.train.loss_w_qpos,
        w_qvel=cfg.train.loss_w_qvel,
        qpos_scale=cfg.train.loss_qpos_scale,
        qvel_scale=cfg.train.loss_qvel_scale,
        huber_delta=cfg.train.loss_huber_delta,
    )

    learning_rate = _learning_rate(cfg)
    state = train_state.TrainState.create(
        apply_fn=None,
        params=init_params,
        tx=optax.adam(learning_rate),
    )

    def loss_for_batch(params, batch: dict[str, Any]):
        return total_loss(
            params,
            qpos_hist=batch["qpos_hist"],
            qvel_hist=batch["qvel_hist"],
            qact_hist=batch["qact_hist"],
            qact_ctrl=batch["qact_ctrl"],
            control=batch["control"],
            qpos_target=batch["qpos_next"],
            qvel_target=batch["qvel_next"],
            time_index=batch["time_index"],
        )

    def loss_components_for_batch(params, batch: dict[str, Any]):
        return loss_components(
            params,
            qpos_hist=batch["qpos_hist"],
            qvel_hist=batch["qvel_hist"],
            qact_hist=batch["qact_hist"],
            qact_ctrl=batch["qact_ctrl"],
            control=batch["control"],
            qpos_target=batch["qpos_next"],
            qvel_target=batch["qvel_next"],
            time_index=batch["time_index"],
        )

    def log_validation(params, step: int) -> float:
        total, qpos, qvel = loss_components_for_batch(
            params,
            loader.val.data,
        )
        logger.add_scalar("val/loss", float(total), step)
        logger.add_scalar("val/loss_qpos", float(qpos), step)
        logger.add_scalar("val/loss_qvel", float(qvel), step)
        return float(total)

    value_and_grad = jax.value_and_grad(loss_for_batch, argnums=0)

    @jax.jit
    def train_step(model_state, batch: dict[str, Any]):
        loss_batch, grads = value_and_grad(model_state.params, batch)
        new_state = model_state.apply_gradients(grads=grads)
        return new_state, loss_batch, _tree_l2_norm_array(grads)

    rng = jax.random.PRNGKey(cfg.train.seed)
    patience = cfg.train.early_stop_patience
    min_delta = cfg.train.early_stop_min_delta

    best_loss = float("inf")
    best_params = state.params
    no_improve = 0
    loss_hist: list[float] = []
    grad_norm_hist: list[float] = []
    last_logged_loss = float("nan")
    last_logged_grad_norm = float("nan")
    last_epoch = -1
    best_path = run_dir / "best_params.pkl"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_params(path: Path, params: Any) -> None:
        with path.open("wb") as file:
            pickle.dump(params, file)

    print(
        "Starting optimization | "
        f"task={task_name} samples={dataset['qpos_hist'].shape[0]} "
        f"max_iter={cfg.train.max_iter} batch_size={cfg.train.batch_size} "
        f"lr={cfg.train.learning_rate} log_every={cfg.logging.log_every} "
        f"checkpoint_every={cfg.logging.checkpoint_every}"
    )

    for epoch in range(cfg.train.max_iter):
        last_epoch = epoch
        rng, batch_seed = jax.random.split(rng)
        batch = loader.sample_train_batch(batch_seed, cfg.train.batch_size)
        state, loss_batch, grad_norm = train_step(state, batch)

        should_log = epoch % cfg.logging.log_every == 0
        if should_log:
            loss, loss_qpos, loss_qvel = loss_components_for_batch(
                state.params,
                loader.train.data,
            )
            loss_val = float(loss)
            grad_norm_val = float(grad_norm)
            batch_loss_val = float(loss_batch)
            val_loss = log_validation(state.params, epoch)

            loss_hist.append(loss_val)
            grad_norm_hist.append(grad_norm_val)
            last_logged_loss = loss_val
            last_logged_grad_norm = grad_norm_val

            logger.add_scalar("train/loss", loss_val, epoch)
            logger.add_scalar("train/loss_qpos", float(loss_qpos), epoch)
            logger.add_scalar("train/loss_qvel", float(loss_qvel), epoch)
            logger.add_scalar("train/grad_norm", grad_norm_val, epoch)
            logger.add_scalar("train/batch_loss", batch_loss_val, epoch)
            logger.add_scalar("train/learning_rate", float(learning_rate(epoch)), epoch)
            for key_path, value in _params_log_entries(state.params, prefix="params"):
                logger.add_scalar(key_path, value, epoch)

            improved = val_loss < (best_loss - min_delta)
            if improved:
                best_loss = val_loss
                best_params = state.params
                save_params(best_path, best_params)
                no_improve = 0
            else:
                no_improve += cfg.logging.log_every

            print(
                f"[epoch {epoch:06d}] loss={loss_val:.6f} best_val={best_loss:.6f} "
                f"val={val_loss:.6f} grad_norm={grad_norm_val:.6f} "
                f"batch={batch_loss_val:.6f} no_improve={no_improve}/{patience}"
            )

        if epoch % cfg.logging.checkpoint_every == 0:
            save_params(checkpoints_dir / f"epoch_{epoch:06d}.pkl", state.params)
            print(f"[epoch {epoch:06d}] checkpoint saved")

        if patience is not None and no_improve >= patience:
            print(
                f"Early stop at epoch={epoch}: no_improve={no_improve} "
                f"(patience={patience}, min_delta={min_delta}) best_val={best_loss:.6f}"
            )
            break

    np.save(run_dir / "loss_history.npy", np.asarray(loss_hist, dtype=np.float64))
    np.save(run_dir / "grad_norm_history.npy", np.asarray(grad_norm_hist, dtype=np.float64))
    save_params(best_path, best_params)
    final_loss_val = last_logged_loss
    print(
        "Optimization done | "
        f"epochs_ran={last_epoch + 1} best_val={best_loss:.6f} "
        f"final_loss={final_loss_val:.6f}"
    )

    return {
        "status": "ok",
        "runner": getattr(cfg.runner, "__name__", str(cfg.runner)),
        "samples": int(dataset["qpos_hist"].shape[0]),
        "last_epoch": last_epoch,
        "epochs_ran": last_epoch + 1,
        "best_loss": best_loss if loss_hist else None,
        "final_loss": last_logged_loss if loss_hist else None,
        "final_grad_norm": last_logged_grad_norm if grad_norm_hist else None,
    }
