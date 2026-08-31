import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple
import mujoco.mjx as mjx
from jax import Array
from sysidmjx.models import ModelState


def generate_loss_train_functions(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    change_model: Callable[[Dict, mjx.Model], mjx.Model],
    make_action: Callable[[Dict, mjx.Data, ModelState], Array],
    w_qpos: float = 1.0,
    w_qvel: float = 0.0,
    qpos_scale: float = 1.0,
    qvel_scale: float = 1.0,
    huber_delta: float | None = None,
) -> Tuple[Callable, Callable, Callable]:
    """Generate JIT-compiled loss, predictor, and loss-component helpers.

    The resulting functions do not impose static-argnums restrictions
    (``params`` can be a ``dict`` PyTree) and handle both ``qact_ctrl``
    shapes ``(T,)`` and ``(T, nu)``.
    """

    # --------------------------------------------------
    # 1.  Simulation helper (predict_next)
    # --------------------------------------------------
    nq, nu = mjx_model.nq, mjx_model.nu  # DOF counts

    @jax.jit
    def _ensure_2d(vec: Array, width: int) -> Array:
        """Convert 1‑D control vector to (T, width) if required."""
        vec = jnp.asarray(vec)
        if vec.ndim == 1:
            return vec[:, None]  # (T,) → (T, 1)
        return vec

    @jax.jit
    def _pointwise_loss(error: Array) -> Array:
        if huber_delta is None:
            return jnp.square(error)
        delta = jnp.asarray(huber_delta, dtype=error.dtype)
        scaled = error / delta
        return jnp.square(delta) * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)

    @jax.jit
    def predict_next(
        params: Dict,
        qpos_hist: Array,
        qvel_hist: Array,
        qact_hist: Array,
        qact_ctrl: Array,  # (T,) or (T, nu)
        control: Dict[str, Array],
        time_index: Array,  # (T,) source-state time indices
    ) -> Array:
        """Roll out a trajectory and return stacked ``qpos‖qvel`` history."""

        qact_ctrl = _ensure_2d(qact_ctrl, nu)
        qpos0 = qpos_hist[-1]
        qvel0 = qvel_hist[-1]
        next_qact_ctrl = jnp.concatenate([qact_ctrl[1:], qact_ctrl[-1:]], axis=0)

        sim_model = change_model(params, mjx_model)
        data0 = mjx_data.replace(qpos=qpos0, qvel=qvel0)

        def append_history(hist: Array, value: Array) -> Array:
            return jnp.concatenate([hist[1:], value[None, :]], axis=0)

        def body(
            carry: Tuple[mjx.Data, Array, Array, Array],
            inputs: Tuple[Array, Array, Dict[str, Array], Array],
        ):
            carry_data, carry_qpos_hist, carry_qvel_hist, carry_qact_hist = carry
            qact_t, next_qact_t, control_t, t_idx = inputs
            # ``qact_t`` has shape (nu,). Guard against 0-D edge-cases.
            if qact_t.ndim == 0:
                qact_t = qact_t[None]
            if next_qact_t.ndim == 0:
                next_qact_t = next_qact_t[None]

            state = ModelState(
                qpos_hist=carry_qpos_hist,
                qvel_hist=carry_qvel_hist,
                qact_hist=carry_qact_hist,
                time_index=t_idx,
                control=control_t,
            )
            u = make_action(params, carry_data, state)
            d = carry_data.replace(ctrl=u)
            d = mjx.step(sim_model, d)
            next_carry = (
                d,
                append_history(carry_qpos_hist, d.qpos),
                append_history(carry_qvel_hist, d.qvel),
                append_history(carry_qact_hist, next_qact_t),
            )
            return next_carry, jnp.hstack([d.qpos, d.qvel])  # (nq+nv,)

        initial_carry = (data0, qpos_hist, qvel_hist, qact_hist)
        _, q_hist = jax.lax.scan(
            body,
            initial_carry,
            (qact_ctrl, next_qact_ctrl, control, time_index),
        )
        return q_hist  # (T, nq+nv)

    # --------------------------------------------------
    # 2.  Losses
    # --------------------------------------------------
    @jax.jit
    def single_loss(
        params: Dict,
        qpos_hist: Array,
        qvel_hist: Array,
        qact_hist: Array,
        qact_ctrl: Array,
        control: Dict[str, Array],
        qpos_target: Array,  # (T, nq) or (T,)
        qvel_target: Array,  # (T, nv) or (T,)
        time_index: Array,
    ) -> Tuple[Array, Array]:
        traj = predict_next(
            params,
            qpos_hist,
            qvel_hist,
            qact_hist,
            qact_ctrl,
            control,
            time_index,
        )
        qpos_pred = traj[:, :nq]
        qvel_pred = traj[:, nq:]
        qpos_target = _ensure_2d(qpos_target, nq)
        qvel_target = _ensure_2d(qvel_target, qvel_pred.shape[-1])
        qpos_error = (qpos_pred - qpos_target) / jnp.asarray(qpos_scale, qpos_pred.dtype)
        qvel_error = (qvel_pred - qvel_target) / jnp.asarray(qvel_scale, qvel_pred.dtype)
        loss_qpos = jnp.mean(_pointwise_loss(qpos_error))
        loss_qvel = jnp.mean(_pointwise_loss(qvel_error))
        return loss_qpos, loss_qvel

    loss_batch = jax.jit(
        jax.vmap(single_loss, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
    )

    @jax.jit
    def _loss_components(
        params,
        qpos_hist,
        qvel_hist,
        qact_hist,
        qact_ctrl,
        control,
        qpos_target,
        qvel_target,
        time_index,
    ):
        loss_qpos_b, loss_qvel_b = loss_batch(
            params,
            qpos_hist,
            qvel_hist,
            qact_hist,
            qact_ctrl,
            control,
            qpos_target,
            qvel_target,
            time_index,
        )
        loss_qpos = jnp.mean(loss_qpos_b)
        loss_qvel = jnp.mean(loss_qvel_b)
        total = w_qpos * loss_qpos + w_qvel * loss_qvel
        return total, loss_qpos, loss_qvel

    @jax.jit
    def total_loss(
        params,
        qpos_hist,
        qvel_hist,
        qact_hist,
        qact_ctrl,
        control,
        qpos_target,
        qvel_target,
        time_index,
    ):
        total, _, _ = _loss_components(
            params,
            qpos_hist,
            qvel_hist,
            qact_hist,
            qact_ctrl,
            control,
            qpos_target,
            qvel_target,
            time_index,
        )
        return total

    return total_loss, predict_next, _loss_components
