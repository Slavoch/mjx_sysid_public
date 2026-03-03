import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple
import mujoco.mjx as mjx
from jax import Array


def generate_loss_train_functions(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    change_model: Callable[[Dict, mjx.Model], mjx.Model],
    make_action: Callable[[Dict, mjx.Data, Array, Array], Array],
    w_qpos: float = 1.0,
    w_qvel: float = 0.0,
) -> Tuple[Callable, Callable, Callable]:
    """Generate JIT‑compiled helpers (loss, train step, predictor).

    The resulting functions **do not** impose any static‑argnums restrictions
    (``params`` can be a ``dict`` PyTree) and gracefully handle both
    ``ctrl_vec`` shapes ``(T,)`` *or* ``(T, nu)``.
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
    def predict_next(
        params: Dict,
        qpos0: Array,
        qvel0: Array,
        ctrl_vec: Array,  # (T,) or (T, nu)
        time_index: Array,  # (T,) absolute times OR scalar start time
    ) -> Array:
        """Roll out a trajectory and return stacked ``qpos‖qvel`` history."""

        # Ensure ctrl_vec is (T, nu)
        ctrl_vec = _ensure_2d(ctrl_vec, nu)

        # Pass absolute per-step time into the controller.
        # time_index = jnp.asarray(time_index).astype(jnp.int32)

        sim_model = change_model(params, mjx_model)
        data0 = mjx_data.replace(qpos=qpos0, qvel=qvel0)

        def body(carry: mjx.Data, inputs: Tuple[Array, Array]):
            ctrl_t, t_idx = inputs
            # ``ctrl_t`` has shape (nu,). Guard against 0‑D edge‑case.
            if ctrl_t.ndim == 0:
                ctrl_t = ctrl_t[None]

            u = make_action(params, carry, ctrl_t, t_idx)
            d = carry.replace(ctrl=u)
            d = mjx.step(sim_model, d)
            return d, jnp.hstack([d.qpos, d.qvel])  # (nq+nv,)

        _, q_hist = jax.lax.scan(body, data0, (ctrl_vec, time_index))
        return q_hist  # (T, nq+nv)

    # --------------------------------------------------
    # 2.  Losses
    # --------------------------------------------------
    @jax.jit
    def single_loss(
        params: Dict,
        qpos0: Array,
        qvel0: Array,
        ctrl_vec: Array,
        qpos_des: Array,  # (T, nq) or (T,)
        qvel_des: Array,  # (T, nv) or (T,)
        time_index: Array,  # Unused - timestep indices generated internally
    ) -> Tuple[Array, Array]:
        traj = predict_next(params, qpos0, qvel0, ctrl_vec, time_index)
        qpos_hist = traj[:, :nq]
        qvel_hist = traj[:, nq:]
        qpos_des = _ensure_2d(qpos_des, nq)  # (T, nq)
        qvel_des = _ensure_2d(qvel_des, qvel_hist.shape[-1])  # (T, nv)
        loss_qpos = jnp.mean(jnp.square(qpos_hist - qpos_des))
        loss_qvel = jnp.mean(jnp.square(qvel_hist - qvel_des))
        return loss_qpos, loss_qvel

    loss_batch = jax.jit(
        jax.vmap(single_loss, in_axes=(None, 0, 0, 0, 0, 0, 0))
    )

    @jax.jit
    def _loss_components(params, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index):
        loss_qpos_b, loss_qvel_b = loss_batch(
            params,
            qpos,
            qvel,
            ctrl_vec,
            qpos_des,
            qvel_des,
            time_index,
        )
        loss_qpos = jnp.mean(loss_qpos_b)
        loss_qvel = jnp.mean(loss_qvel_b)
        total = w_qpos * loss_qpos + w_qvel * loss_qvel
        return total, loss_qpos, loss_qvel

    @jax.jit
    def total_loss(params, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index):
        total, _, _ = _loss_components(
            params, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index
        )
        return total

    # --------------------------------------------------
    # 3.  Optimisation step
    # --------------------------------------------------
    value_and_grad = jax.jit(jax.value_and_grad(total_loss, argnums=0))

    @jax.jit
    def train_step(model_state, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index):
        loss_val, grads = value_and_grad(
            model_state.params, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index
        )
        _, loss_qpos, loss_qvel = _loss_components(
            model_state.params, qpos, qvel, ctrl_vec, qpos_des, qvel_des, time_index
        )
        new_state = model_state.apply_gradients(grads=grads)
        return new_state, {
            "loss": loss_val,
            "loss_qpos": loss_qpos,
            "loss_qvel": loss_qvel,
            "grads": grads,
        }

    return total_loss, train_step, predict_next


def get_batch(
    dataset: Dict[str, Array],
    seed: jax.random.PRNGKey,
    indxs: Array,
    batch_size: int,
):
    """
    Selects a random batch of data from the dataset.

    Parameters:
        dataset (Dict[str, Array]): The dataset containing multiple data
            arrays.
        seed (jax.random.PRNGKey): Random seed for shuffling.
        indxs (Array): Indices of available data.
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: A batch of data and shuffled indices.
    """
    indxs = jax.random.permutation(seed, indxs)
    batch_indices = indxs[:batch_size]
    batch = {}
    for key, value in dataset.items():
        # Index first dimension, preserving all other dimensions
        batch[key] = value[batch_indices]
    return batch, indxs
