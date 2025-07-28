import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple
import mujoco.mjx as mjx
from jax import Array


def generate_loss_train_functions(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    change_model: Callable[[Dict, mjx.Model], mjx.Model],
    make_action: Callable[[Dict, mjx.Data, Array], Array],
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
    ) -> Array:
        """Roll out a trajectory and return stacked ``qpos‖qvel`` history."""

        # Ensure ctrl_vec is (T, nu)
        ctrl_vec = _ensure_2d(ctrl_vec, nu)

        sim_model = change_model(params, mjx_model)
        data0 = mjx_data.replace(qpos=qpos0, qvel=qvel0)

        def body(carry: mjx.Data, ctrl_t: Array):
            # ``ctrl_t`` has shape (nu,). Guard against 0‑D edge‑case.
            if ctrl_t.ndim == 0:
                ctrl_t = ctrl_t[None]

            u = make_action(params, carry, ctrl_t)
            d = carry.replace(ctrl=u)
            d = mjx.step(sim_model, d)
            return d, jnp.hstack([d.qpos, d.qvel])  # (nq+nv,)

        _, q_hist = jax.lax.scan(body, data0, ctrl_vec)
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
    ) -> Array:
        traj = predict_next(params, qpos0, qvel0, ctrl_vec)
        qpos_hist = traj[:, :nq]
        qpos_des = _ensure_2d(qpos_des, nq)  # (T, nq)
        return jnp.mean(jnp.square(qpos_hist - qpos_des))

    loss_batch = jax.jit(jax.vmap(single_loss, in_axes=(None, 0, 0, 0, 0)))

    @jax.jit
    def total_loss(params, qpos, qvel, ctrl_vec, qpos_des):
        return jnp.mean(loss_batch(params, qpos, qvel, ctrl_vec, qpos_des))

    # --------------------------------------------------
    # 3.  Optimisation step
    # --------------------------------------------------
    value_and_grad = jax.jit(jax.value_and_grad(total_loss, argnums=0))

    @jax.jit
    def train_step(model_state, qpos, qvel, ctrl_vec, qpos_des):
        loss_val, grads = value_and_grad(model_state.params, qpos, qvel, ctrl_vec, qpos_des)
        new_state = model_state.apply_gradients(grads=grads)
        return new_state, {"loss": loss_val, "grads": grads}

    return total_loss, train_step, predict_next


def get_batch(
    dataset: Dict[str, Array], seed: jax.random.PRNGKey, indxs: Array, batch_size: int
):
    """
    Selects a random batch of data from the dataset.

    Parameters:
        dataset (Dict[str, Array]): The dataset containing multiple data arrays.
        seed (jax.random.PRNGKey): Random seed for shuffling.
        indxs (Array): Indices of available data.
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: A batch of data and shuffled indices.
    """
    indxs = jax.random.permutation(seed, indxs)
    batch = {key: value[indxs[:batch_size], :] for key, value in dataset.items()}
    return batch, indxs
