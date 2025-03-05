import jax
import mujoco
from mujoco import mjx
from typing import Callable, Dict
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jp


def generate_loss_train_functions(
    mjx_model: mujoco.mjx.Model,
    mjx_data: mujoco.mjx.Data,
    change_model: Callable[[Dict, mjx.Model], mjx.Model],
    make_action: Callable[[Dict, mjx.Data, ArrayImpl], ArrayImpl],
):
    """
    Generates functions for loss computation, training, and gradient updates for a MuJoCo-JAX model.

    Parameters:
        mjx_model (mujoco.mjx.Model): The base MuJoCo model used for simulation.
        mjx_data (mujoco.mjx.Data): The associated data structure containing simulation state.
        change_model (Callable[[Dict, mjx.Model], mjx.Model]): 
            A function that modifies the model parameters based on input parameters.
        make_action (Callable[[Dict, mjx.Data, ArrayImpl], ArrayImpl]): 
            A function that computes the control input for the simulation.

    Returns:
        tuple: A tuple containing:
            - total_loss (Callable): Function that computes the total loss for a batch.
            - train_step (Callable): Function that performs a single training step.
    """

    # JIT-compiled function for advancing the simulation step
    sim_step = jax.jit(mjx.step)

    @jax.jit
    def predict_next(params: Dict, qpos: ArrayImpl, qvel: ArrayImpl, ctrl_vec: ArrayImpl) -> ArrayImpl:
        """
        Predicts the next states of the system given initial conditions and control inputs.

        Parameters:
            params (Dict): Model parameters to be updated.
            qpos (ArrayImpl): Initial position values.
            qvel (ArrayImpl): Initial velocity values.
            ctrl_vec (ArrayImpl): Control inputs over time.

        Returns:
            ArrayImpl: History of position and velocity over time.
        """
        model = change_model(params, mjx_model)
        data = mjx_data.tree_replace({"qpos": qpos, "qvel": qvel})

        def step(carry: mjx.Data, ctrl: ArrayImpl):
            """
            Performs a single simulation step by applying control inputs.

            Parameters:
                carry (mjx.Data): Current simulation state.
                ctrl (ArrayImpl): Control input at the current step.

            Returns:
                tuple: Updated simulation state and concatenated qpos-qvel history.
            """
            data = carry.tree_replace({"ctrl": make_action(params, carry, ctrl)})
            data = sim_step(model, data)
            return data, jp.hstack([data.qpos, data.qvel])

        final_data, qpos_qvel_history = jax.lax.scan(step, data, ctrl_vec.reshape(-1, 1))
        return qpos_qvel_history

    @jax.jit
    def single_loss(
        params: Dict, qpos: ArrayImpl, qvel: ArrayImpl, ctrl_vec: ArrayImpl, qpos_des: ArrayImpl
    ) -> ArrayImpl:
        """
        Computes the loss for a single trajectory.

        Parameters:
            params (Dict): Model parameters.
            qpos (ArrayImpl): Initial position values.
            qvel (ArrayImpl): Initial velocity values.
            ctrl_vec (ArrayImpl): Control input sequence.
            qpos_des (ArrayImpl): Desired trajectory.

        Returns:
            ArrayImpl: Computed loss value.
        """
        qpos_qvel_history = predict_next(params, qpos, qvel, ctrl_vec)
        qpos_history = qpos_qvel_history[:, 0]
        loss = jp.linalg.norm(qpos_history - qpos_des)
        return loss

    # Vectorized loss computation over batch inputs
    loss_vec = jax.jit(jax.vmap(single_loss, in_axes=(None, 0, 0, 0, 0)))

    # Computes the total loss averaged over all samples
    total_loss = lambda params, qpos, qvel, ctrl_vec, qpos_des: jp.sum(
        loss_vec(params, qpos, qvel, ctrl_vec, qpos_des)
    ) / qpos.shape[0]

    # Computes gradients of the loss function w.r.t. model parameters
    grad = jax.grad(total_loss, argnums=0)

    @jax.jit
    def train_step(
        model_state, qpos: ArrayImpl, qvel: ArrayImpl, ctrl_vec: ArrayImpl, qpos_des: ArrayImpl
    ):
        """
        Performs a single training step by computing gradients and updating model parameters.

        Parameters:
            model_state: The current model state containing parameters.
            qpos (ArrayImpl): Initial position values.
            qvel (ArrayImpl): Initial velocity values.
            ctrl_vec (ArrayImpl): Control input sequence.
            qpos_des (ArrayImpl): Desired trajectory.

        Returns:
            tuple: Updated model state and computed gradients.
        """
        grads = grad(model_state.params, qpos, qvel, ctrl_vec, qpos_des)
        new_state = model_state.apply_gradients(grads=grads)
        return new_state, grads

    return total_loss, train_step


def get_batch(dataset: Dict[str, ArrayImpl], seed: jax.random.PRNGKey, indxs: ArrayImpl, batch_size: int):
    """
    Selects a random batch of data from the dataset.

    Parameters:
        dataset (Dict[str, ArrayImpl]): The dataset containing multiple data arrays.
        seed (jax.random.PRNGKey): Random seed for shuffling.
        indxs (ArrayImpl): Indices of available data.
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: A batch of data and shuffled indices.
    """
    indxs = jax.random.permutation(seed, indxs)
    batch = {key: value[indxs[:batch_size], :] for key, value in dataset.items()}
    return batch, indxs
