from typing import Tuple, Dict
import jax
import jax.numpy as jp
from mujoco import mjx
from jax import Array 
from jax import config
from functools import partial


config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")


def build_torque_optimizer(mjx_model: mjx.Model, mjx_data: mjx.Data, alpha):
    sim_step = jax.jit(mjx.step)

    @jax.jit
    def predict(
        qpos: Array, qvel: Array, tau: Array
    ) -> Tuple[Array, Array]:
        data = mjx_data.tree_replace({"qpos": qpos, "qvel": qvel})
        data = mjx.forward(mjx_model, data)
        data = data.tree_replace({"ctrl": tau})
        data = sim_step(mjx_model, data)
        return data.qpos, data.qvel

    @jax.jit
    def loss_fn_single(tau, qpos, qvel, qpos_target, qvel_target, alpha):
        qpos_sim, qvel_sim = predict(qpos, qvel, tau)
        pos_loss = jp.sum((qpos_sim - qpos_target) ** 2)
        vel_loss = jp.sum((qvel_sim - qvel_target) ** 2)
        return pos_loss + alpha * vel_loss

    grad_fn_single = jax.grad(loss_fn_single)
    loss_fn = jax.jit(jax.vmap(partial(loss_fn_single, alpha=alpha)))
    grad_fn = jax.jit(jax.vmap(partial(grad_fn_single, alpha=alpha)))

    def optimize_torque(qpos, qvel, qpos_target, qvel_target, lr=1e-2, steps=100):
        batch_size = qpos.shape[0]
        tau_seq = jp.zeros((batch_size, mjx_model.nu))  # [batch, nu]

        info = {
            "loss": [],
            "grad_norm": [],
            "tau_hist": [],
        }

        for step in range(steps):
            info["tau_hist"].append(tau_seq.copy())
            loss = loss_fn(tau_seq, qpos, qvel, qpos_target, qvel_target)
            grad = grad_fn(tau_seq, qpos, qvel, qpos_target, qvel_target)

            tau_seq = tau_seq - lr * grad
            info["loss"].append(jp.mean(loss))
            info["grad_norm"].append(jp.linalg.norm(grad))
            print(
                f"[Step {step:03d}] Avg Loss: {info['loss'][-1]:.6f} | "
                f"Grad norm: {info['grad_norm'][-1]:.6f}"
            )

        return tau_seq, info  # shape [batch, nu]

    return optimize_torque


def find_optimal_tau(
    dataset: Dict[str, Array],
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    steps=100,
    lr=1e-2,
    alpha=0.01,
):
    optimizer = build_torque_optimizer(mjx_model, mjx_data, alpha)

    qtau, info = optimizer(
        qpos=dataset["qpos"],
        qvel=dataset["qvel"],
        qpos_target=dataset["qpos_next"],
        qvel_target=dataset["qvel_next"],
        lr=lr,
        steps=steps,
    )
    return qtau, info
