from __future__ import annotations
from typing import Tuple, Dict

import jax, jax.numpy as jnp
from jax import Array, lax
from functools import partial
from mujoco import mjx          # pip install mujoco
from jax import debug


# ──────────────────────────────────────────────────────────────────────────────
#  JAX flags
# ──────────────────────────────────────────────────────────────────────────────
jax.config.update("jax_debug_nans",   True)
jax.config.update("jax_enable_x64",   True)
jax.config.update("jax_default_matmul_precision", "high")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def quat_dist(q1: Array, q2: Array) -> Array:
    """Geodesic distance on S³, in [0, 2]."""
    inner = jnp.abs(jnp.sum(q1 * q2, axis=-1))
    inner = jnp.clip(inner, 0.0, 1.0)
    return 2.0 * jnp.arccos(inner)        # radians


def assemble_qpos(root_xyz: Array, root_quat: Array,
                  joint_q: Array) -> Array:
    """Concatenate into full MuJoCo qpos = [xyz‖quat‖joint]."""
    return jnp.concatenate([root_xyz, root_quat, joint_q], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_torque_sequence_optimizer(
        *,
        mjx_model: mjx.Model,
        mjx_data:  mjx.Data,
        root_xyz0: Array,             # shape (3,) – you *must* supply
        qvel0:     Array,             # shape (nv,) – all initial vels
        weights:   Dict[str, float] | None = None,
):
    """
    Returns `optimize_tau_seq(dataset, *, lr, steps)`.

    Unknowns (root xyz, velocities) are fixed here so the jit can compile once.
    """

    if weights is None:
        weights = dict(joint=1.0, orient=1.0, avel=0.1)

    root_xyz0 = jnp.asarray(root_xyz0)
    qvel0     = jnp.asarray(qvel0)

    sim_step = jax.jit(mjx.step)

    # ------------------------------------------------------------------#
    # one-sample rollout                                                #
    # ------------------------------------------------------------------#
    def rollout_single(qpos0: Array, tau_seq: Array
                       ) -> Tuple[Array, Array]:
        """
        tau_seq (T,nu) → qpos_traj, qvel_traj both (T,nq/nv)
        (state AFTER each control, length == horizon == tau_seq.shape[0])
        """

        def _step(carry, tau_t):
            qpos_t, qvel_t = carry
            d = mjx_data.tree_replace({"qpos": qpos_t,
                                       "qvel": qvel_t,
                                       "ctrl": tau_t})
            d = sim_step(mjx_model, d)
            return (d.qpos, d.qvel), (d.qpos, d.qvel)

        (_, _), (qpos_traj, qvel_traj) = lax.scan(
            _step,
            (qpos0, qvel0),          # carry starts with *fixed* qvel0
            tau_seq
        )
        return qpos_traj, qvel_traj

    rollout_single = jax.jit(rollout_single)

    # ------------------------------------------------------------------#
    # loss for ONE sample                                               #
    # ------------------------------------------------------------------#
    def loss_with_terms(tau_seq, qpos0, joint_target, quat_target, avel_target):
        qpos_traj, qvel_traj = rollout_single(qpos0, tau_seq)

        joint_sim = qpos_traj[:, 7:]
        quat_sim  = qpos_traj[:, 3:7]
        avel_sim  = qvel_traj[:, 3:6]

        def mse(x, y):
            return jnp.mean((x - y) ** 2)     # 1/(T·Nq) factor built-in
        joint_loss  = mse(joint_sim,  joint_target)
        orient_loss = mse(quat_sim,   quat_target)          # after sign fix, see below
        avel_loss   = mse(avel_sim,   avel_target)

        total_loss = (weights["joint"]  * joint_loss +
                    weights["orient"] * orient_loss +
                    weights["avel"]   * avel_loss)

        return total_loss, {
            "joint": joint_loss,
            "orient": orient_loss,
            "avel": avel_loss,
        }
    
    def loss_single(tau_seq, qpos0, joint_target, quat_target, avel_target):
        total, _ = loss_with_terms(tau_seq, qpos0, joint_target, quat_target, avel_target)
        return total

    grad_single = jax.grad(loss_single)

    loss_batch = jax.jit(jax.vmap(loss_single,
                                  in_axes=(0, 0, 0, 0, 0)))
    grad_batch = jax.jit(jax.vmap(grad_single,
                                  in_axes=(0, 0, 0, 0, 0)))

    # ------------------------------------------------------------------#
    #  optimiser                                                        #
    # ------------------------------------------------------------------#
    def optimize_tau_seq(dataset: Dict[str, Array],
                         *,
                         lr:    float = 1e-2,
                         steps: int   = 500
                         ) -> Tuple[Array, Dict]:
        """
        dataset must contain
          - 'joint_qpos' : (B, T, n_joint)
          - 'root_quat'  : (B, T, 4)
          - 'root_avel'  : (B, T, 3)

        Only frames 1…T-1 are matched; frame 0 seeds the simulation.
        """

        B, T, _ = dataset["joint_qpos"].shape
        horizon = T - 1                         # because of the +1 shift

        # create qpos0 for each sample
        qpos0 = assemble_qpos(
            jnp.broadcast_to(root_xyz0, (B, 3)),
            dataset["root_quat"][:, 0],
            dataset["joint_qpos"][:, 0]
        )

        # shifted targets -------------------------------------------------#
        tgt_joint = dataset["joint_qpos"][:, 1:]
        tgt_quat  = dataset["root_quat"][:, 1:]
        tgt_avel  = dataset["root_avel"][:, 1:]

        tau_seq = jnp.zeros((B, horizon, mjx_model.nu))

        log = dict(loss=[], grad_norm=[], tau_hist=[])

        for k in range(steps):
            log["tau_hist"].append(tau_seq)
            total, terms = loss_with_terms(
                tau_seq[0],
                qpos0[0],
                tgt_joint[0],
                tgt_quat[0],
                tgt_avel[0],
            )

            print("Total Loss:", float(total))
            print("  Joint  :", float(terms["joint"]))
            print("  Orient :", float(terms["orient"]))
            print("  Avel   :", float(terms["avel"]))

            loss = loss_batch(tau_seq, qpos0,
                              tgt_joint, tgt_quat, tgt_avel)
            grad = grad_batch(tau_seq, qpos0,
                              tgt_joint, tgt_quat, tgt_avel)
            rms   = jnp.sqrt(jnp.mean(grad**2))
            max_g = jnp.max(jnp.abs(grad))
            print("grad_rms:", float(rms), "grad_max:", float(max_g))

            tau_seq = tau_seq - lr * grad / (1e-8 + jnp.linalg.norm(grad))

            log["loss"].append(jnp.mean(loss))
            log["grad_norm"].append(jnp.linalg.norm(grad))

            print(f"[{k:04d}]  ⟨loss⟩={log['loss'][-1]:.6f}  "
                  f"grad_norm={log['grad_norm'][-1]:.6f}")

        return tau_seq, log

    return optimize_tau_seq
