# run_experiment.py
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import mujoco
from mujoco import mjx
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sysidmjx.core import generate_loss_train_functions, get_batch
from sysidmjx.assets.single_motor.dataloader.single_joint import data_load

def run_experiment(PARAMS):
    seed_key = jax.random.PRNGKey(PARAMS["SEED"])
    dataset, _ = data_load(
        data_path=PARAMS["DATASET"]["PATH"],
        motor_id=PARAMS["DATASET"]["motor_id"],
        num_lags=PARAMS["DATASET"]["num_lags"],
    )

    mj_model = mujoco.MjModel.from_xml_path(PARAMS["SIM"]["PATH"])
    mj_model.opt.timestep = PARAMS["DATASET"]["DT"]
    mj_model.opt.iterations = PARAMS["SIM"]["ITERATIONS"]
    mj_model.opt.integrator = PARAMS["SIM"]["INTEGRATOR"]
    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    keys = jax.random.split(seed_key, num=5)

    init_params = {
        "armature": jax.random.uniform(keys[2], maxval=0.8, shape=(1,)),
        "frictionloss": jax.random.uniform(keys[3], maxval=1.5, shape=(1,)),
        "damping": jax.random.uniform(keys[4], maxval=1.5, shape=(1,)),
    }

    zero_params = {
        "armature": jnp.array([0.0]),
        "frictionloss": jnp.array([0.0]),
        "damping": jnp.array([0.0]),
    }

    @jax.jit
    def change_model(params, old_model):
        return old_model.replace(
            dof_armature=abs(params["armature"]),
            dof_frictionloss=abs(params["frictionloss"]),
            dof_damping=abs(params["damping"]),
        )

    @jax.jit
    def make_action(params, data, ctrl):
        kp, kv = 20, 1
        tau = kp * (ctrl - data.qpos) - kv * data.qvel
        return tau

    total_loss, train_step, _ = generate_loss_train_functions(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        change_model=change_model,
        make_action=make_action,
    )

    state = train_state.TrainState.create(
        apply_fn=None,
        params=init_params,
        tx=PARAMS["TRAIN"]["TX"],
    )

    loss_hist = []
    params_hist = []
    indxs = jnp.array(range(dataset["qpos"].shape[0]))

    for epoch in range(PARAMS["TRAIN"]["EPOCH_NUM"]):
        loss = total_loss(
            state.params,
            qpos=dataset["qpos"],
            qvel=dataset["qvel"],
            ctrl_vec=dataset["qact"],
            qpos_des=dataset["qpos_next"],
        )
        loss_hist.append(loss)
        params_hist.append(state.params)

        batch, indxs = get_batch(dataset, seed_key, indxs, PARAMS["TRAIN"]["BATCH_SIZE"])
        state, _ = train_step(
            state,
            qpos=batch["qpos"],
            qvel=batch["qvel"],
            ctrl_vec=batch["qact"],
            qpos_des=batch["qpos_next"],
        )

    baseline_loss = total_loss(
        zero_params,
        qpos=dataset["qpos"],
        qvel=dataset["qvel"],
        ctrl_vec=dataset["qact"],
        qpos_des=dataset["qpos_next"],
    )

    adjusted_model_loss = np.array(loss_hist)
    base_line = np.ones_like(adjusted_model_loss)

    folder_path = os.path.join("assets/experiments", PARAMS["EXPERIMENT_NAME"])
    img_path = os.path.join(folder_path, "pictures")
    os.makedirs(img_path, exist_ok=True)

    joblib.dump(
        {
            "params": params_hist,
            "loss_hist": adjusted_model_loss,
            "baseline_loss": baseline_loss,
        },
        os.path.join(folder_path, "optimization_info.joblib"),
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(adjusted_model_loss / baseline_loss, label="Relative loss")
    ax1.plot(base_line, label="Baseline loss (normalized to 1)")
    ax1.set_title("Parameters convergence")
    ax1.set_ylabel("Relative loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot([p["armature"] for p in params_hist], label="armature")
    ax2.plot([p["frictionloss"] for p in params_hist], label="frictionloss")
    ax2.plot([p["damping"] for p in params_hist], label="damping")
    ax2.set_ylabel("Motor parameters")
    ax2.legend()
    ax2.grid(True)

    fig.supxlabel("Epoch")
    fig.savefig(os.path.join(img_path, "loss.png"), format="png", bbox_inches="tight")
    plt.close(fig)
