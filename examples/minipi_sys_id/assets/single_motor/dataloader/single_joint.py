import pandas as pd
import jax.numpy as jnp

def data_load(data_path, motor_id, num_lags):
    df_hist = pd.read_csv(data_path)

    hist = dict(qpos=[], qvel=[], act_desired=[], time=[])
    for property in ["qpos", "qvel", "qact"]:
        hist[property] = {}
        for lag_id in range(num_lags + 1):
            data = jnp.vstack(
                [df_hist[f"{property}_motor{motor_id}_lag{lag_id}"].values]
            ).T
            hist[property][f"motor{motor_id}_lag{lag_id}"] = data

    df = df_hist[
        [
            f"qpos_motor{motor_id}_lag0",
            f"qvel_motor{motor_id}_lag0",
            f"qact_motor{motor_id}_lag0",
        ]
    ]
    df = df.rename(
        columns={
            f"qpos_motor{motor_id}_lag0": "qpos",
            f"qvel_motor{motor_id}_lag0": "qvel",
            f"qact_motor{motor_id}_lag0": "qact",
        }
    )
    for i in range(num_lags):
        df[f"qpos_next_{i}"] = hist["qpos"][f"motor{motor_id}_lag{i+1}"]

    for i in range(num_lags):
        df[f"qact_next_{i}"] = hist["qact"][f"motor{motor_id}_lag{i+1}"]

    dataset = {
        "qpos": jnp.array(df["qpos"].values).reshape(-1, 1),
        "qvel": jnp.array(df["qvel"].values).reshape(-1, 1),
        "qact": jnp.array(
            df[[f"qact_next_{i}" for i in range(num_lags)]]
        ),
        "qpos_next": jnp.array(
            df[[f"qpos_next_{i}" for i in range(num_lags)]]
        ),
    }
    return dataset, df
