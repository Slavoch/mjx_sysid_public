import pandas as pd
import jax.numpy as jnp


def data_load(
    data_path: str,
    motor_id: int,
    num_lags: int,
    dtype=jnp.float32,
):
    df_hist = pd.read_csv(data_path).dropna().reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 1.  Collect lagged columns into a dict-of-dicts
    # ------------------------------------------------------------------ #
    hist = {}
    for feat in ["qpos", "qvel", "qact"]:
        hist[feat] = {}
        for lag in range(num_lags + 1):
            col = f"{feat}_motor{motor_id}_lag{lag}"
            hist[feat][f"lag{lag}"] = df_hist[col].values[:, None]  # (N,1)

    # ------------------------------------------------------------------ #
    # 2.  Base dataframe with current (lag 0) signals
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(
        {
            "qpos": hist["qpos"]["lag0"].squeeze(),
            "qvel": hist["qvel"]["lag0"].squeeze(),
            "qact": hist["qact"]["lag0"].squeeze(),
        }
    )

    # ------------------------------------------------------------------ #
    # 3.  Future targets  (lag 1 … lag K)
    # ------------------------------------------------------------------ #
    for k in range(num_lags):
        df[f"qpos_next_{k}"] = hist["qpos"][f"lag{k+1}"].squeeze()
        df[f"qvel_next_{k}"] = hist["qvel"][f"lag{k+1}"].squeeze()
        df[f"qact_next_{k}"] = hist["qact"][f"lag{k+1}"].squeeze()

    # ------------------------------------------------------------------ #
    # 4.  Package into JAX arrays
    # ------------------------------------------------------------------ #
    N = len(df)
    # Per-sample per-step absolute time indices (N, num_lags):
    # time_index[i, k] = i + k
    time_index = (
        jnp.arange(N, dtype=jnp.int32)[:, None]
        + jnp.arange(num_lags, dtype=jnp.int32)[None, :]
    )
    dataset = {
        "qpos": jnp.asarray(df["qpos"].values[:, None], dtype=dtype),
        "qvel": jnp.asarray(df["qvel"].values[:, None], dtype=dtype),
        "qact": jnp.asarray(
            df[[f"qact_next_{k}" for k in range(num_lags)]].values,
            dtype=dtype,
        ),
        "qpos_next": jnp.asarray(
            df[[f"qpos_next_{k}" for k in range(num_lags)]].values,
            dtype=dtype,
        ),
        "qvel_next": jnp.asarray(
            df[[f"qvel_next_{k}" for k in range(num_lags)]].values,
            dtype=dtype,
        ),
        "time_index": time_index,
    }

    return dataset, df
