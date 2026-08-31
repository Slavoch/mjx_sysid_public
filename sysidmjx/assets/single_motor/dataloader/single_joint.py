from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd

from config.base import DatasetSpec


DatasetDict = dict[str, Any]


@dataclass(frozen=True)
class DatasetSplit:
    data: DatasetDict
    indices: jax.Array


@dataclass(frozen=True)
class RunFrame:
    run_id: int
    file: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class SingleMotorDataLoader:
    full: DatasetDict
    train: DatasetSplit
    val: DatasetSplit
    frames_by_run: tuple[RunFrame, ...]

    @classmethod
    def from_spec(
        cls,
        *,
        spec: DatasetSpec,
        history_len: int,
        prediction_horizon: int,
        val_prediction_horizon: int | None = None,
        dtype=jnp.float64,
        repo_root: Path | None = None,
    ) -> "SingleMotorDataLoader":
        root = Path.cwd() if repo_root is None else repo_root
        _validate_spec(spec, root)
        val_horizon = prediction_horizon if val_prediction_horizon is None else val_prediction_horizon
        full, frames = _load_runs(
            runs=spec.runs,
            root=root,
            history_len=history_len,
            prediction_horizon=prediction_horizon,
            rollout_fields=spec.rollout_fields,
            dtype=dtype,
        )
        val, _ = _load_runs(
            runs=spec.val_runs,
            root=root,
            history_len=history_len,
            prediction_horizon=val_horizon,
            rollout_fields=spec.rollout_fields,
            dtype=dtype,
        )
        return cls(
            full=full,
            train=DatasetSplit(data=full, indices=jnp.arange(full["qpos_hist"].shape[0])),
            val=DatasetSplit(data=val, indices=jnp.arange(val["qpos_hist"].shape[0])),
            frames_by_run=tuple(frames),
        )

    def sample_train_batch(
        self,
        seed: jax.Array,
        batch_size: int,
    ) -> DatasetDict:
        return _sample_split_batch(self.train, seed, batch_size)


def _resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (root / path).resolve()


def _validate_spec(spec: DatasetSpec, root: Path) -> None:
    model_xml = _resolve_path(spec.model_xml, root)
    if not model_xml.exists():
        raise FileNotFoundError(f"Model XML does not exist: {model_xml}")
    for run in spec.runs:
        run_path = _resolve_path(run, root)
        if not run_path.exists():
            raise FileNotFoundError(f"Run CSV does not exist: {run_path}")
    for run in spec.val_runs:
        run_path = _resolve_path(run, root)
        if not run_path.exists():
            raise FileNotFoundError(f"Validation run CSV does not exist: {run_path}")


def _validate_raw_frame(frame: pd.DataFrame, path: Path) -> None:
    required = {"time_s", "qpos", "qvel", "qact"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Run CSV {path} is missing columns: {sorted(missing)}")


def _validate_rollout_fields(
    frame: pd.DataFrame,
    path: Path,
    rollout_fields: dict[str, str],
) -> None:
    missing = set(rollout_fields.values()).difference(frame.columns)
    if missing:
        raise ValueError(f"Run CSV {path} is missing rollout fields: {sorted(missing)}")


def _make_windows(
    *,
    frame: pd.DataFrame,
    run_id: int,
    history_len: int,
    prediction_horizon: int,
    rollout_fields: dict[str, str],
    dtype: Any,
) -> DatasetDict:
    qpos = jnp.asarray(frame["qpos"].values, dtype=dtype)
    qvel = jnp.asarray(frame["qvel"].values, dtype=dtype)
    qact = jnp.asarray(frame["qact"].values, dtype=dtype)
    n = qpos.shape[0]
    sample_count = n - history_len - prediction_horizon + 1
    if sample_count <= 0:
        raise ValueError(
            "Run is too short for requested history_len and prediction_horizon: "
            f"rows={n}, history_len={history_len}, prediction_horizon={prediction_horizon}"
        )

    starts = jnp.arange(sample_count)
    history_offsets = jnp.arange(history_len)
    control_offsets = history_len - 1 + jnp.arange(prediction_horizon)
    target_offsets = history_len + jnp.arange(prediction_horizon)
    hist_idx = starts[:, None] + history_offsets[None, :]
    control_idx = starts[:, None] + control_offsets[None, :]
    target_idx = starts[:, None] + target_offsets[None, :]
    time_index = control_idx.astype(jnp.int32)
    control = {
        name: jnp.asarray(frame[column].values, dtype=dtype)[control_idx][..., None]
        for name, column in rollout_fields.items()
    }

    return {
        "qpos_hist": qpos[hist_idx][..., None],
        "qvel_hist": qvel[hist_idx][..., None],
        "qact_hist": qact[hist_idx][..., None],
        "qact_ctrl": qact[control_idx][..., None],
        "qpos_next": qpos[target_idx][..., None],
        "qvel_next": qvel[target_idx][..., None],
        "time_index": time_index,
        "control": control,
        "run_id": jnp.full((sample_count,), run_id, dtype=jnp.int32),
    }


def _concat_datasets(datasets: list[DatasetDict]) -> DatasetDict:
    if not datasets:
        raise ValueError("DatasetSpec must contain at least one run.")
    return _concat_values(datasets)


def _concat_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {
            key: _concat_values([value[key] for value in values])
            for key in first
        }
    return jnp.concatenate(values, axis=0)


def _load_runs(
    *,
    runs: tuple[str, ...],
    root: Path,
    history_len: int,
    prediction_horizon: int,
    rollout_fields: dict[str, str],
    dtype: Any,
) -> tuple[DatasetDict, tuple[RunFrame, ...]]:
    datasets: list[DatasetDict] = []
    frames: list[RunFrame] = []
    for run_id, run in enumerate(runs):
        run_path = _resolve_path(run, root)
        frame = pd.read_csv(run_path).dropna().reset_index(drop=True)
        _validate_raw_frame(frame, run_path)
        _validate_rollout_fields(frame, run_path, rollout_fields)
        datasets.append(
            _make_windows(
                frame=frame,
                run_id=run_id,
                history_len=history_len,
                prediction_horizon=prediction_horizon,
                rollout_fields=rollout_fields,
                dtype=dtype,
            )
        )
        frames.append(
            RunFrame(
                run_id=run_id,
                file=run_path,
                frame=frame,
            )
        )
    return _concat_datasets(datasets), tuple(frames)


def _take(dataset: DatasetDict, indices: jax.Array) -> DatasetDict:
    return {key: _take_value(value, indices) for key, value in dataset.items()}


def _take_value(value: Any, indices: jax.Array) -> Any:
    if isinstance(value, dict):
        return {key: _take_value(child, indices) for key, child in value.items()}
    return value[indices]


def _sample_split_batch(
    split: DatasetSplit,
    seed: jax.Array,
    batch_size: int,
) -> DatasetDict:
    n = split.indices.shape[0]
    batch_indices = jax.random.choice(
        seed,
        n,
        shape=(batch_size,),
        replace=batch_size > n,
    )
    return _take(split.data, batch_indices)
