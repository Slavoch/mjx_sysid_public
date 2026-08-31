from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable
import mujoco


@dataclass(frozen=True)
class DatasetSpec:
    model_xml: str
    runs: tuple[str, ...]
    val_runs: tuple[str, ...] = ()
    rollout_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetSpec
    dt: float = 0.002
    history_len: int = 1
    prediction_horizon: int = 1
    val_prediction_horizon: int | None = None


@dataclass(frozen=True)
class SimConfig:
    model_xml: str
    integrator: Any = mujoco.mjtIntegrator.mjINT_EULER
    iterations: int = 1


@dataclass(frozen=True)
class AssetConfig:
    assets_root: str = "assets"
    robot_assets_subdir: str = "robots"
    dataset_assets_subdir: str = "datasets"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    max_iter: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-2
    learning_rate_final: float | None = None
    early_stop_patience: int = 200
    early_stop_min_delta: float = 0.0
    loss_w_qpos: float = 1.0
    loss_w_qvel: float = 0.0
    loss_qpos_scale: float = 1.0
    loss_qvel_scale: float = 1.0
    loss_huber_delta: float | None = None


@dataclass(frozen=True)
class ComponentConfig:
    func: Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: tuple[ComponentConfig, ...] = ()
    frictions: tuple[ComponentConfig, ...] = ()
    changers: tuple[ComponentConfig, ...] = ()


@dataclass(frozen=True)
class LoggingConfig:
    runs_root: str = "runs"
    log_every: int = 1
    checkpoint_every: int = 1000


@dataclass(frozen=True)
class RunConfig:
    runner: Callable[..., dict[str, Any]]
    data: DataConfig
    sim: SimConfig
    train: TrainConfig
    experiment: ExperimentConfig
    assets: AssetConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)


def _callable_to_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
    return f"{module}.{qualname}" if module else qualname


def _to_serializable(value: Any) -> Any:
    if callable(value):
        return _callable_to_name(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "name") and hasattr(value, "value"):
        return f"{value.__class__.__name__}.{value.name}"
    if type(value).__module__.startswith("mujoco"):
        return str(value)
    if is_dataclass(value):
        return {f.name: _to_serializable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value
