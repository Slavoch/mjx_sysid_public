from __future__ import annotations

from collections.abc import Callable

from config.base import RunConfig
from config.tasks import encos8112

TaskFactory = Callable[[], RunConfig]


TASK_REGISTRY: dict[str, TaskFactory] = {
    "encos8112.pd_then_changers": encos8112.pd_then_changers,
    "encos8112.pd_then_changers_friction_solver": encos8112.pd_then_changers_friction_solver,
    "encos8112.pd_then_karnopp_stribeck": encos8112.pd_then_karnopp_stribeck,
    "encos8112.pd_then_isotropic_nn_frictionloss": encos8112.pd_then_isotropic_nn_frictionloss,
    "encos8112.pd_then_isotropic_nn": encos8112.pd_then_isotropic_nn,
    "encos8112.pd_then_nn": encos8112.pd_then_nn,
}


def available_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def get_task_config(task_key: str) -> RunConfig:
    try:
        return TASK_REGISTRY[task_key]()
    except KeyError as exc:
        choices = ", ".join(available_tasks())
        raise KeyError(
            f"Unknown task '{task_key}'. Available tasks: {choices}"
        ) from exc
