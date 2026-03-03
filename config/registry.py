from __future__ import annotations

from collections.abc import Callable

from config.base import RunConfig
from config.tasks import minipi, roki2
# from config.tasks.minipi import baseline as minipi_baseline
# from config.tasks.roki2 import torque

TaskFactory = Callable[[], RunConfig]


TASK_REGISTRY: dict[str, TaskFactory] = {
    "roki2.torque": roki2.torque,
    "roki2.karnopp_smooth": roki2.karnopp_smooth,
    "roki2.nn32_pd_friction": roki2.nn32_pd_friction,
    "roki2.nn32_ctrl_and_friction": roki2.nn32_ctrl_and_friction,
    "roki2.pd_learning": roki2.pd_learning,
    "roki2.nn32_with_changers": roki2.nn32_with_changers,
    "minipi.baseline": minipi.baseline,
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
