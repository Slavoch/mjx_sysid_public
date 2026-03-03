from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Uniform:
    min: float
    max: float


@dataclass(frozen=True)
class Normal:
    mean: float
    std: float


Distribution = Union[Uniform, Normal]
