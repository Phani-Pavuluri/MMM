"""Logistic saturation on normalized input."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.transforms.base import SaturationBase


class LogisticSaturation(SaturationBase):
    name = "logistic"

    def __init__(self, midpoint: float = 1.0, growth: float = 1.0) -> None:
        self.midpoint = float(midpoint)
        self.growth = float(growth)

    def fit(self, x: np.ndarray, **kwargs: Any) -> LogisticSaturation:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = self.growth * (x - self.midpoint)
        z = np.clip(z, -60, 60)
        return 1.0 / (1.0 + np.exp(-z))

    def parameter_metadata(self) -> dict[str, Any]:
        return {"midpoint": self.midpoint, "growth": self.growth}
