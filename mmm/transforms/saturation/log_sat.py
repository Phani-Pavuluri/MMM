"""Log compression saturation."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.transforms.base import SaturationBase
from mmm.utils.math import safe_log


class LogSaturation(SaturationBase):
    name = "log"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def fit(self, x: np.ndarray, **kwargs: Any) -> LogSaturation:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return safe_log(1.0 + x / (self.scale + 1e-12))

    def parameter_metadata(self) -> dict[str, Any]:
        return {"scale": self.scale}
