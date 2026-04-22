"""Hill saturation — monotonic, bounded."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.transforms.base import SaturationBase


class HillSaturation(SaturationBase):
    name = "hill"

    def __init__(self, half_max: float = 1.0, slope: float = 2.0) -> None:
        if half_max <= 0:
            raise ValueError("half_max must be > 0")
        if slope <= 0:
            raise ValueError("slope must be > 0")
        self.half_max = float(half_max)
        self.slope = float(slope)

    def fit(self, x: np.ndarray, **kwargs: Any) -> HillSaturation:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x**self.slope / (self.half_max**self.slope + x**self.slope + 1e-12)

    def parameter_metadata(self) -> dict[str, Any]:
        return {"half_max": self.half_max, "slope": self.slope}
