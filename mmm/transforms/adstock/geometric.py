"""Geometric (infinite lag) adstock — smooth default."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.transforms.base import AdstockBase


class GeometricAdstock(AdstockBase):
    name = "geometric"

    def __init__(self, decay: float = 0.5) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0,1)")
        self.decay = float(decay)

    def fit(self, x: np.ndarray, **kwargs: Any) -> GeometricAdstock:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        out = np.zeros_like(x)
        carry = 0.0
        for i, v in enumerate(x):
            carry = v + self.decay * carry
            out[i] = carry
        return out

    def parameter_metadata(self) -> dict[str, Any]:
        return {"decay": self.decay}
