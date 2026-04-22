"""Discrete Weibull-shaped adstock kernel (normalized, finite horizon)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats  # scipy is core dep

from mmm.transforms.base import AdstockBase


class WeibullAdstock(AdstockBase):
    name = "weibull"

    def __init__(self, shape: float = 1.5, scale: float = 2.0, max_lag: int = 12) -> None:
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        self.shape = float(shape)
        self.scale = float(scale)
        self.max_lag = int(max_lag)
        self._weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, **kwargs: Any) -> WeibullAdstock:
        lags = np.arange(self.max_lag + 1)
        w = stats.weibull_min.pdf(lags, c=self.shape, scale=self.scale)
        w = w / (w.sum() + 1e-12)
        self._weights = w
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            self.fit(x)
        assert self._weights is not None
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        out = np.zeros(n)
        w = self._weights
        for t in range(n):
            acc = 0.0
            for lag, wt in enumerate(w):
                idx = t - lag
                if idx >= 0:
                    acc += wt * x[idx]
            out[t] = acc
        return out

    def parameter_metadata(self) -> dict[str, Any]:
        return {"shape": self.shape, "scale": self.scale, "max_lag": self.max_lag}
