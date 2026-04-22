"""Per-channel mean scaling for numerical stability."""

from __future__ import annotations

import numpy as np


class MeanScaler:
    def __init__(self) -> None:
        self.scale_: float | None = None

    def fit(self, x: np.ndarray) -> MeanScaler:
        m = float(np.mean(np.asarray(x, dtype=float)))
        self.scale_ = m if m > 0 else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("Scaler not fit")
        return np.asarray(x, dtype=float) / self.scale_
