"""Small numerical helpers."""

from __future__ import annotations

import numpy as np


def safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, floor))


def winsorize_series(x: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    lo, hi = np.quantile(x, [lower_q, upper_q])
    return np.clip(x, lo, hi)
