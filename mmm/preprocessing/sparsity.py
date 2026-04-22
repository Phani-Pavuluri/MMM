"""E17: safe handling for zero-inflated / bursty spend."""

from __future__ import annotations

import numpy as np


def spend_floor_for_log(spend: np.ndarray, floor: float = 0.5) -> np.ndarray:
    """Floor zeros for log paths without shifting signal mass aggressively."""
    return np.where(spend <= 0, floor, spend)


def burstiness_index(spend: np.ndarray) -> float:
    """Coefficient of variation as burstiness proxy."""
    m = np.mean(spend)
    return float(np.std(spend) / (m + 1e-12))
