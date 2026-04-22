"""E3: Fourier seasonality for weekly index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_week_features(week_index: np.ndarray, n_harmonics: int) -> np.ndarray:
    """week_index in [0, 2pi) annual cycle approximated over 52 weeks."""
    if n_harmonics <= 0:
        return np.zeros((len(week_index), 0))
    t = 2 * np.pi * (week_index % 52.0) / 52.0
    parts = []
    for k in range(1, n_harmonics + 1):
        parts.append(np.sin(k * t))
        parts.append(np.cos(k * t))
    return np.column_stack(parts)


def fourier_from_panel(df: pd.DataFrame, geo_column: str, week_column: str, n_harmonics: int) -> np.ndarray:
    wk = df.groupby(geo_column)[week_column].rank(method="dense").to_numpy(dtype=float) - 1.0
    return fourier_week_features(wk, n_harmonics)
