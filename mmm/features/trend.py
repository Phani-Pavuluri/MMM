"""E3: trend basis on normalized within-geo time rank."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trend_basis(df: pd.DataFrame, geo_column: str, week_column: str, n_knots: int = 4) -> np.ndarray:
    """Polynomials of normalized time in [0,1] per geo."""
    if n_knots <= 0:
        return np.zeros((len(df), 0))
    degree = min(4, max(1, n_knots - 1))
    rk = df.groupby(geo_column)[week_column].rank(method="dense").to_numpy(dtype=float) - 1.0
    denom = df.groupby(geo_column)[week_column].transform("nunique").to_numpy(dtype=float) - 1.0
    r = rk / np.maximum(denom, 1.0)
    r = np.clip(r, 0.0, 1.0)
    return np.column_stack([r**k for k in range(1, degree + 1)])
