"""E3: simple holiday proximity flags."""

from __future__ import annotations

import numpy as np
import pandas as pd


def holiday_proximity(
    dates: pd.Series,
    holiday_dates: list[pd.Timestamp],
    window_days: int = 3,
) -> np.ndarray:
    d = pd.to_datetime(dates)
    out = np.zeros(len(d), dtype=float)
    for h in holiday_dates:
        out += (d - h).abs() <= pd.Timedelta(days=window_days)
    return np.clip(out, 0.0, 1.0)
