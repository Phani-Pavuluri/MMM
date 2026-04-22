"""E18: replay-style implied lift from counterfactual spend (coarse)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema


def implied_lift_ratio(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    df_control: pd.DataFrame,
    df_treat: pd.DataFrame,
    schema: PanelSchema,
) -> float:
    """Ratio of mean predicted KPI (level) treat vs control minus 1 (model-implied relative lift)."""
    yc = predict_fn(df_control)
    yt = predict_fn(df_treat)
    return float(yt.mean() / (yc.mean() + 1e-12) - 1.0)
