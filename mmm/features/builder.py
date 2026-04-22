"""Assemble optional extra controls for diagnostics / baselines (E3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.config.extensions import FeatureEngineConfig
from mmm.data.schema import PanelSchema
from mmm.features.seasonality import fourier_from_panel
from mmm.features.trend import trend_basis


def build_extra_control_matrix(df: pd.DataFrame, schema: PanelSchema, cfg: FeatureEngineConfig) -> np.ndarray:
    blocks: list[np.ndarray] = []
    if cfg.trend_spline_knots > 0:
        T = trend_basis(df, schema.geo_column, schema.week_column, n_knots=cfg.trend_spline_knots)
        blocks.append(T)
    if cfg.fourier_yearly_harmonics > 0:
        F = fourier_from_panel(df, schema.geo_column, schema.week_column, cfg.fourier_yearly_harmonics)
        blocks.append(F)
    if cfg.holiday_country:
        from mmm.features.holidays import holiday_proximity

        hol = holiday_proximity(df[schema.week_column], [], window_days=3)
        blocks.append(hol.reshape(-1, 1))
    if not blocks:
        return np.zeros((len(df), 0))
    return np.hstack(blocks)
