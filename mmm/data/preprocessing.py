"""Preprocessing helpers: filtering, holdouts, winsorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema
from mmm.utils.math import winsorize_series


@dataclass
class PreprocessConfig:
    geos_include: list[str] | None = None
    channels_include: list[str] | None = None
    winsorize_target: tuple[float, float] | None = None  # quantiles
    drop_zero_spend_weeks: bool = False


def apply_preprocess(df: pd.DataFrame, schema: PanelSchema, cfg: PreprocessConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.geos_include is not None:
        out = out[out[schema.geo_column].isin(cfg.geos_include)]
    if cfg.channels_include is not None:
        keep = set(cfg.channels_include)
        chans = [c for c in schema.channel_columns if c in keep]
        if len(chans) != len(cfg.channels_include):
            missing = set(cfg.channels_include) - set(schema.channel_columns)
            raise ValueError(f"Unknown channels in filter: {missing}")
    if cfg.winsorize_target is not None:
        lo, hi = cfg.winsorize_target
        vals = out[schema.target_column].to_numpy(dtype=float)
        out[schema.target_column] = winsorize_series(vals, lo, hi)
    if cfg.drop_zero_spend_weeks:
        spend_sum = out[list(schema.channel_columns)].sum(axis=1)
        out = out[spend_sum > 0]
    return out.reset_index(drop=True)


def time_holdout_mask(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    holdout_last_n_weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train, test) based on global last N distinct weeks."""
    weeks = pd.Index(df[schema.week_column]).unique().sort_values()
    if holdout_last_n_weeks <= 0 or holdout_last_n_weeks >= len(weeks):
        raise ValueError("holdout_last_n_weeks must be in (0, n_distinct_weeks)")
    holdout_weeks = set(weeks[-holdout_last_n_weeks:])
    test = df[schema.week_column].isin(holdout_weeks).to_numpy()
    train = ~test
    return train, test
