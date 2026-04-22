"""Strict panel schema validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PanelSchema:
    geo_column: str
    week_column: str
    target_column: str
    channel_columns: tuple[str, ...]
    control_columns: tuple[str, ...] = ()


class PanelValidationError(ValueError):
    pass


def validate_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    allow_nan_controls: bool = True,
) -> pd.DataFrame:
    required = (
        schema.geo_column,
        schema.week_column,
        schema.target_column,
        *schema.channel_columns,
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PanelValidationError(f"Missing required columns: {missing}")

    if df[schema.geo_column].isna().any():
        raise PanelValidationError(f"NaNs in {schema.geo_column}")
    if df[schema.week_column].isna().any():
        raise PanelValidationError(f"NaNs in {schema.week_column}")
    if df[schema.target_column].isna().any():
        raise PanelValidationError(f"NaNs in target {schema.target_column}")

    for ch in schema.channel_columns:
        if (df[ch] < 0).any():
            raise PanelValidationError(f"Negative spend in channel {ch}")

    for ctrl in schema.control_columns:
        if ctrl not in df.columns:
            raise PanelValidationError(f"Missing control {ctrl}")
        if not allow_nan_controls and df[ctrl].isna().any():
            raise PanelValidationError(f"NaNs in control {ctrl}")

    dup = df.duplicated(subset=[schema.geo_column, schema.week_column])
    if dup.any():
        raise PanelValidationError("Duplicate (geo, week) rows present")

    return df


def to_internal_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """Return canonical long panel sorted by geo, week."""
    out = df.copy()
    if sort:
        out = out.sort_values([schema.geo_column, schema.week_column]).reset_index(drop=True)
    return out


def week_index_per_geo(df: pd.DataFrame, geo_column: str, week_column: str) -> pd.Series:
    """Dense integer week order within each geo (for CV indexing)."""
    return df.groupby(geo_column)[week_column].rank(method="dense").astype(int) - 1
