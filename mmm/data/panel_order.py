"""Canonical (geo, week) ordering for modeling and diagnostics."""

from __future__ import annotations

import pandas as pd

from mmm.data.schema import PanelSchema


def sort_panel_for_modeling(df: pd.DataFrame, schema: PanelSchema) -> pd.DataFrame:
    """Return a copy sorted by ``geo_column`` then ``week_column`` (reset index)."""
    return df.sort_values([schema.geo_column, schema.week_column]).reset_index(drop=True)
