"""Pooling structures for hierarchical / grouped models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.config.schema import PoolingMode
from mmm.data.schema import PanelSchema


@dataclass(frozen=True)
class PoolingSpec:
    mode: PoolingMode
    group_column: str | None = None  # e.g. region for nested pooling


def partial_pooling_indices(df: pd.DataFrame, schema: PanelSchema) -> np.ndarray:
    """Integer geo index for hierarchical grouping."""
    geos = pd.Index(df[schema.geo_column].unique())
    mapping = {g: i for i, g in enumerate(geos)}
    return df[schema.geo_column].map(mapping).to_numpy(dtype=int)
