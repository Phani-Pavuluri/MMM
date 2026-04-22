"""E12: simple in-memory caches (process-local)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema
from mmm.validation.cv import CVStrategyBase


def _df_fingerprint(df: pd.DataFrame, schema: PanelSchema) -> str:
    key = json.dumps(
        {
            "rows": len(df),
            "cols": list(df.columns),
            "head": df.head(2).to_dict(),
            "geo": schema.geo_column,
            "week": schema.week_column,
        },
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


_cv_cache: dict[tuple[str, str], Any] = {}
_transform_cache: dict[tuple[str, str, str], np.ndarray] = {}


def get_cv_splits(
    cache_key: str,
    df: pd.DataFrame,
    schema: PanelSchema,
    strategy: CVStrategyBase,
) -> list[tuple[np.ndarray, np.ndarray]]:
    fp = _df_fingerprint(df, schema)
    k = (cache_key, fp)
    if k not in _cv_cache:
        _cv_cache[k] = strategy.split(df, schema)
    return _cv_cache[k]


def get_transformed_media(
    cache_key: str,
    fp_extra: str,
    builder: Any,
) -> np.ndarray:
    k = (cache_key, fp_extra, "X_media")
    if k not in _transform_cache:
        _transform_cache[k] = builder()
    return _transform_cache[k]
