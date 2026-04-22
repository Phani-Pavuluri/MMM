"""E12: simple in-memory caches (process-local)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema
from mmm.validation.cv import CVStrategyBase


def _df_fingerprint(df: pd.DataFrame, schema: PanelSchema) -> str:
    """Stable checksum over the full modeling-relevant columns (not just row count / head rows)."""
    cols = [schema.geo_column, schema.week_column, schema.target_column, *schema.channel_columns]
    sub = df.loc[:, cols].reset_index(drop=True)
    h = pd.util.hash_pandas_object(sub, index=True)
    blob = np.asarray(h.values, dtype=np.uint64).tobytes() + str(len(df)).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


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
