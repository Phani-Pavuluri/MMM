"""Time-series CV strategies with explicit split axes (calendar week vs legacy geo-rank)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.config.schema import CVConfig, CVMode, CVSplitAxis
from mmm.data.schema import PanelSchema, week_index_per_geo


def _sort_panel(df: pd.DataFrame, schema: PanelSchema) -> pd.DataFrame:
    return df.sort_values([schema.geo_column, schema.week_column]).reset_index(drop=True)


def _calendar_week_index(df: pd.DataFrame, schema: PanelSchema) -> np.ndarray:
    """Dense 0..W-1 index per row from global calendar week (same calendar week → same id)."""
    w = df[schema.week_column]
    if not np.issubdtype(pd.Series(w).dtype, np.datetime64):
        wnum = pd.to_numeric(w, errors="coerce")
        if bool(wnum.isna().any()):
            raise ValueError(
                f"Calendar CV: missing or non-numeric values in {schema.week_column!r}; "
                "fix data or use a different split_axis."
            )
        u = np.sort(np.unique(wnum.to_numpy()))
        return np.searchsorted(u, wnum.to_numpy()).astype(int)
    wt = pd.to_datetime(w, errors="coerce")
    if bool(wt.isna().any()):
        raise ValueError(
            f"Calendar CV: invalid or missing datetimes in {schema.week_column!r}; "
            "fix data or use a different split_axis."
        )
    norm = wt.dt.normalize()
    uniq = pd.Index(norm.unique()).sort_values()
    mapping = {t: i for i, t in enumerate(uniq)}
    return norm.map(mapping).astype(int).to_numpy()


class CVStrategyBase(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (train_loss_mask, val_loss_mask) booleans aligned to sorted panel row order."""


@dataclass
class RollingGeoRankCV(CVStrategyBase):
    """Legacy: within-geo dense rank, synchronized cut across all rows (deprecated for production)."""

    n_splits: int
    min_train_weeks: int
    horizon_weeks: int
    gap_weeks: int = 0

    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        df = _sort_panel(df, schema)
        widx = week_index_per_geo(df, schema.geo_column, schema.week_column).to_numpy()
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        max_w = int(widx.max()) + 1
        if max_w < self.min_train_weeks + self.horizon_weeks + self.gap_weeks:
            return splits
        edges = np.linspace(self.min_train_weeks, max_w - self.horizon_weeks - self.gap_weeks, self.n_splits + 1)
        for i in range(self.n_splits):
            cut = int(edges[i + 1])
            train = widx < cut
            val = (widx >= cut + self.gap_weeks) & (widx < cut + self.gap_weeks + self.horizon_weeks)
            if val.any() and train.sum() >= self.min_train_weeks:
                splits.append((train, val))
        return splits


@dataclass
class ExpandingGeoRankCV(CVStrategyBase):
    n_splits: int
    min_train_weeks: int
    horizon_weeks: int
    gap_weeks: int = 0

    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        df = _sort_panel(df, schema)
        widx = week_index_per_geo(df, schema.geo_column, schema.week_column).to_numpy()
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        max_w = int(widx.max()) + 1
        start_train = self.min_train_weeks
        step = max(1, (max_w - start_train - self.horizon_weeks - self.gap_weeks) // max(self.n_splits, 1))
        cut = start_train
        while cut + self.gap_weeks + self.horizon_weeks <= max_w and len(splits) < self.n_splits:
            train = widx < cut
            val = (widx >= cut + self.gap_weeks) & (widx < cut + self.gap_weeks + self.horizon_weeks)
            if val.any() and train.sum() >= self.min_train_weeks:
                splits.append((train, val))
            cut += step
        return splits


@dataclass
class RollingCalendarWindowCV(CVStrategyBase):
    """Train/val masks from global calendar week index (recommended for weekly geo panels)."""

    n_splits: int
    min_train_weeks: int
    horizon_weeks: int
    gap_weeks: int = 0

    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        df = _sort_panel(df, schema)
        wid = _calendar_week_index(df, schema)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        max_w = int(wid.max()) + 1
        if max_w < self.min_train_weeks + self.horizon_weeks + self.gap_weeks:
            return splits
        edges = np.linspace(self.min_train_weeks, max_w - self.horizon_weeks - self.gap_weeks, self.n_splits + 1)
        for i in range(self.n_splits):
            cut = int(edges[i + 1])
            train = wid < cut
            val = (wid >= cut + self.gap_weeks) & (wid < cut + self.gap_weeks + self.horizon_weeks)
            if val.any() and train.sum() >= self.min_train_weeks:
                splits.append((train, val))
        return splits


@dataclass
class ExpandingCalendarWindowCV(CVStrategyBase):
    n_splits: int
    min_train_weeks: int
    horizon_weeks: int
    gap_weeks: int = 0

    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        df = _sort_panel(df, schema)
        wid = _calendar_week_index(df, schema)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        max_w = int(wid.max()) + 1
        start_train = self.min_train_weeks
        step = max(1, (max_w - start_train - self.horizon_weeks - self.gap_weeks) // max(self.n_splits, 1))
        cut = start_train
        while cut + self.gap_weeks + self.horizon_weeks <= max_w and len(splits) < self.n_splits:
            train = wid < cut
            val = (wid >= cut + self.gap_weeks) & (wid < cut + self.gap_weeks + self.horizon_weeks)
            if val.any() and train.sum() >= self.min_train_weeks:
                splits.append((train, val))
            cut += step
        return splits


@dataclass
class GeoBlockedHoldoutCV(CVStrategyBase):
    """Hold out disjoint geo sets per fold (no time leakage across geos in val)."""

    n_splits: int
    seed: int
    min_train_geos: int = 1

    def split(self, df: pd.DataFrame, schema: PanelSchema) -> list[tuple[np.ndarray, np.ndarray]]:
        df = _sort_panel(df, schema)
        geos = df[schema.geo_column].unique()
        if len(geos) < 2:
            return []
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(geos)
        k = max(1, min(self.n_splits, len(perm)))
        chunks = np.array_split(perm, k)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for val_geos in chunks:
            vset = set(val_geos.tolist())
            if not vset:
                continue
            val = df[schema.geo_column].isin(vset).to_numpy()
            train = ~val
            if not val.any() or train.sum() == 0:
                continue
            if df.loc[train, schema.geo_column].nunique() < self.min_train_geos:
                continue
            out.append((train, val))
        return out


# Back-compat aliases
RollingWindowCV = RollingGeoRankCV
ExpandingWindowCV = ExpandingGeoRankCV


def auto_cv_mode(df: pd.DataFrame, schema: PanelSchema, cfg: CVConfig) -> CVStrategyBase:
    axis = cfg.split_axis
    if isinstance(axis, str):
        axis = CVSplitAxis(axis)

    def _rolling_calendar():
        return RollingCalendarWindowCV(cfg.n_splits, cfg.min_train_weeks, cfg.horizon_weeks, cfg.gap_weeks)

    def _expanding_calendar():
        return ExpandingCalendarWindowCV(cfg.n_splits, cfg.min_train_weeks, cfg.horizon_weeks, cfg.gap_weeks)

    def _rolling_geo():
        return RollingGeoRankCV(cfg.n_splits, cfg.min_train_weeks, cfg.horizon_weeks, cfg.gap_weeks)

    def _expanding_geo():
        return ExpandingGeoRankCV(cfg.n_splits, cfg.min_train_weeks, cfg.horizon_weeks, cfg.gap_weeks)

    if axis == CVSplitAxis.GEO_BLOCKED:
        return GeoBlockedHoldoutCV(cfg.n_splits, seed=cfg.geo_blocked_seed)

    if axis == CVSplitAxis.GEO_RANK:
        if cfg.mode == CVMode.ROLLING:
            return _rolling_geo()
        if cfg.mode == CVMode.EXPANDING:
            return _expanding_geo()
        max_w = int(week_index_per_geo(df, schema.geo_column, schema.week_column).max()) + 1
        if max_w < 3 * cfg.min_train_weeks:
            return _expanding_geo()
        return _rolling_geo()

    # calendar (default)
    if cfg.mode == CVMode.ROLLING:
        return _rolling_calendar()
    if cfg.mode == CVMode.EXPANDING:
        return _expanding_calendar()
    wid = _calendar_week_index(_sort_panel(df, schema), schema)
    max_w = int(wid.max()) + 1
    if max_w < 3 * cfg.min_train_weeks:
        return _expanding_calendar()
    return _rolling_calendar()
