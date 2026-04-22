"""
Explicit replay estimand semantics — **no implicit full-panel mean**.

Every replay unit must carry a serialized :class:`ReplayEstimandSpec` (``CalibrationUnit.replay_estimand``)
defining geo scope, evaluation window, aggregation rule, and KPI alignment for implied vs observed lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema

GeoScope = Literal["all", "listed"]
AggregationRule = Literal["mean", "sum", "geo_mean_then_global_mean"]


@dataclass(frozen=True)
class ReplayEstimandSpec:
    """
    Machine-evaluable estimand for replay implied lift.

    ``week_start`` / ``week_end`` are compared against ``schema.week_column`` using the same
    element-wise comparison rules as the panel (numeric or datetime-coercible).
    """

    geo_scope: GeoScope
    geo_ids: tuple[str, ...]
    week_start: Any
    week_end: Any
    aggregation: AggregationRule
    target_kpi_column: str
    lift_scale: str
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "geo_scope": self.geo_scope,
            "geo_ids": list(self.geo_ids),
            "week_start": self.week_start,
            "week_end": self.week_end,
            "aggregation": self.aggregation,
            "target_kpi_column": self.target_kpi_column,
            "lift_scale": self.lift_scale,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReplayEstimandSpec:
        if not isinstance(d, dict):
            raise ValueError("replay_estimand must be a dict")
        gs = d.get("geo_scope", "listed")
        if gs not in ("all", "listed"):
            raise ValueError(f"replay_estimand.geo_scope must be 'all' or 'listed', got {gs!r}")
        gids = d.get("geo_ids") or []
        if gs == "listed" and not gids:
            raise ValueError("replay_estimand.geo_ids required when geo_scope='listed'")
        agg = d.get("aggregation", "mean")
        if agg not in ("mean", "sum", "geo_mean_then_global_mean"):
            raise ValueError(f"invalid replay_estimand.aggregation: {agg!r}")
        tk = str(d.get("target_kpi_column", "")).strip()
        if not tk:
            raise ValueError("replay_estimand.target_kpi_column is required")
        ls = str(d.get("lift_scale", "")).strip()
        if not ls:
            raise ValueError("replay_estimand.lift_scale is required")
        if "week_start" not in d or "week_end" not in d:
            raise ValueError("replay_estimand.week_start and week_end are required")
        return cls(
            geo_scope=gs,  # type: ignore[arg-type]
            geo_ids=tuple(str(x) for x in gids),
            week_start=d["week_start"],
            week_end=d["week_end"],
            aggregation=agg,  # type: ignore[arg-type]
            target_kpi_column=tk,
            lift_scale=ls,
            notes=str(d.get("notes", "")),
        )


def _week_mask(series: pd.Series, start: Any, end: Any) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        a, b = float(start), float(end)
        return ((series.to_numpy(dtype=float) >= a) & (series.to_numpy(dtype=float) <= b)).astype(bool)
    s = pd.to_datetime(series, errors="coerce")
    a = pd.to_datetime(start, errors="coerce")
    b = pd.to_datetime(end, errors="coerce")
    return ((s >= a) & (s <= b)).fillna(False).to_numpy()


def eval_mask_for_replay(panel: pd.DataFrame, schema: PanelSchema, spec: ReplayEstimandSpec) -> np.ndarray:
    """Boolean mask aligned to ``panel`` rows (caller must ensure panel index alignment)."""
    gcol, wcol = schema.geo_column, schema.week_column
    m = np.ones(len(panel), dtype=bool)
    if spec.geo_scope == "listed":
        m &= panel[gcol].astype(str).isin(set(spec.geo_ids)).to_numpy()
    m &= _week_mask(panel[wcol], spec.week_start, spec.week_end)
    return m


def aggregate_level_delta_masked(
    yhat_obs: np.ndarray,
    yhat_cf: np.ndarray,
    panel: pd.DataFrame,
    schema: PanelSchema,
    spec: ReplayEstimandSpec,
) -> tuple[float, dict[str, Any]]:
    """``yhat_*`` level KPI predictions; returns aggregated implied lift per ``spec.aggregation``."""
    mask = eval_mask_for_replay(panel, schema, spec)
    if not np.any(mask):
        raise ValueError("replay estimand produced empty eval_mask; check week/geo window")
    d = yhat_obs[mask] - yhat_cf[mask]
    meta: dict[str, Any] = {"n_eval_rows": int(mask.sum()), "aggregation": spec.aggregation}
    if spec.aggregation == "mean":
        return float(np.mean(d)), meta
    if spec.aggregation == "sum":
        return float(np.sum(d)), meta
    gcol = schema.geo_column
    locs = np.where(mask)[0]
    deltas = yhat_obs[locs] - yhat_cf[locs]
    geos = panel.iloc[locs][gcol].astype(str).to_numpy()
    geo_deltas: dict[str, list[float]] = {}
    for i, g in enumerate(geos):
        geo_deltas.setdefault(str(g), []).append(float(deltas[i]))
    per_geo_means = [float(np.mean(v)) for v in geo_deltas.values()]
    meta["n_geos"] = len(per_geo_means)
    return float(np.mean(per_geo_means)), meta
