"""Dynamic (non–constant-in-time) spend paths for full-panel counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema


@dataclass(frozen=True)
class SpendSegment:
    """Inclusive ``[week_start, week_end]`` window (numeric or date-like, same dtype as panel week column)."""

    week_start: Any
    week_end: Any
    spend_by_channel: dict[str, float]


@dataclass(frozen=True)
class PiecewiseSpendPath:
    """Piecewise-constant channel spend by calendar week segment (later segments override overlaps)."""

    kind: Literal["piecewise_calendar_week"] = "piecewise_calendar_week"
    segments: tuple[SpendSegment, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PiecewiseSpendPath:
        if str(raw.get("kind", "piecewise_calendar_week")) != "piecewise_calendar_week":
            raise ValueError("only kind=piecewise_calendar_week is supported for spend paths")
        segs = raw.get("segments") or []
        if not isinstance(segs, list) or not segs:
            raise ValueError("segments must be a non-empty list")
        out: list[SpendSegment] = []
        for i, s in enumerate(segs):
            if not isinstance(s, dict):
                raise ValueError(f"segment {i} must be a mapping")
            if "week_start" not in s or "week_end" not in s:
                raise ValueError(f"segment {i} requires week_start and week_end")
            sp = s.get("spend_by_channel") or s.get("candidate_spend")
            if not isinstance(sp, dict) or not sp:
                raise ValueError(f"segment {i} requires spend_by_channel (or candidate_spend)")
            out.append(
                SpendSegment(
                    week_start=s["week_start"],
                    week_end=s["week_end"],
                    spend_by_channel={str(k): float(v) for k, v in sp.items()},
                )
            )
        return cls(segments=tuple(out))


def counterfactual_piecewise_spend_panel(
    panel: pd.DataFrame,
    schema: PanelSchema,
    path: PiecewiseSpendPath,
) -> pd.DataFrame:
    """
    Copy panel with channel spends overwritten row-wise by the last matching segment.

    Week filtering uses the same numeric / datetime coercion rules as replay estimands.
    """
    out = sort_panel_for_modeling(panel.copy(), schema)
    wcol = schema.week_column
    wseries = out[wcol]
    for ch in schema.channel_columns:
        out[ch] = out[ch].astype(float)
    for seg in path.segments:
        mask = _week_between(wseries, seg.week_start, seg.week_end)
        for ch, val in seg.spend_by_channel.items():
            if ch not in schema.channel_columns:
                continue
            out.loc[mask, ch] = float(val)
    return out


def _week_between(series: pd.Series, start: Any, end: Any) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        a, b = float(start), float(end)
        return (series.astype(float) >= a) & (series.astype(float) <= b)
    s = pd.to_datetime(series, errors="coerce")
    a = pd.to_datetime(start, errors="coerce")
    b = pd.to_datetime(end, errors="coerce")
    return (s >= a) & (s <= b)


def time_mean_spend_by_channel(panel_cf: pd.DataFrame, schema: PanelSchema) -> dict[str, float]:
    """Average spend per channel over rows (used for budget accounting vs constant vectors)."""
    return {ch: float(panel_cf[ch].astype(float).mean()) for ch in schema.channel_columns}
