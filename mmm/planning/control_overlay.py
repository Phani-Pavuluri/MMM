"""Non-spend scenario overlays (promos, holidays, controls) for full-panel μ simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema


@dataclass(frozen=True)
class ControlOverlaySpec:
    """
    Sparse overrides matched on ``(geo_column, week_column)``.

    Each row: ``{"geo": ..., "week": ..., "column": "control_name", "value": float}``.
    """

    rows: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ControlOverlaySpec:
        ov = raw.get("overrides") or raw.get("rows")
        if not isinstance(ov, list) or not ov:
            raise ValueError("control_overlay requires non-empty overrides list")
        rows: list[dict[str, Any]] = []
        for i, r in enumerate(ov):
            if not isinstance(r, dict):
                raise ValueError(f"override {i} must be a mapping")
            for k in ("geo", "week", "column", "value"):
                if k not in r:
                    raise ValueError(f"override {i} missing {k!r}")
            rows.append(dict(r))
        return cls(rows=tuple(rows))


def apply_control_overlay(
    panel: pd.DataFrame,
    schema: PanelSchema,
    overlay: ControlOverlaySpec | None,
) -> pd.DataFrame:
    """Return a sorted copy of ``panel`` with overlay values written in-place on matched rows."""
    if overlay is None or not overlay.rows:
        return sort_panel_for_modeling(panel.copy(), schema)
    out = sort_panel_for_modeling(panel.copy(), schema)
    gcol, wcol = schema.geo_column, schema.week_column
    wseries = out[wcol]
    for r in overlay.rows:
        col = str(r["column"])
        if col not in out.columns:
            raise ValueError(f"control_overlay column {col!r} not in panel columns")
        geo = str(r["geo"])
        week = r["week"]
        val = float(r["value"])
        m_geo = out[gcol].astype(str) == geo
        if pd.api.types.is_numeric_dtype(wseries):
            m_w = wseries.astype(float) == float(week)
        else:
            wk = pd.to_datetime(wseries, errors="coerce")
            wv = pd.to_datetime(week, errors="coerce")
            m_w = wk == wv
        m = m_geo & m_w
        if not bool(m.any()):
            raise ValueError(f"control_overlay matched no rows for geo={geo!r} week={week!r} column={col!r}")
        out.loc[m, col] = val
    return out
