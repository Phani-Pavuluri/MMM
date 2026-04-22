"""ROI / mROAS summaries from extension curve bundles (Sprint 6 / reporting)."""

from __future__ import annotations

from typing import Any


def curve_bundles_to_roi_summary(curve_bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-channel spend grid midpoints and level-space mROAS proxy at mid grid (reporting only)."""
    rows: list[dict[str, Any]] = []
    for b in curve_bundles:
        ch = b.get("channel", "?")
        grid = b.get("spend_grid") or []
        mroas_cons = b.get("mroas_level_consistent")
        mroas_lin = b.get("mroas_level_proxy")
        if not isinstance(grid, list) or len(grid) < 1:
            continue
        mid = float(grid[len(grid) // 2])
        m_cons_mid: float | None = None
        m_lin_mid: float | None = None
        if isinstance(mroas_cons, list) and len(mroas_cons) == len(grid):
            m_cons_mid = float(mroas_cons[len(mroas_cons) // 2])
        if isinstance(mroas_lin, list) and len(mroas_lin) == len(grid):
            m_lin_mid = float(mroas_lin[len(mroas_lin) // 2])
        rows.append(
            {
                "channel": ch,
                "spend_mid_grid": mid,
                "mroas_level_consistent_mid_grid": m_cons_mid,
                "mroas_level_proxy_mid_grid": m_lin_mid,
                "mroas_preferred_mid_grid": m_cons_mid if m_cons_mid is not None else m_lin_mid,
                "y_level_scale": (b.get("roi_bridge") or {}).get("y_level_scale"),
            }
        )
    return {"channels": rows, "source": "curve_bundles_to_roi_summary"}
