"""E19: heuristic framework / spec recommendations."""

from __future__ import annotations

from typing import Any


def recommend_configuration(
    n_rows: int,
    n_channels: int,
    identifiability_risk: float,
    compute_budget: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if n_rows < 30 * max(n_channels, 1):
        out["framework"] = "ridge_bo"
        out["rationale"] = "history per channel-dimension is limited; BO+ridge explores transforms safely"
    else:
        out["framework"] = "bayesian"
        out["rationale"] = "enough rows to consider full posterior uncertainty"
    if identifiability_risk > 0.55:
        out["pooling"] = "partial"
        out["rationale2"] = "elevated identifiability risk; shrinkage across geos helps"
    out["model_form"] = "semi_log"
    if compute_budget == "low":
        out["framework"] = "ridge_bo"
        out["compute"] = "forced ridge_bo for low compute budget"
    return out
