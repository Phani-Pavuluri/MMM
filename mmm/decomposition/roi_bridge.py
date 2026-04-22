"""Level-space marginal revenue / mROAS proxy from modeling-scale curves (Sprint 6).

For semi-log and log-log MMMs the conditional mean is ``E[y|X] ≈ exp(μ)`` with ``μ`` additive in
transformed media. Holding other inputs fixed, ``d(log y)/dS`` along one channel equals the
curve marginal ``d(β·x(S))/dS``. Then ``dy/dS ≈ y · d(log y)/dS``.

We approximate ``y`` with ``y_level_scale`` (typically mean observed or fitted KPI in levels).
This is a **first-order local** bridge; cross-effects and intercept shifts are ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def marginal_revenue_and_mroas_level_proxy(
    *,
    marginal_roi_modeling: np.ndarray,
    model_form: str,
    y_level_scale: float,
) -> dict[str, Any]:
    if y_level_scale <= 0:
        raise ValueError("y_level_scale must be positive for level-space bridge")
    m = np.asarray(marginal_roi_modeling, dtype=float)
    y = float(y_level_scale)
    # Same chain rule for semi_log (log link on y) and log_log (log y, log media); marginal on
    # curve is always d(portion of μ)/dS in the Gaussian mean on the log scale.
    marginal_revenue = y * m
    return {
        "marginal_revenue_level_proxy": marginal_revenue.tolist(),
        "mroas_level_proxy": marginal_revenue.tolist(),
        "roi_bridge": {
            "model_form": model_form,
            "y_level_scale": y,
            "formula": "marginal_revenue_level_proxy = y_level_scale * marginal_roi_modeling",
            "interpretation": (
                "dy/dS ≈ y * d(log y)/dS; y approximated by y_level_scale. Spend and KPI must use "
                "consistent currency units for mROAS to be interpretable as revenue per spend."
            ),
        },
    }


def consistent_level_kpi_and_mroas(
    spend_grid: np.ndarray,
    response_modeling: np.ndarray,
    y_anchor: float,
    *,
    anchor_spend: float | None = None,
) -> dict[str, Any]:
    """
    Calibrate ``Y(S)=exp(μ_rest + r(S))`` so ``Y`` matches ``y_anchor`` at the anchor spend on the grid.

    Here ``r(S)`` is the modeled partial contribution ``β·x(S)`` on the log-mean scale (same units as
    ``response_on_modeling_scale`` in curve artifacts). ``mroas_level_consistent`` is ``dY/dS`` along
    that partial curve — **stronger** than the global linear proxy when interpreting a single channel
    in isolation (still ignores cross-channel feedback in μ).
    """
    g = np.asarray(spend_grid, dtype=float)
    r = np.asarray(response_modeling, dtype=float)
    if g.size != r.size or g.size < 2:
        raise ValueError("spend_grid and response_modeling must align and have length >= 2")
    idx = (
        int(np.argmin(np.abs(g - float(anchor_spend))))
        if anchor_spend is not None
        else int(len(r) // 2)
    )
    y0 = float(max(y_anchor, 1e-12))
    mu_rest = float(np.log(y0) - r[idx])
    y_level = np.exp(mu_rest + r)
    edge = 2 if g.size >= 3 else 1
    mroas_c = np.gradient(y_level, g, edge_order=edge)
    return {
        "kpi_level_implied_by_partial_curve": y_level.tolist(),
        "mroas_level_consistent": mroas_c.tolist(),
        "roi_bridge_consistent": {
            "method": "exp(mu_rest + r(S)); mu_rest=log(y_anchor)-r(S_anchor)",
            "anchor_spend": float(g[idx]),
            "y_anchor": y0,
        },
    }


def attach_level_roi_to_curve_artifact(
    artifact: dict[str, Any],
    *,
    y_level_scale: float,
    target_column: str,
    anchor_spend: float | None = None,
) -> dict[str, Any]:
    """Mutate-copy a curve bundle dict with level-space marginal fields (linear proxy + consistent curve)."""
    mmod = np.asarray(artifact["marginal_roi_modeling_scale"], dtype=float)
    mf = str(artifact.get("model_form", "semi_log"))
    extra = marginal_revenue_and_mroas_level_proxy(
        marginal_roi_modeling=mmod,
        model_form=mf,
        y_level_scale=y_level_scale,
    )
    out = {**artifact, **extra}
    out["roi_bridge"]["target_kpi_column"] = target_column
    g = np.asarray(artifact["spend_grid"], dtype=float)
    r = np.asarray(artifact["response_on_modeling_scale"], dtype=float)
    cons = consistent_level_kpi_and_mroas(g, r, y_level_scale, anchor_spend=anchor_spend)
    out["kpi_level_implied_by_partial_curve"] = cons["kpi_level_implied_by_partial_curve"]
    out["mroas_level_consistent"] = cons["mroas_level_consistent"]
    out["roi_bridge"]["consistent_partial_curve"] = cons["roi_bridge_consistent"]
    out["marginal_roi_definition"] = (
        "marginal_roi_modeling_scale: d(β·x)/d(spend) on log-mean scale; "
        "mroas_level_proxy: linear dy/dS ≈ y_anchor * marginal_modeling (global scale); "
        "mroas_level_consistent: d/dS exp(mu_rest + β·x(S)) anchored so Y matches y_anchor at anchor spend."
    )
    return out
