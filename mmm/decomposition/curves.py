"""Response and marginal ROI curves from transformed spend grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation


@dataclass
class ResponseCurve:
    spend_grid: np.ndarray
    response: np.ndarray
    marginal_roi: np.ndarray


def validate_response_curve_diagnostics(curve: ResponseCurve, *, min_grid_points: int = 5) -> list[str]:
    """
    Non-authoritative validity checks for curve objects (diagnostic / research surfaces).

    Returns human-readable issue codes (empty list if checks pass).
    """
    issues: list[str] = []
    g = np.asarray(curve.spend_grid, dtype=float)
    if g.size < min_grid_points:
        issues.append(f"sparse_spend_grid:n={g.size}")
    if np.any(~np.isfinite(curve.response)) or np.any(~np.isfinite(curve.marginal_roi)):
        issues.append("non_finite_response_or_mroi")
    if g.size >= 2 and not np.all(np.diff(g) > 0):
        issues.append("spend_grid_not_strictly_increasing")
    if curve.response.size >= 3:
        dr = np.diff(curve.response)
        sign_changes = int(np.sum(dr[1:] * dr[:-1] < 0))
        if sign_changes > 2:
            issues.append("response_curve_highly_non_monotone")
    m = np.asarray(curve.marginal_roi, dtype=float)
    if m.size and float(np.nanmax(np.abs(m))) > 1e6 * (float(np.nanmax(np.abs(curve.response))) + 1.0):
        issues.append("marginal_increment_exploded_vs_response_scale")
    return issues


def _response_curve_core(
    spend_grid: np.ndarray,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
    beta: float,
    model_form: str,
    horizon_weeks: int = 52,
) -> ResponseCurve:
    ad = GeometricAdstock(decay)
    sat = HillSaturation(half_max=hill_half, slope=hill_slope)
    resp = []
    for s in spend_grid:
        series = np.full(max(2, int(horizon_weeks)), float(s))
        x = apply_adstock_saturation_series(series, ad, sat)[-1]
        if model_form == "log_log":
            from mmm.utils.math import safe_log

            x = float(safe_log(np.array([x]))[0])
        else:
            x = float(x)
        resp.append(beta * x)
    resp = np.asarray(resp)
    edge = 2 if spend_grid.size >= 3 else 1
    mroi = np.gradient(resp, spend_grid, edge_order=edge)
    return ResponseCurve(spend_grid=spend_grid, response=resp, marginal_roi=mroi)


def build_curve_for_channel(
    spend_grid: np.ndarray,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
    beta: float,
    model_form: str,
    horizon_weeks: int = 52,
) -> ResponseCurve:
    """
    Canonical strict curve builder for artifacted / decision-adjacent surfaces.

    Fails fast on sparse grids or failed sanity diagnostics.
    """
    curve = _response_curve_core(
        spend_grid,
        decay=decay,
        hill_half=hill_half,
        hill_slope=hill_slope,
        beta=beta,
        model_form=model_form,
        horizon_weeks=horizon_weeks,
    )
    diag = validate_response_curve_diagnostics(curve)
    if diag:
        raise ValueError("response_curve_validation_failed: " + "; ".join(diag))
    return curve


def build_curve_for_channel_research_only(
    spend_grid: np.ndarray,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
    beta: float,
    model_form: str,
    horizon_weeks: int = 52,
    bypass_reason: str,
) -> tuple[ResponseCurve, dict]:
    """
    Explicit **non-canonical** curve builder for notebooks / diagnostics.

    Does **not** enforce strict grid-density rules. Returns the curve plus a typed quantity envelope
    (``CurveQuantityResult.section_dict()``) that is **never** decision-safe and must not feed
    decision bundles or prod optimizers.
    """
    from mmm.contracts.quantity_models import CurveQuantityResult

    curve = _response_curve_core(
        spend_grid,
        decay=decay,
        hill_half=hill_half,
        hill_slope=hill_slope,
        beta=beta,
        model_form=model_form,
        horizon_weeks=horizon_weeks,
    )
    issues = validate_response_curve_diagnostics(curve, min_grid_points=2)
    qty = CurveQuantityResult(
        spend_grid=np.asarray(curve.spend_grid, dtype=float).tolist(),
        response_on_modeling_scale=np.asarray(curve.response, dtype=float).tolist(),
        marginal_roi_modeling_scale=np.asarray(curve.marginal_roi, dtype=float).tolist(),
        validity_diagnostics={
            "strict_checks_relaxed": True,
            "bypass_reason": bypass_reason,
            "diagnostic_issues": issues,
        },
        non_canonical_builder="research_only_sparse_or_relaxed",
    )
    return curve, qty.section_dict()
