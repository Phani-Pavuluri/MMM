"""Optimize spend using a serialized single-channel curve bundle (Sprint 5)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from mmm.economics.canonical import assert_planner_scope_supported, validate_optimizer_objective_key


def optimize_spend_from_curve_bundle(
    bundle: dict,
    *,
    current_spend: float,
    total_budget: float,
    spend_min: float,
    spend_max: float,
) -> dict:
    """
    Maximize interpolated ``response_on_modeling_scale`` under box + sum(spend)=budget.

    Bundle must contain ``spend_grid`` and ``response_on_modeling_scale`` (Sprint 5 artifact).
    """
    grid = np.asarray(bundle["spend_grid"], dtype=float)
    resp = np.asarray(bundle["response_on_modeling_scale"], dtype=float)
    if grid.size < 2 or resp.size != grid.size:
        raise ValueError("Invalid curve_bundle: need aligned spend_grid and response")
    f = interp1d(grid, resp, kind="linear", fill_value="extrapolate", bounds_error=False)

    x0 = np.clip(current_spend, spend_min, spend_max)

    def neg_obj(x: np.ndarray) -> float:
        return -float(f(np.clip(x[0], grid.min(), grid.max())))

    bounds = [(spend_min, spend_max)]
    res = minimize(neg_obj, x0=np.array([x0]), method="L-BFGS-B", bounds=bounds)
    xs = float(np.clip(res.x[0], spend_min, spend_max))
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "optimal_spend": xs,
        "objective_at_optimum": float(f(xs)),
        "total_budget_hint": float(total_budget),
        "source": "curve_bundle_interpolated_single_channel",
    }


def _make_response_interp(bundle: dict):
    return _make_bundle_interp(bundle, "response_on_modeling_scale")


def _make_bundle_interp(bundle: dict, value_key: str):
    grid = np.asarray(bundle["spend_grid"], dtype=float)
    if value_key not in bundle:
        raise KeyError(f"curve_bundle missing {value_key!r} for optimizer objective")
    resp = np.asarray(bundle[value_key], dtype=float)
    if grid.size < 2 or resp.size != grid.size:
        raise ValueError("Invalid curve_bundle: need aligned spend_grid and response series")
    f = interp1d(grid, resp, kind="linear", fill_value="extrapolate", bounds_error=False)
    gmin, gmax = float(grid.min()), float(grid.max())

    def value_at(spend: float) -> float:
        return float(f(np.clip(spend, gmin, gmax)))

    return value_at


def optimize_budget_from_curve_bundles(
    channel_names: list[str],
    bundles: list[dict],
    *,
    current_spend: np.ndarray,
    total_budget: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
    objective_value_key: str = "response_on_modeling_scale",
    economics_contract: dict | None = None,
) -> dict:
    """
    Multi-channel spend allocation: maximize sum of interpolated values on each bundle's spend grid.

    ``objective_value_key`` selects which series to optimize (default: modeling-scale partial
    contribution). Use ``kpi_level_implied_by_partial_curve`` **only** when every bundle includes that
    key from the anchored economics path (same definition as curve artifacts).

    When ``economics_contract`` is provided (e.g. from curve artifacts), objective keys and planner
    scope are validated against the canonical contract.
    """
    if economics_contract is not None:
        assert_planner_scope_supported(economics_contract)
        validate_optimizer_objective_key(objective_value_key, economics_contract)
    n = len(channel_names)
    if len(bundles) != n:
        raise ValueError("bundles length must match channel_names")
    if current_spend.shape != (n,) or channel_min.shape != (n,) or channel_max.shape != (n,):
        raise ValueError("current_spend, channel_min, channel_max must shape (n_channels,)")
    lo_sum = float(np.sum(channel_min))
    hi_sum = float(np.sum(channel_max))
    if lo_sum > total_budget + 1e-5:
        raise ValueError(f"total_budget {total_budget} is below sum(channel_min)={lo_sum}")
    if hi_sum < total_budget - 1e-5:
        raise ValueError(f"total_budget {total_budget} exceeds sum(channel_max)={hi_sum}")

    interps = [_make_bundle_interp(b, objective_value_key) for b in bundles]

    x0 = np.clip(current_spend.astype(float), channel_min, channel_max)
    x0 = x0 * (total_budget / max(float(x0.sum()), 1e-12))
    x0 = np.clip(x0, channel_min, channel_max)
    for _ in range(n * 20):
        gap = total_budget - float(x0.sum())
        if abs(gap) < 1e-6:
            break
        if gap > 0:
            slack = channel_max - x0
            tot = float(slack.sum())
            if tot > 1e-12:
                x0 = np.clip(x0 + slack * (gap / tot), channel_min, channel_max)
            else:
                break
        else:
            slack = x0 - channel_min
            tot = float(slack.sum())
            if tot > 1e-12:
                x0 = np.clip(x0 - slack * ((-gap) / tot), channel_min, channel_max)
            else:
                break

    def neg_total(x: np.ndarray) -> float:
        return -float(sum(interps[i](float(x[i])) for i in range(n)))

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]
    bounds = [(float(lo), float(hi)) for lo, hi in zip(channel_min, channel_max, strict=True)]
    res = minimize(neg_total, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    opt_x = np.clip(res.x, channel_min, channel_max)
    obj = float(sum(interps[i](float(opt_x[i])) for i in range(n)))
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "optimal_spend": {c: float(x) for c, x in zip(channel_names, opt_x, strict=True)},
        "objective_at_optimum": obj,
        "total_budget": float(total_budget),
        "source": "curve_bundles_interpolated_multichannel",
        "objective_value_key": objective_value_key,
    }
