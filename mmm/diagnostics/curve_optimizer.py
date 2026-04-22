"""Curve-interpolation optimizers — diagnostics / research only; clamped support; never PROD."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.economics.canonical import assert_planner_scope_supported, validate_optimizer_objective_key


def _assert_curve_optimizer_allowed(config: MMMConfig) -> None:
    if config.run_environment == RunEnvironment.PROD:
        raise PermissionError("Curve-bundle optimizers are disabled in run_environment=prod")
    if not config.allow_unsafe_decision_apis:
        raise PermissionError(
            "Curve-bundle optimizers require allow_unsafe_decision_apis=True (diagnostics / research only)"
        )


def _make_bundle_interp_clamped(bundle: dict, value_key: str):
    grid = np.asarray(bundle["spend_grid"], dtype=float)
    if value_key not in bundle:
        raise KeyError(f"curve_bundle missing {value_key!r} for optimizer objective")
    resp = np.asarray(bundle[value_key], dtype=float)
    if grid.size < 2 or resp.size != grid.size:
        raise ValueError("Invalid curve_bundle: need aligned spend_grid and response series")
    gmin, gmax = float(grid.min()), float(grid.max())
    order = np.argsort(grid)
    g_sorted = grid[order]
    r_sorted = resp[order]

    def value_at(spend: float) -> tuple[float, bool]:
        x = float(spend)
        clipped = x < gmin - 1e-12 or x > gmax + 1e-12
        xc = float(np.clip(x, gmin, gmax))
        y = float(np.interp(xc, g_sorted, r_sorted))
        return y, clipped

    return value_at, gmin, gmax


def optimize_spend_from_curve_bundle(
    bundle: dict,
    *,
    config: MMMConfig,
    current_spend: float,
    total_budget: float,
    spend_min: float,
    spend_max: float,
) -> dict:
    """Maximize clamped-linear interpolated response — ``decision_safe`` is False (diagnostic surface)."""
    _assert_curve_optimizer_allowed(config)
    grid = np.asarray(bundle["spend_grid"], dtype=float)
    resp = np.asarray(bundle["response_on_modeling_scale"], dtype=float)
    if grid.size < 2 or resp.size != grid.size:
        raise ValueError("Invalid curve_bundle: need aligned spend_grid and response")
    value_at, gmin, gmax = _make_bundle_interp_clamped(bundle, "response_on_modeling_scale")

    x0 = float(np.clip(current_spend, spend_min, spend_max))

    clipped_any = False

    def neg_obj(x: np.ndarray) -> float:
        nonlocal clipped_any
        v, cl = value_at(float(x[0]))
        if cl:
            clipped_any = True
        return -v

    bounds = [(spend_min, spend_max)]
    res = minimize(neg_obj, x0=np.array([x0]), method="L-BFGS-B", bounds=bounds)
    xs = float(np.clip(res.x[0], spend_min, spend_max))
    v_fin, cl_fin = value_at(xs)
    if cl_fin:
        clipped_any = True
    return {
        "success": bool(res.success),
        "optimizer_success": bool(res.success),
        "decision_safe": False,
        "stability_score": 0.0,
        "num_restarts": 1,
        "message": str(res.message),
        "optimal_spend": xs,
        "objective_at_optimum": float(v_fin),
        "total_budget_hint": float(total_budget),
        "source": "curve_bundle_interpolated_single_channel_clamped",
        "curve_support_clipped": bool(clipped_any),
    }


def optimize_budget_from_curve_bundles(
    channel_names: list[str],
    bundles: list[dict],
    *,
    config: MMMConfig,
    current_spend: np.ndarray,
    total_budget: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
    objective_value_key: str = "response_on_modeling_scale",
    economics_contract: dict | None = None,
) -> dict:
    """Multi-channel curve interpolation optimizer — clamped support; ``decision_safe`` is False."""
    _assert_curve_optimizer_allowed(config)
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

    interps = [_make_bundle_interp_clamped(b, objective_value_key) for b in bundles]
    clipped_any = False

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
        nonlocal clipped_any
        s = 0.0
        for i in range(n):
            v, cl = interps[i][0](float(x[i]))
            if cl:
                clipped_any = True
            s += v
        return -s

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]
    bounds = [(float(lo), float(hi)) for lo, hi in zip(channel_min, channel_max, strict=True)]
    res = minimize(neg_total, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    opt_x = np.clip(res.x, channel_min, channel_max)
    obj = 0.0
    for i in range(n):
        v, _ = interps[i][0](float(opt_x[i]))
        obj += v
    return {
        "success": bool(res.success),
        "optimizer_success": bool(res.success),
        "decision_safe": False,
        "stability_score": 0.0,
        "num_restarts": 1,
        "message": str(res.message),
        "optimal_spend": {c: float(x) for c, x in zip(channel_names, opt_x, strict=True)},
        "objective_at_optimum": float(obj),
        "total_budget": float(total_budget),
        "source": "curve_bundles_interpolated_multichannel_clamped",
        "objective_value_key": objective_value_key,
        "curve_support_clipped": bool(clipped_any),
    }
