"""Risk-aware budget optimization using posterior Δμ draws (linear Ridge design matrix)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from mmm.config.schema import RunEnvironment
from mmm.economics.canonical import build_economics_contract
from mmm.planning.baseline import BaselinePlan, bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate
from mmm.planning.posterior_planning import (
    RiskObjective,
    assert_posterior_planning_allowed,
    delta_mu_draws_hierarchical_geo_beta,
    delta_mu_draws_linear_ridge,
    risk_objective_scalar,
)


def optimize_budget_risk_aware(
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None = None,
    current_spend: np.ndarray,
    total_budget: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
    bayesian_fit_meta: dict[str, Any],
    linear_coef_draws: np.ndarray | None = None,
    hierarchical_draw_pack: dict[str, Any] | None = None,
    intercept_draws: np.ndarray | None = None,
    objective: RiskObjective = "p50",
    cvar_alpha: float = 0.1,
    risk_lambda: float = 1.0,
    max_draws_per_eval: int | None = 200,
) -> dict[str, Any]:
    """
    Maximize a **risk-aware** scalar derived from posterior **Δμ draws** (same μ path as
    :func:`mmm.planning.posterior_planning.simulate_posterior`).

    Pass **exactly one** of ``linear_coef_draws`` (global) or ``hierarchical_draw_pack`` (per-geo PyMC export).
    """
    if (linear_coef_draws is None) == (hierarchical_draw_pack is None):
        raise ValueError("provide exactly one of linear_coef_draws or hierarchical_draw_pack")
    if ctx.config.budget.geo_budget_enabled:
        raise NotImplementedError(
            "optimize_budget_risk_aware does not support budget.geo_budget_enabled; use global optimization."
        )
    assert_posterior_planning_allowed(
        ctx.config,
        bayesian_fit_meta,
        linear_coef_draws,
        hierarchical_draw_pack=hierarchical_draw_pack,
    )

    names = list(ctx.schema.channel_columns)
    n = len(names)
    if current_spend.shape != (n,) or channel_min.shape != (n,) or channel_max.shape != (n,):
        raise ValueError("current_spend, channel_min, channel_max must align with channel_columns")

    base = baseline_plan or bau_baseline_from_panel(ctx.panel, ctx.schema)
    base_vec = np.array([float(base.spend_by_channel[c]) for c in names], dtype=float)

    lo_sum = float(np.sum(channel_min))
    hi_sum = float(np.sum(channel_max))
    if lo_sum > total_budget + 1e-5:
        raise ValueError(f"total_budget {total_budget} below sum(channel_min)={lo_sum}")
    if hi_sum < total_budget - 1e-5:
        raise ValueError(f"total_budget {total_budget} exceeds sum(channel_max)={hi_sum}")

    agg = ctx.config.extensions.product.planning_delta_mu_aggregation

    def spend_dict(x: np.ndarray) -> dict[str, float]:
        return {names[i]: float(x[i]) for i in range(n)}

    def delta_triple(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sp = spend_dict(x)
        if hierarchical_draw_pack is not None:
            return delta_mu_draws_hierarchical_geo_beta(
                ctx,
                baseline_plan=base,
                spend_plan=sp,
                hierarchical_draw_pack=hierarchical_draw_pack,
                spend_path_plan=None,
                spend_plan_geo=None,
                delta_mu_aggregation=agg,
                max_draws=max_draws_per_eval,
            )
        return delta_mu_draws_linear_ridge(
            ctx,
            baseline_plan=base,
            spend_plan=sp,
            linear_coef_draws=linear_coef_draws,  # type: ignore[arg-type]
            intercept_draws=intercept_draws,
            spend_path_plan=None,
            spend_plan_geo=None,
            delta_mu_aggregation=agg,
            max_draws=max_draws_per_eval,
        )

    def neg_risk(x: np.ndarray) -> float:
        _, _, dlt = delta_triple(x)
        val = risk_objective_scalar(dlt, objective, cvar_alpha=cvar_alpha, risk_lambda=risk_lambda)
        return -float(val)

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

    cons = [{"type": "eq", "fun": lambda xv: float(np.sum(xv)) - float(total_budget)}]
    bounds = [(float(lo), float(hi)) for lo, hi in zip(channel_min, channel_max, strict=True)]
    res = minimize(neg_risk, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    opt_x = np.clip(res.x, channel_min, channel_max)
    _, _, dlt_opt = delta_triple(opt_x)
    final_sim = simulate(
        spend_dict(opt_x),
        ctx,
        baseline_plan=base,
        uncertainty_mode="point",
        delta_mu_aggregation=agg,
    )
    p10, p50, p90 = (float(x) for x in np.percentile(dlt_opt, [10, 50, 90]))
    risk_val = risk_objective_scalar(dlt_opt, objective, cvar_alpha=cvar_alpha, risk_lambda=risk_lambda)
    ec = build_economics_contract(ctx.config)
    draw_source = "hierarchical_draw_pack" if hierarchical_draw_pack is not None else "linear_coef_draws"
    prod = ctx.config.run_environment == RunEnvironment.PROD
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "recommended_spend_plan": {c: float(x) for c, x in zip(names, opt_x, strict=True)},
        "objective": objective,
        "risk_objective_value": float(risk_val),
        "objective_delta_mu_point": float(final_sim.delta_mu),
        "posterior_delta_mu_p10": p10,
        "posterior_delta_mu_p50": p50,
        "posterior_delta_mu_p90": p90,
        "simulation_at_recommendation": final_sim.to_json(),
        "source": "full_model_simulation_slsqp_risk_aware_draws",
        "baseline_type": base.baseline_type.value,
        "baseline_spend_reference": {c: float(base_vec[i]) for i, c in enumerate(names)},
        "cvar_alpha": float(cvar_alpha),
        "risk_lambda": float(risk_lambda),
        "max_draws_per_eval": max_draws_per_eval,
        "draw_mode": "hierarchical" if hierarchical_draw_pack is not None else "linear",
        "posterior_planning_metadata": {
            "posterior_planning_mode": "draw_based_approximation",
            "decision_safe": not prod,
            "economics_contract_version": str(ec.get("contract_version", "")),
            "draw_source_artifact": draw_source,
            "canonical_contract_note": (
                "Each SLSQP objective evaluation uses precomputed Δμ draws from delta_mu_draws_* "
                "(same μ / design-matrix path as training and as mmm.planning.decision_simulate.simulate "
                "for point Δμ). This is a performance-certified draw layer, not a second economics definition."
            ),
        },
    }
