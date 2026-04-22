"""Budget optimization via full-panel simulate() — objective is Δμ (not curve interpolation)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from mmm.planning.baseline import (
    BaselinePlan,
    bau_baseline_from_panel,
    bau_baseline_per_geo_from_panel,
    channel_means_from_geo_plan,
    disclosure_for_non_bau_optimization,
)
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate


def _geo_list(panel, schema) -> list[str]:
    return sorted({str(x) for x in panel[schema.geo_column].unique()})


def _flat_to_geo_dict(x: np.ndarray, geos: list[str], names: list[str]) -> dict[str, dict[str, float]]:
    n_c = len(names)
    out: dict[str, dict[str, float]] = {}
    k = 0
    for g in geos:
        out[g] = {names[i]: float(x[k + i]) for i in range(n_c)}
        k += n_c
    return out


def _optimize_budget_geo(
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None,
    total_budget: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
) -> dict[str, Any]:
    bcfg = ctx.config.budget
    names = list(ctx.schema.channel_columns)
    geos = _geo_list(ctx.panel, ctx.schema)
    n_g, n_c = len(geos), len(names)
    dim = n_g * n_c
    if n_g == 0 or n_c == 0:
        raise ValueError("geo budget optimization requires non-empty geos and channels")

    base = baseline_plan or bau_baseline_per_geo_from_panel(ctx.panel, ctx.schema)
    lo = np.zeros(dim, dtype=float)
    hi = np.zeros(dim, dtype=float)
    for gi, g in enumerate(geos):
        for ci, c in enumerate(names):
            idx = gi * n_c + ci
            lo[idx] = float(bcfg.geo_channel_min.get(g, {}).get(c, float(channel_min[ci])))
            hi[idx] = float(bcfg.geo_channel_max.get(g, {}).get(c, float(channel_max[ci])))

    def x0_vector() -> np.ndarray:
        arr = np.zeros(dim, dtype=float)
        if base.spend_by_geo_channel:
            for gi, g in enumerate(geos):
                row = base.spend_by_geo_channel.get(g, {})
                for ci, c in enumerate(names):
                    arr[gi * n_c + ci] = float(row.get(c, 0.0))
        else:
            for gi in range(n_g):
                for ci, c in enumerate(names):
                    arr[gi * n_c + ci] = float(base.spend_by_channel.get(c, 0.0))
        return np.clip(arr, lo, hi)

    x0 = x0_vector()
    s0 = float(x0.sum())
    if s0 > 1e-12:
        x0 = x0 * (total_budget / s0)
    x0 = np.clip(x0, lo, hi)
    for _ in range(dim * 20):
        gap = total_budget - float(x0.sum())
        if abs(gap) < 1e-6:
            break
        if gap > 0:
            slack = hi - x0
            tot = float(slack.sum())
            if tot > 1e-12:
                x0 = np.clip(x0 + slack * (gap / tot), lo, hi)
            else:
                break
        else:
            slack = x0 - lo
            tot = float(slack.sum())
            if tot > 1e-12:
                x0 = np.clip(x0 - slack * ((-gap) / tot), lo, hi)
            else:
                break

    lo_sum = float(lo.sum())
    hi_sum = float(hi.sum())
    if lo_sum > total_budget + 1e-5:
        raise ValueError(f"total_budget {total_budget} below sum of lower bounds {lo_sum}")
    if hi_sum < total_budget - 1e-5:
        raise ValueError(f"total_budget {total_budget} exceeds sum of upper bounds {hi_sum}")

    cons: list[dict[str, Any]] = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]

    for g, floor_v in bcfg.geo_floor_total.items():
        if g not in geos:
            continue
        gi = geos.index(str(g))

        def _make_floor(gi_: int, fv: float, nc: int) -> Any:
            def f(x: np.ndarray) -> float:
                sl = x[gi_ * nc : (gi_ + 1) * nc]
                return float(np.sum(sl)) - fv

            return f

        cons.append({"type": "ineq", "fun": _make_floor(gi, float(floor_v), n_c)})

    for g, cap_v in bcfg.geo_cap_total.items():
        if g not in geos:
            continue
        gi = geos.index(str(g))

        def _make_cap(gi_: int, cv: float, nc: int) -> Any:
            def f(x: np.ndarray) -> float:
                sl = x[gi_ * nc : (gi_ + 1) * nc]
                return cv - float(np.sum(sl))

            return f

        cons.append({"type": "ineq", "fun": _make_cap(gi, float(cap_v), n_c)})

    for grp, cap_v in bcfg.geo_group_max_total.items():
        g_list = [str(gid) for gid in bcfg.geo_groups.get(grp, [])]
        idxs = [geos.index(g) for g in g_list if g in set(geos)]
        if not idxs:
            continue

        def _make_group_cap(idxs_: list[int], cv: float, nc: int) -> Any:
            def f(xv: np.ndarray) -> float:
                sm = 0.0
                for gi_ in idxs_:
                    sm += float(xv[gi_ * nc : (gi_ + 1) * nc].sum())
                return cv - sm

            return f

        cons.append({"type": "ineq", "fun": _make_group_cap(idxs, float(cap_v), n_c)})

    agg = ctx.config.extensions.product.planning_delta_mu_aggregation
    names_list = names

    def neg_delta_mu(x: np.ndarray) -> float:
        gdict = _flat_to_geo_dict(x, geos, names_list)
        sim = simulate(
            {},
            ctx,
            baseline_plan=base,
            uncertainty_mode="point",
            spend_plan_geo=gdict,
            delta_mu_aggregation=agg,
        )
        return -float(sim.delta_mu)

    bounds = [(float(lo[i]), float(hi[i])) for i in range(dim)]
    res = minimize(neg_delta_mu, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    opt_x = np.clip(res.x, lo, hi)
    opt_geo = _flat_to_geo_dict(opt_x, geos, names)
    final_sim = simulate(
        {},
        ctx,
        baseline_plan=base,
        uncertainty_mode="point",
        spend_plan_geo=opt_geo,
        delta_mu_aggregation=agg,
    )
    disc = disclosure_for_non_bau_optimization(base)
    mean_ch = channel_means_from_geo_plan(opt_geo, ctx.schema, geos)
    base_vec = np.array([float(base.spend_by_channel[c]) for c in names], dtype=float)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "recommended_spend_plan": {c: float(mean_ch[c]) for c in names},
        "recommended_spend_plan_by_geo": opt_geo,
        "objective_delta_mu": float(final_sim.delta_mu),
        "simulation_at_recommendation": final_sim.to_json(),
        "source": "full_model_simulation_slsqp_geo",
        "baseline_type": base.baseline_type.value,
        "optimization_disclosure": disc,
        "baseline_spend_reference": {c: float(base_vec[i]) for i, c in enumerate(names)},
        "product_language_note": (
            "This is a simulation-scored **per-geo** budget allocation from full-panel Δμ; "
            "constraints follow budget.geo_* config; validate before decisioning."
        ),
    }


def optimize_budget_via_simulation(
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None = None,
    current_spend: np.ndarray,
    total_budget: float,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
) -> dict[str, Any]:
    """
    Maximize **Δμ vs baseline** subject to sum(spend)=budget and box constraints.

    Each objective evaluation calls :func:`mmm.planning.decision_simulate.simulate`.

    When ``ctx.config.budget.geo_budget_enabled`` is ``True``, optimizes a flattened
    ``(geo × channel)`` vector with pooled total ``sum_{g,c} spend = total_budget`` plus
    ``budget.geo_*`` inequalities; ``current_spend`` is ignored (warm start from baseline / BAU per geo).
    """
    names = list(ctx.schema.channel_columns)
    n = len(names)
    if current_spend.shape != (n,) or channel_min.shape != (n,) or channel_max.shape != (n,):
        raise ValueError("current_spend, channel_min, channel_max must align with channel_columns")
    if ctx.config.budget.geo_budget_enabled:
        return _optimize_budget_geo(
            ctx,
            baseline_plan=baseline_plan,
            total_budget=total_budget,
            channel_min=channel_min,
            channel_max=channel_max,
        )
    base = baseline_plan or bau_baseline_from_panel(ctx.panel, ctx.schema)
    base_vec = np.array([float(base.spend_by_channel[c]) for c in names], dtype=float)

    lo_sum = float(np.sum(channel_min))
    hi_sum = float(np.sum(channel_max))
    if lo_sum > total_budget + 1e-5:
        raise ValueError(f"total_budget {total_budget} below sum(channel_min)={lo_sum}")
    if hi_sum < total_budget - 1e-5:
        raise ValueError(f"total_budget {total_budget} exceeds sum(channel_max)={hi_sum}")

    def spend_dict(x: np.ndarray) -> dict[str, float]:
        return {names[i]: float(x[i]) for i in range(n)}

    agg = ctx.config.extensions.product.planning_delta_mu_aggregation

    def neg_delta_mu(x: np.ndarray) -> float:
        sim = simulate(
            spend_dict(x),
            ctx,
            baseline_plan=base,
            uncertainty_mode="point",
            delta_mu_aggregation=agg,
        )
        return -float(sim.delta_mu)

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

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]
    bounds = [(float(lo), float(hi)) for lo, hi in zip(channel_min, channel_max, strict=True)]
    res = minimize(neg_delta_mu, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    opt_x = np.clip(res.x, channel_min, channel_max)
    final_sim = simulate(
        spend_dict(opt_x),
        ctx,
        baseline_plan=base,
        uncertainty_mode="point",
        delta_mu_aggregation=agg,
    )
    disc = disclosure_for_non_bau_optimization(base)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "recommended_spend_plan": {c: float(x) for c, x in zip(names, opt_x, strict=True)},
        "objective_delta_mu": float(final_sim.delta_mu),
        "simulation_at_recommendation": final_sim.to_json(),
        "source": "full_model_simulation_slsqp",
        "baseline_type": base.baseline_type.value,
        "optimization_disclosure": disc,
        "baseline_spend_reference": {c: float(base_vec[i]) for i, c in enumerate(names)},
        "product_language_note": (
            "This is a simulation-scored budget allocation candidate from full-panel Δμ; "
            "do not describe as globally optimal unless validation and business review pass."
        ),
    }
