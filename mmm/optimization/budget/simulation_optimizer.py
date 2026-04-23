"""Budget optimization via full-panel simulate() — objective is Δμ (not curve interpolation)."""

from __future__ import annotations

from collections.abc import Callable
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


def _effective_n_starts(n: int) -> int:
    """Clamp multistart count to a decision-grade band (10–20)."""
    return int(np.clip(int(n), 10, 20))


def _delta_mu_std_budget_perturbation_national(
    x_center: np.ndarray,
    names: list[str],
    *,
    ctx: RidgeFitContext,
    base: BaselinePlan,
    agg: Any,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
    total_budget: float,
    rng: np.random.Generator,
    n_samples: int = 8,
) -> float:
    vals: list[float] = []
    for _ in range(max(2, n_samples)):
        pert = rng.uniform(0.95, 1.05, size=len(x_center))
        x2 = _feasible_budget_vector(
            np.clip(x_center * pert, channel_min, channel_max),
            channel_min,
            channel_max,
            total_budget,
        )
        sim = simulate(
            {names[i]: float(x2[i]) for i in range(len(names))},
            ctx,
            baseline_plan=base,
            uncertainty_mode="point",
            delta_mu_aggregation=agg,
        )
        vals.append(float(sim.delta_mu))
    return float(np.std(np.asarray(vals, dtype=float)))


def _delta_mu_std_budget_perturbation_geo(
    x_center: np.ndarray,
    geos: list[str],
    names: list[str],
    *,
    ctx: RidgeFitContext,
    base: BaselinePlan,
    agg: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    total_budget: float,
    rng: np.random.Generator,
    n_samples: int = 8,
) -> float:
    vals: list[float] = []
    for _ in range(max(2, n_samples)):
        pert = rng.uniform(0.95, 1.05, size=len(x_center))
        x2 = _feasible_budget_vector(np.clip(x_center * pert, lo, hi), lo, hi, total_budget)
        gdict = _flat_to_geo_dict(x2, geos, names)
        sim = simulate(
            {},
            ctx,
            baseline_plan=base,
            uncertainty_mode="point",
            spend_plan_geo=gdict,
            delta_mu_aggregation=agg,
        )
        vals.append(float(sim.delta_mu))
    return float(np.std(np.asarray(vals, dtype=float)))


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


def _feasible_budget_vector(x_seed: np.ndarray, lo: np.ndarray, hi: np.ndarray, total_budget: float) -> np.ndarray:
    """Clip to bounds, scale toward ``total_budget`` sum, then redistribute slack so sum matches (box-feasible)."""
    x0 = np.clip(x_seed.astype(float), lo, hi)
    s0 = float(x0.sum())
    if s0 > 1e-12:
        x0 = x0 * (total_budget / s0)
    x0 = np.clip(x0, lo, hi)
    dim = len(x0)
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
    return x0


def _multistart_slsqp(
    neg_f: Callable[[np.ndarray], float],
    *,
    x0_feasible: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    total_budget: float,
    bounds: list[tuple[float, float]],
    constraints: list[dict[str, Any]],
    n_starts: int,
    rng: np.random.Generator,
) -> tuple[Any, dict[str, Any]]:
    """Several feasible randomized starts; return best objective (lowest ``fun``) and audit metadata."""
    dim = len(x0_feasible)
    best = None
    best_idx = 0
    starts: list[dict[str, Any]] = []
    x0b = _feasible_budget_vector(x0_feasible, lo, hi, total_budget)
    for k in range(n_starts):
        if k == 0:
            x0 = x0b.copy()
        else:
            pert = rng.uniform(0.88, 1.12, size=dim)
            x0 = _feasible_budget_vector(x0b * pert, lo, hi, total_budget)
        res = minimize(neg_f, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        rec = {
            "start_index": k,
            "optimizer_success": bool(res.success),
            "objective_neg_delta_mu": float(res.fun),
            "message": str(res.message),
        }
        starts.append(rec)
        if best is None or float(res.fun) < float(best.fun):
            best = res
            best_idx = k
    assert best is not None
    meta = {
        "n_starts": n_starts,
        "starts": starts,
        "chosen_start_index": best_idx,
        "sum_budget_target": float(total_budget),
    }
    return best, meta


def _vector_binding_report(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    labels: list[str],
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Which decision variables sit on lower/upper bounds (constraint activation proxy)."""
    xv, lv, hv = np.asarray(x, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    at_lo = xv - lv <= tol
    at_hi = hv - xv <= tol
    idx_lo = [labels[i] for i in np.where(at_lo)[0]]
    idx_hi = [labels[i] for i in np.where(at_hi)[0]]
    return {
        "n_at_lower_bound": int(np.sum(at_lo)),
        "n_at_upper_bound": int(np.sum(at_hi)),
        "fraction_at_any_bound": float(np.mean((at_lo | at_hi).astype(float))),
        "channels_at_lower": idx_lo[:64],
        "channels_at_upper": idx_hi[:64],
    }


def _decision_safe_false_reasons(
    *,
    optimizer_success: bool,
    allocation_stable: bool,
    jitter_ok: bool,
) -> list[str]:
    """Programmatic reasons when ``decision_safe`` is false but the solver may still report success."""
    out: list[str] = []
    if not optimizer_success:
        out.append("numerical_optimizer_did_not_report_success")
    if not allocation_stable:
        out.append("allocation_unstable_across_perturbed_multistart_re_solves")
    if not jitter_ok:
        out.append("delta_mu_materially_sensitive_to_small_budget_perturbation_vs_scale")
    return out


def _objective_path_economics_metadata(*, mode: str) -> dict[str, Any]:
    return {
        "canonical_decision_quantity": "delta_mu_vs_baseline",
        "uncertainty_mode_in_objective": "point",
        "exact_vs_approximate": (
            "point_delta_mu_under_fixed_ridge_coefficients_is_exact_given_the_fitted_mu_map"
        ),
        "approximation_flags": {
            "slsqp": "local_nlp_solver_may_not_find_global_budget_optimum",
            "multistart": "finite_random_feasible_starts_cap_exploration",
            "mode": mode,
        },
    }


def _normalized_allocation_stability(
    neg_f: Callable[[np.ndarray], float],
    opt_x: np.ndarray,
    *,
    lo: np.ndarray,
    hi: np.ndarray,
    total_budget: float,
    bounds: list[tuple[float, float]],
    constraints: list[dict[str, Any]],
    n_checks: int,
    rng: np.random.Generator,
    l1_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """Re-solve from perturbed feasible starts; large L1 movement of normalized allocation => unstable."""
    if n_checks <= 0:
        return True, {"skipped": True, "max_l1_norm_diff": 0.0, "l1_threshold": l1_threshold}
    ox = np.clip(opt_x.astype(float), lo, hi)
    denom = max(float(ox.sum()), 1e-12)
    ref = ox / denom
    max_l1 = 0.0
    checks: list[dict[str, Any]] = []
    for _ in range(n_checks):
        pert = 1.0 + rng.uniform(-0.025, 0.025, size=len(ox))
        x0p = _feasible_budget_vector(ox * pert, lo, hi, total_budget)
        r2 = minimize(neg_f, x0=x0p, method="SLSQP", bounds=bounds, constraints=constraints)
        x2 = np.clip(r2.x, lo, hi)
        nr = x2 / max(float(x2.sum()), 1e-12)
        l1 = float(np.sum(np.abs(ref - nr)))
        max_l1 = max(max_l1, l1)
        checks.append({"optimizer_success": bool(r2.success), "l1_norm_diff_normalized_alloc": l1})
    stable = max_l1 <= l1_threshold
    return stable, {
        "n_checks": n_checks,
        "checks": checks,
        "max_l1_norm_diff": max_l1,
        "l1_threshold": l1_threshold,
        "allocation_stable": stable,
    }


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

    prod = ctx.config.extensions.product
    rng = np.random.default_rng(int(ctx.config.ridge_bo.sampler_seed))
    x0_feas = _feasible_budget_vector(x0_vector(), lo, hi, total_budget)

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
    res, multistart_meta = _multistart_slsqp(
        neg_delta_mu,
        x0_feasible=x0_feas,
        lo=lo,
        hi=hi,
        total_budget=total_budget,
        bounds=bounds,
        constraints=cons,
        n_starts=_effective_n_starts(prod.simulation_optimizer_n_starts),
        rng=rng,
    )
    opt_x = np.clip(res.x, lo, hi)
    stab_ok, stab_meta = _normalized_allocation_stability(
        neg_delta_mu,
        opt_x,
        lo=lo,
        hi=hi,
        total_budget=total_budget,
        bounds=bounds,
        constraints=cons,
        n_checks=prod.simulation_optimizer_stability_checks,
        rng=rng,
        l1_threshold=prod.simulation_optimizer_stability_max_l1,
    )
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
    geo_ch_labels = [f"{g}:{c}" for g in geos for c in names]
    binding = _vector_binding_report(opt_x, lo, hi, labels=geo_ch_labels)
    dm_std = _delta_mu_std_budget_perturbation_geo(
        opt_x,
        geos,
        names_list,
        ctx=ctx,
        base=base,
        agg=agg,
        lo=lo,
        hi=hi,
        total_budget=total_budget,
        rng=rng,
    )
    stability_score = float(1.0 / (1.0 + dm_std))
    ref_scale = max(abs(float(final_sim.delta_mu)), 1.0)
    jitter_ok = dm_std <= 0.02 * ref_scale
    opt_decision_safe = bool(res.success and stab_ok and jitter_ok)
    dsr = _decision_safe_false_reasons(
        optimizer_success=bool(res.success),
        allocation_stable=stab_ok,
        jitter_ok=jitter_ok,
    )
    sim_js = final_sim.to_json()
    sim_js.setdefault(
        "economics_metadata",
        {
            "kpi_column": ctx.schema.target_column,
            "baseline_definition": sim_js.get("baseline_definition"),
            "approximation_flags": {"optimizer": "slsqp_geo_multistart", "simulation": "full_panel_ridge"},
        },
    )
    return {
        "success": bool(res.success),
        "optimizer_success": bool(res.success),
        "decision_safe": opt_decision_safe,
        "allocation_stable": stab_ok,
        "stability_score": stability_score,
        "num_restarts": int(multistart_meta["n_starts"]),
        "delta_mu_budget_perturbation_std": dm_std,
        "message": str(res.message),
        "recommended_spend_plan": {c: float(mean_ch[c]) for c in names},
        "recommended_spend_plan_by_geo": opt_geo,
        "objective_delta_mu": float(final_sim.delta_mu),
        "simulation_at_recommendation": sim_js,
        "source": "full_model_simulation_slsqp_geo",
        "baseline_type": base.baseline_type.value,
        "optimization_disclosure": disc,
        "baseline_spend_reference": {c: float(base_vec[i]) for i, c in enumerate(names)},
        "multistart": multistart_meta,
        "stability": stab_meta,
        "constraint_binding": binding,
        "decision_safe_false_reasons": [] if opt_decision_safe else dsr,
        "objective_path_economics_metadata": _objective_path_economics_metadata(mode="geo_pooled_total_budget"),
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
    from mmm.decision.gates import decision_pipeline_active

    if not decision_pipeline_active():
        raise RuntimeError(
            "optimize_budget_via_simulation is decision-gated: use mmm.decision.api.run_decision_optimization, "
            "or in tests wrap the call with mmm.decision.gates.allow_decision_pipeline()."
        )
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

    prod = ctx.config.extensions.product
    agg = prod.planning_delta_mu_aggregation
    rng = np.random.default_rng(int(ctx.config.ridge_bo.sampler_seed))

    def neg_delta_mu(x: np.ndarray) -> float:
        sim = simulate(
            spend_dict(x),
            ctx,
            baseline_plan=base,
            uncertainty_mode="point",
            delta_mu_aggregation=agg,
        )
        return -float(sim.delta_mu)

    x0_feas = _feasible_budget_vector(
        np.clip(current_spend.astype(float), channel_min, channel_max),
        channel_min,
        channel_max,
        total_budget,
    )

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]
    bounds = [(float(lo), float(hi)) for lo, hi in zip(channel_min, channel_max, strict=True)]
    res, multistart_meta = _multistart_slsqp(
        neg_delta_mu,
        x0_feasible=x0_feas,
        lo=channel_min,
        hi=channel_max,
        total_budget=total_budget,
        bounds=bounds,
        constraints=cons,
        n_starts=_effective_n_starts(prod.simulation_optimizer_n_starts),
        rng=rng,
    )
    opt_x = np.clip(res.x, channel_min, channel_max)
    stab_ok, stab_meta = _normalized_allocation_stability(
        neg_delta_mu,
        opt_x,
        lo=channel_min,
        hi=channel_max,
        total_budget=total_budget,
        bounds=bounds,
        constraints=cons,
        n_checks=prod.simulation_optimizer_stability_checks,
        rng=rng,
        l1_threshold=prod.simulation_optimizer_stability_max_l1,
    )
    final_sim = simulate(
        spend_dict(opt_x),
        ctx,
        baseline_plan=base,
        uncertainty_mode="point",
        delta_mu_aggregation=agg,
    )
    disc = disclosure_for_non_bau_optimization(base)
    ch_labels = [f"national:{c}" for c in names]
    binding = _vector_binding_report(opt_x, channel_min, channel_max, labels=ch_labels)
    dm_std = _delta_mu_std_budget_perturbation_national(
        opt_x,
        names,
        ctx=ctx,
        base=base,
        agg=agg,
        channel_min=channel_min,
        channel_max=channel_max,
        total_budget=total_budget,
        rng=rng,
    )
    stability_score = float(1.0 / (1.0 + dm_std))
    ref_scale = max(abs(float(final_sim.delta_mu)), 1.0)
    jitter_ok = dm_std <= 0.02 * ref_scale
    opt_decision_safe = bool(res.success and stab_ok and jitter_ok)
    dsr = _decision_safe_false_reasons(
        optimizer_success=bool(res.success),
        allocation_stable=stab_ok,
        jitter_ok=jitter_ok,
    )
    sim_js = final_sim.to_json()
    sim_js.setdefault(
        "economics_metadata",
        {
            "kpi_column": ctx.schema.target_column,
            "baseline_definition": sim_js.get("baseline_definition"),
            "approximation_flags": {"optimizer": "slsqp_national_multistart", "simulation": "full_panel_ridge"},
        },
    )
    return {
        "success": bool(res.success),
        "optimizer_success": bool(res.success),
        "decision_safe": opt_decision_safe,
        "allocation_stable": stab_ok,
        "stability_score": stability_score,
        "num_restarts": int(multistart_meta["n_starts"]),
        "delta_mu_budget_perturbation_std": dm_std,
        "message": str(res.message),
        "recommended_spend_plan": {c: float(x) for c, x in zip(names, opt_x, strict=True)},
        "objective_delta_mu": float(final_sim.delta_mu),
        "simulation_at_recommendation": sim_js,
        "source": "full_model_simulation_slsqp",
        "baseline_type": base.baseline_type.value,
        "optimization_disclosure": disc,
        "baseline_spend_reference": {c: float(base_vec[i]) for i, c in enumerate(names)},
        "multistart": multistart_meta,
        "stability": stab_meta,
        "constraint_binding": binding,
        "decision_safe_false_reasons": [] if opt_decision_safe else dsr,
        "objective_path_economics_metadata": _objective_path_economics_metadata(mode="national_total_budget"),
        "product_language_note": (
            "This is a simulation-scored budget allocation candidate from full-panel Δμ; "
            "do not describe as globally optimal unless validation and business review pass."
        ),
    }
