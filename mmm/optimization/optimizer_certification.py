"""Deterministic optimizer certification with grid-derived analytic optima."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.decision.gates import allow_decision_pipeline
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate

REPORT_VERSION = "mmm_optimizer_certification_v2"

# L1 distance from observed allocation to grid optimum, as fraction of budget.
OPTIMUM_L1_TOLERANCE = 0.18
REPEATABILITY_L1_STD_MAX = 0.12

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Optimizer certification uses controlled Ridge contexts — not production panel generalization.",
    "certification_mode=analytic_tolerance proves grid optimum recovery within L1 tolerance.",
    (
        "certification_mode=directional_fallback proves channel dominance only — "
        "not exact optimum recovery."
    ),
)


def _minimal_panel(*, n_weeks: int = 12) -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    for g in ("G0", "G1"):
        for w in range(n_weeks):
            rows.append(
                {
                    "geo_id": g,
                    "week_start_date": w,
                    "revenue": 100.0,
                    "channel_a": 10.0,
                    "channel_b": 10.0,
                }
            )
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("channel_a", "channel_b"))
    return panel, schema


def _ridge_ctx(
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    coef: np.ndarray,
    seed: int,
    best_params: dict[str, float] | None = None,
) -> RidgeFitContext:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        random_seed=seed,
        ridge_bo={"sampler_seed": seed},
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        extensions={
            "product": {
                "simulation_optimizer_n_starts": 10,
                "simulation_optimizer_stability_checks": 2,
            }
        },
    )
    bp = best_params or {"decay": 0.01, "hill_half": 1e6, "hill_slope": 1.0}
    return RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=coef,
        intercept=np.array([np.log(100.0)]),
        best_params=bp,
    )


def _allocation_vector(result: dict[str, Any], names: list[str]) -> np.ndarray:
    plan = result.get("recommended_spend_plan") or result.get("allocation") or {}
    if not isinstance(plan, dict):
        plan = result.get("spend_by_channel") or {}
    return np.array([float(plan.get(n, 0.0)) for n in names], dtype=float)


def _grid_optimum_allocation(
    ctx: RidgeFitContext,
    *,
    budget: float,
    n_grid: int = 91,
) -> tuple[np.ndarray, float]:
    """Exhaustive grid on channel_a budget share (2-channel certification surfaces)."""
    names = list(ctx.schema.channel_columns)
    if len(names) != 2:
        raise ValueError("grid optimum requires exactly two channels")
    base = bau_baseline_from_panel(ctx.panel, ctx.schema)
    best_x = np.array([budget / 2, budget / 2])
    best_dm = -np.inf
    for i in range(max(2, n_grid)):
        share_a = i / (n_grid - 1)
        spend_a = budget * share_a
        spend_b = budget * (1.0 - share_a)
        plan = {names[0]: float(spend_a), names[1]: float(spend_b)}
        dm = float(simulate(plan, ctx, baseline_plan=base).delta_mu)
        if dm > best_dm:
            best_dm = dm
            best_x = np.array([spend_a, spend_b])
    return best_x, best_dm


def _certify_scenario(
    *,
    scenario: str,
    ctx: RidgeFitContext,
    seed: int,
    budget: float,
    coef_ratio_expected: float | None = None,
) -> dict[str, Any]:
    names = list(ctx.schema.channel_columns)
    n = len(names)
    with allow_decision_pipeline():
        result = optimize_budget_via_simulation(
            ctx,
            current_spend=np.array([budget / n, budget / n]),
            total_budget=budget,
            channel_min=np.zeros(n),
            channel_max=np.full(n, budget),
        )
    obs = _allocation_vector(result, names)
    analytic, analytic_delta_mu = _grid_optimum_allocation(ctx, budget=budget)
    base = bau_baseline_from_panel(ctx.panel, ctx.schema)
    obs_plan = {names[i]: float(obs[i]) for i in range(n)}
    obs_delta_mu = float(simulate(obs_plan, ctx, baseline_plan=base).delta_mu)
    total = float(obs.sum())
    feasibility = abs(total - budget) < 1.0
    l1_err = float(np.linalg.norm(obs - analytic, ord=1) / max(budget, 1e-9))
    optimum_distance = l1_err
    optimizer_error = l1_err
    obj_gap = abs(obs_delta_mu - analytic_delta_mu)
    convergence = bool(result.get("optimizer_success", result.get("success")))
    allocation_stable = bool(result.get("allocation_stable", True))
    coef = np.asarray(ctx.coef, dtype=float).ravel()
    high_idx = int(np.argmax(coef))
    corner_dominant = float(max(analytic) / max(budget, 1e-9)) >= 0.95
    ratio_ok = True
    if coef_ratio_expected is not None and obs[1] > 1e-9:
        obs_ratio = float(obs[0] / obs[1])
        ratio_err = abs(obs_ratio - coef_ratio_expected) / coef_ratio_expected
        ratio_ok = ratio_err <= 0.25
    if corner_dominant:
        certification_mode = "directional_fallback"
        directional_ok = obs[high_idx] > obs[1 - high_idx]
        within_band = directional_ok and obj_gap <= max(0.05, abs(analytic_delta_mu) * 2 + 1e-6)
    else:
        certification_mode = "analytic_tolerance"
        within_band = optimum_distance <= OPTIMUM_L1_TOLERANCE and ratio_ok and obj_gap <= max(
            0.05, abs(analytic_delta_mu) * 0.15 + 1e-6
        )
    cert_status = "pass" if within_band and feasibility and convergence else "fail"
    return {
        "scenario": scenario,
        "certification_mode": certification_mode,
        "analytic_optimum_allocation": {names[i]: float(analytic[i]) for i in range(n)},
        "analytic_optimum_delta_mu": analytic_delta_mu,
        "expected_allocation_band": {
            "l1_tolerance": OPTIMUM_L1_TOLERANCE,
            "center": {names[i]: float(analytic[i]) for i in range(n)},
        },
        "observed_allocation": {names[i]: float(obs[i]) for i in range(n)},
        "optimizer_error": optimizer_error,
        "optimum_distance": optimum_distance,
        "allocation_error_l1_norm": l1_err,
        "observed_delta_mu": obs_delta_mu,
        "objective_gap": obj_gap,
        "corner_dominant_analytic": corner_dominant,
        "feasibility": feasibility,
        "convergence": convergence,
        "allocation_stable": allocation_stable,
        "certification_status": cert_status,
        "seed": seed,
    }


def _repeatability_scenario(*, seeds: tuple[int, ...] = (11, 22, 33)) -> dict[str, Any]:
    panel, schema = _minimal_panel()
    names = list(schema.channel_columns)
    coef = np.array([0.4, 0.2])
    allocs: list[np.ndarray] = []
    for seed in seeds:
        ctx = _ridge_ctx(panel, schema, coef=coef, seed=seed)
        n = len(names)
        with allow_decision_pipeline():
            result = optimize_budget_via_simulation(
                ctx,
                current_spend=np.array([50.0, 50.0]),
                total_budget=100.0,
                channel_min=np.zeros(n),
                channel_max=np.full(n, 100.0),
            )
        allocs.append(_allocation_vector(result, names))
    stack = np.vstack(allocs)
    l1_spread = float(np.max(np.std(stack / 100.0, axis=0)))
    stable = l1_spread <= REPEATABILITY_L1_STD_MAX
    return {
        "scenario": "repeatability",
        "certification_mode": "analytic_tolerance",
        "seeds": list(seeds),
        "allocation_l1_std": l1_spread,
        "local_optimum_stability": stable,
        "optimum_distance": l1_spread,
        "optimizer_error": l1_spread,
        "certification_status": "pass" if stable else "fail",
    }


def _rollup_certification_mode(scenarios: list[dict[str, Any]]) -> str:
    if any(str(s.get("certification_status")) != "pass" for s in scenarios):
        return "smoke"
    modes = [str(s.get("certification_mode", "")) for s in scenarios]
    if any(m == "directional_fallback" for m in modes):
        return "directional_fallback"
    if all(m == "analytic_tolerance" for m in modes):
        return "analytic_tolerance"
    return "smoke"


def _run_scenario_a(*, seed: int) -> dict[str, Any]:
    """Higher coef on channel_a — grid may be corner; directional + objective gap certified."""
    panel, schema = _minimal_panel()
    coef = np.array([0.80, 0.20])
    ctx = _ridge_ctx(panel, schema, coef=coef, seed=seed)
    return _certify_scenario(scenario="A_log_elasticity", ctx=ctx, seed=seed, budget=100.0, coef_ratio_expected=4.0)


def _run_scenario_b(*, seed: int) -> dict[str, Any]:
    panel, schema = _minimal_panel()
    coef = np.array([0.35, 0.175])
    ctx = _ridge_ctx(
        panel,
        schema,
        coef=coef,
        seed=seed,
        best_params={"decay": 0.05, "hill_half": 50.0, "hill_slope": 2.0},
    )
    return _certify_scenario(
        scenario="B_saturated_two_channel",
        ctx=ctx,
        seed=seed,
        budget=100.0,
        coef_ratio_expected=2.0,
    )


def build_optimizer_certification_report(*, seed: int = 42) -> dict[str, Any]:
    scenarios = [
        _run_scenario_a(seed=seed),
        _run_scenario_b(seed=seed + 1),
        _repeatability_scenario(),
    ]
    n_pass = sum(1 for s in scenarios if s.get("certification_status") == "pass")
    status = "pass" if n_pass == len(scenarios) else "fail"
    certification_mode = _rollup_certification_mode(scenarios)
    return {
        "report_version": REPORT_VERSION,
        "certification_status": status,
        "certification_mode": certification_mode,
        "n_pass": n_pass,
        "n_scenarios": len(scenarios),
        "optimum_l1_tolerance": OPTIMUM_L1_TOLERANCE,
        "scenarios": scenarios,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
    }
