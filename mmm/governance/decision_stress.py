"""Decision stress via full-panel simulate() and optimize_budget_via_simulation()."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.decision.gates import allow_decision_pipeline
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext, ridge_context_from_summary

REPORT_VERSION = "mmm_decision_stress_v2"

StressSeverity = Literal["low", "moderate", "high", "critical"]
RecommendedAction = Literal["monitor", "review", "block"]

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Decision stress reports are diagnostic — no automatic budget changes are applied.",
    "stress_scope=train_time: stress is computed at train time only; it is not recomputed at decide time.",
    "Stress uses actual simulate/optimize paths when ridge_fit_summary and panel are available.",
)


def _allocation_dict(result: dict[str, Any], names: list[str]) -> dict[str, float]:
    plan = result.get("recommended_spend_plan") or result.get("allocation") or result.get("spend_by_channel") or {}
    if not isinstance(plan, dict):
        return {n: 0.0 for n in names}
    return {n: float(plan.get(n, 0.0)) for n in names}


def _top_channel(alloc: dict[str, float]) -> str | None:
    if not alloc:
        return None
    return max(alloc.keys(), key=lambda k: alloc[k])


def _allocation_l1_delta(a: dict[str, float], b: dict[str, float], budget: float) -> float:
    names = sorted(set(a) | set(b))
    va = np.array([a.get(n, 0.0) for n in names], dtype=float)
    vb = np.array([b.get(n, 0.0) for n in names], dtype=float)
    return float(np.linalg.norm(va - vb, ord=1) / max(budget, 1e-9))


def _try_build_context(
    config: MMMConfig,
    extension_report: dict[str, Any],
    panel: pd.DataFrame | None,
    schema: PanelSchema | None,
) -> RidgeFitContext | None:
    rfs = extension_report.get("ridge_fit_summary")
    if not isinstance(rfs, dict) or panel is None or schema is None:
        return None
    try:
        return ridge_context_from_summary(panel, schema, config, rfs)
    except (ValueError, KeyError, TypeError):
        return None


def _baseline_optimize(
    ctx: RidgeFitContext,
    *,
    budget: float,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    names = list(ctx.schema.channel_columns)
    n = len(names)
    base = bau_baseline_from_panel(ctx.panel, ctx.schema)
    with allow_decision_pipeline():
        result = optimize_budget_via_simulation(
            ctx,
            baseline_plan=base,
            current_spend=np.ones(n, dtype=float) * (budget / n),
            total_budget=budget,
            channel_min=np.zeros(n),
            channel_max=np.full(n, budget),
        )
    alloc = _allocation_dict(result, names)
    sim_at = result.get("simulation_at_recommendation") or {}
    delta_mu = float(sim_at.get("delta_mu", result.get("objective_delta_mu", 0.0)) or 0.0)
    return alloc, delta_mu, result


def _stressed_optimize(
    ctx: RidgeFitContext,
    *,
    budget: float,
    coef_scale: float = 1.0,
) -> tuple[dict[str, float], float]:
    if coef_scale != 1.0:
        ctx = RidgeFitContext(
            panel=ctx.panel,
            schema=ctx.schema,
            config=ctx.config,
            best_params=ctx.best_params,
            coef=np.asarray(ctx.coef, dtype=float) * coef_scale,
            intercept=ctx.intercept,
        )
    alloc, delta_mu, _ = _baseline_optimize(ctx, budget=budget)
    return alloc, delta_mu


def _signal_only_scenarios(config: MMMConfig, extension_report: dict[str, Any]) -> list[dict[str, Any]]:
    er = extension_report
    readiness = er.get("calibration_readiness_report") or {}
    cal = er.get("calibration_summary") or {}
    gov = er.get("governance") or {}
    scenarios: list[dict[str, Any]] = []
    stale = bool(readiness.get("stale_calibration_warning"))
    scenarios.append({"name": "stale_calibration", "triggered": stale, "severity": "high" if stale else "low"})
    gap_sev = str(cal.get("replay_generalization_gap_severity") or "")
    scenarios.append(
        {
            "name": "replay_degradation",
            "triggered": gap_sev in ("moderate", "severe"),
            "severity": "critical" if gap_sev == "severe" else ("moderate" if gap_sev == "moderate" else "low"),
        }
    )
    if not bool(gov.get("approved_for_optimization")):
        scenarios.append({"name": "governance_not_approved", "triggered": True, "severity": "critical"})
    return scenarios


def _behavioral_stress_scenarios(
    config: MMMConfig,
    extension_report: dict[str, Any],
    ctx: RidgeFitContext,
) -> tuple[list[dict[str, Any]], dict[str, float], float, list[float], list[float], float, float]:
    names = list(ctx.schema.channel_columns)
    budget = float(config.budget.total_budget or len(names) * 1e5)
    base_alloc, base_delta_mu, _ = _baseline_optimize(ctx, budget=budget)
    base_top = _top_channel(base_alloc)
    l1_moves: list[float] = []
    delta_mu_shifts: list[float] = []
    flip_count = 0
    n_behavioral = 0
    scenarios: list[dict[str, Any]] = []

    readiness = extension_report.get("calibration_readiness_report") or {}
    stale = bool(readiness.get("stale_calibration_warning"))
    if stale:
        _, st_dm = _baseline_optimize(ctx, budget=budget)
        l1_moves.append(0.0)
        delta_mu_shifts.append(abs(st_dm - base_delta_mu))
        scenarios.append(
            {
                "name": "stale_calibration",
                "triggered": True,
                "severity": "high",
                "delta_mu": st_dm,
                "baseline_delta_mu": base_delta_mu,
                "allocation": _baseline_optimize(ctx, budget=budget)[0],
            }
        )
    else:
        scenarios.append({"name": "stale_calibration", "triggered": False, "severity": "low"})

    pert_alloc, pert_dm = _stressed_optimize(ctx, budget=budget, coef_scale=0.85)
    n_behavioral += 1
    l1 = _allocation_l1_delta(base_alloc, pert_alloc, budget)
    l1_moves.append(l1)
    delta_mu_shifts.append(abs(pert_dm - base_delta_mu))
    top_pert = _top_channel(pert_alloc)
    flipped = base_top is not None and top_pert != base_top
    if flipped:
        flip_count += 1
    scenarios.append(
        {
            "name": "coefficient_perturbation",
            "triggered": l1 > 0.08 or flipped,
            "severity": "moderate" if flipped or l1 > 0.15 else "low",
            "allocation_l1_delta": l1,
            "decision_flip_top_channel": flipped,
            "delta_mu_shift": abs(pert_dm - base_delta_mu),
        }
    )

    cal = extension_report.get("calibration_summary") or {}
    missing_evidence = bool(cal.get("replay_calibration_active")) and cal.get("replay_train_loss") is None
    scenarios.append(
        {
            "name": "missing_evidence",
            "triggered": missing_evidence,
            "severity": "moderate" if missing_evidence else "low",
        }
    )

    shock_budget = budget * 1.25
    shock_alloc, shock_dm, _ = _baseline_optimize(ctx, budget=shock_budget)
    n_behavioral += 1
    l1_shock = _allocation_l1_delta(base_alloc, shock_alloc, budget)
    l1_moves.append(l1_shock)
    delta_mu_shifts.append(abs(shock_dm - base_delta_mu))
    top_shock = _top_channel(shock_alloc)
    if base_top and top_shock != base_top:
        flip_count += 1
    scenarios.append(
        {
            "name": "budget_shock",
            "triggered": l1_shock > 0.1,
            "severity": "moderate" if l1_shock > 0.2 else "low",
            "allocation_l1_delta": l1_shock,
            "budget_multiplier": 1.25,
            "delta_mu_shift": abs(shock_dm - base_delta_mu),
        }
    )

    gap_sev = str(cal.get("replay_generalization_gap_severity") or "")
    replay_stress = gap_sev == "severe"
    degraded_alloc, degraded_dm = _stressed_optimize(ctx, budget=budget, coef_scale=0.92)
    n_behavioral += 1
    l1_rep = _allocation_l1_delta(base_alloc, degraded_alloc, budget)
    l1_moves.append(l1_rep)
    delta_mu_shifts.append(abs(degraded_dm - base_delta_mu))
    scenarios.append(
        {
            "name": "replay_degradation",
            "triggered": replay_stress or l1_rep > 0.1,
            "severity": "critical" if replay_stress else ("moderate" if l1_rep > 0.15 else "low"),
            "allocation_l1_delta": l1_rep,
            "replay_generalization_gap_severity": gap_sev or None,
        }
    )

    flip_rate = float(flip_count / max(n_behavioral, 1))
    instability = float(np.mean(l1_moves)) if l1_moves else 0.0
    return scenarios, base_alloc, base_delta_mu, l1_moves, delta_mu_shifts, flip_rate, instability


def _aggregate_severity(scenarios: list[dict[str, Any]]) -> StressSeverity:
    order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    best = "low"
    for sc in scenarios:
        if not sc.get("triggered"):
            continue
        sev = str(sc.get("severity", "low"))
        if order.get(sev, 0) > order[best]:
            best = sev  # type: ignore[assignment]
    return best  # type: ignore[return-value]


def build_decision_stress_report(
    config: MMMConfig,
    extension_report: dict[str, Any],
    *,
    panel: pd.DataFrame | None = None,
    schema: PanelSchema | None = None,
    baseline_allocation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Stress-test decisions using actual simulate/optimize when panel + ridge summary exist."""
    ctx = _try_build_context(config, extension_report, panel, schema)
    if ctx is None:
        scenarios = _signal_only_scenarios(config, extension_report)
        severity = _aggregate_severity(scenarios)
        return {
            "report_version": REPORT_VERSION,
            "stress_mode": "signal_only",
            "stress_scope": "train_time_signal_only",
            "stress_severity": severity,
            "allocation_stability": severity in ("low", "moderate"),
            "decision_flip_rate": 0.0,
            "decision_instability_index": 0.0,
            "recommended_action": "review" if severity in ("high", "critical") else "monitor",
            "n_scenarios_triggered": sum(1 for s in scenarios if s.get("triggered")),
            "scenarios": scenarios,
            "baseline_allocation": baseline_allocation,
            "governance_warnings": list(GOVERNANCE_WARNINGS),
            "auto_budget_change": False,
        }

    scenarios, base_alloc, base_dm, _l1, _dm, flip_rate, instability = _behavioral_stress_scenarios(
        config, extension_report, ctx
    )
    severity = _aggregate_severity(scenarios)
    n_triggered = sum(1 for s in scenarios if s.get("triggered"))

    recommended: RecommendedAction = "monitor"
    if severity == "critical":
        recommended = "block"
    elif severity == "high" or flip_rate >= 0.34 or instability >= 0.2 or n_triggered >= 2:
        recommended = "review"

    allocation_stable = instability < 0.12 and flip_rate < 0.34

    return {
        "report_version": REPORT_VERSION,
        "stress_mode": "behavioral",
        "stress_scope": "train_time",
        "stress_severity": severity,
        "allocation_stability": allocation_stable,
        "decision_flip_rate": round(flip_rate, 4),
        "decision_instability_index": round(instability, 4),
        "recommended_action": recommended,
        "n_scenarios_triggered": n_triggered,
        "scenarios": scenarios,
        "baseline_allocation": baseline_allocation or base_alloc,
        "baseline_delta_mu": base_dm,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
        "auto_budget_change": False,
    }
