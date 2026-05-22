"""PR 5B: research-only robust optimization diagnostics (does not replace prod optimize-budget)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.extensions import RobustOptimizationResearchConfig
from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext, ridge_context_from_fit

REPORT_VERSION = "mmm_robust_optimization_research_v1"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Robust optimization research is diagnostic only — not production budget optimization.",
    "Lower-confidence-bound and risk-adjusted scores use uncertainty proxies, not calibrated monetary intervals.",
    "Do not treat robust_optimization_research allocations as recommended prod spend plans.",
    "decision_safe is always false for this artifact.",
)

ObjectiveKind = Literal[
    "maximize_expected_delta_mu",
    "maximize_lower_confidence_bound_proxy",
    "maximize_risk_adjusted_score",
]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _uncertainty_inputs(
    propagation: dict[str, Any] | None,
) -> tuple[float, bool, list[str], dict[str, Any]]:
    """Combined proxy and whether bootstrap/posterior summaries exist (still not prod CIs)."""
    warnings: list[str] = []
    if not isinstance(propagation, dict):
        return 0.35, False, ["missing_uncertainty_propagation_report"], {}

    sources = propagation.get("uncertainty_sources") or {}
    mags: list[float] = []
    for key in ("parameter_uncertainty", "experiment_uncertainty", "hierarchy_uncertainty", "allocation_uncertainty"):
        block = sources.get(key) if isinstance(sources, dict) else None
        if isinstance(block, dict):
            mags.append(float(block.get("magnitude_proxy", 0.0)))
            if block.get("present"):
                pass

    ridge_boot = propagation.get("ridge_bootstrap_summary") or {}
    bayes_post = propagation.get("bayesian_posterior_summary") or {}
    summarized_bootstrap = bool(ridge_boot.get("summarized"))
    summarized_posterior = bool(bayes_post.get("summarized"))
    has_calibrated_inputs = summarized_bootstrap or summarized_posterior

    if not has_calibrated_inputs:
        warnings.append(
            "No calibrated uncertainty summaries (bootstrap or posterior); scores use propagation "
            "magnitude_proxy labels only — not decision-grade intervals."
        )
    if propagation.get("prod_monetary_ci_allowed") is True:
        warnings.append("unexpected prod_monetary_ci_allowed on propagation report")

    combined = float(np.mean(mags)) if mags else 0.35
    return _clip01(combined), has_calibrated_inputs, warnings, dict(sources)


def _feasible_channel_vector(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    total: float,
) -> np.ndarray:
    x = np.clip(x, lo, hi)
    s = float(np.sum(x))
    if s <= 0:
        return np.full_like(x, total / len(x))
    return x * (total / s)


def _generate_candidates(
    names: list[str],
    baseline_vec: np.ndarray,
    coef_proxy: np.ndarray,
    *,
    n_candidates: int,
    total_budget: float,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[str, dict[str, float]]]:
    """Deterministic + random reallocation scenarios around BAU."""
    out: list[tuple[str, dict[str, float]]] = []
    base_dict = {names[i]: float(baseline_vec[i]) for i in range(len(names))}
    out.append(("bau_baseline", dict(base_dict)))

    if len(names) >= 2:
        ranked = np.argsort(coef_proxy)
        low_i, high_i = int(ranked[0]), int(ranked[-1])
        shift = 0.1 * total_budget
        x_shift = baseline_vec.copy()
        x_shift[low_i] = max(lo[low_i], x_shift[low_i] - shift)
        x_shift[high_i] = min(hi[high_i], x_shift[high_i] + shift)
        x_shift = _feasible_channel_vector(x_shift, lo, hi, total_budget)
        out.append(
            (
                "shift_to_high_coef_proxy",
                {names[i]: float(x_shift[i]) for i in range(len(names))},
            )
        )

    for k in range(max(0, n_candidates - len(out))):
        noise = rng.uniform(0.92, 1.08, size=len(names))
        x = _feasible_channel_vector(baseline_vec * noise, lo, hi, total_budget)
        out.append((f"random_reallocation_{k}", {names[i]: float(x[i]) for i in range(len(names))}))

    return out[: max(n_candidates, 1)]


def _evaluate_candidate(
    spend: dict[str, float],
    *,
    ctx: RidgeFitContext | None,
    baseline_plan: Any,
) -> float | None:
    if ctx is None:
        return None
    from mmm.planning.decision_simulate import simulate

    try:
        sim = simulate(
            spend,
            ctx,
            baseline_plan=baseline_plan,
            uncertainty_mode="point",
            delta_mu_aggregation=ctx.config.extensions.product.planning_delta_mu_aggregation,
        )
        return float(sim.delta_mu)
    except Exception:
        return None


def _stability_metrics(
    spend: dict[str, float],
    *,
    ctx: RidgeFitContext | None,
    baseline_plan: Any,
    names: list[str],
    baseline_vec: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    total_budget: float,
    rng: np.random.Generator,
    n_scenarios: int,
    perturb_pct: float,
) -> dict[str, Any]:
    if ctx is None:
        return {"available": False, "reason": "no_ridge_fit_context"}
    x0 = np.array([float(spend.get(c, baseline_vec[i])) for i, c in enumerate(names)], dtype=float)
    deltas: list[float] = []
    for _ in range(max(2, n_scenarios)):
        pert = rng.uniform(1.0 - perturb_pct, 1.0 + perturb_pct, size=len(names))
        x = _feasible_channel_vector(x0 * pert, lo, hi, total_budget)
        d = _evaluate_candidate(
            {names[i]: float(x[i]) for i in range(len(names))},
            ctx=ctx,
            baseline_plan=baseline_plan,
        )
        if d is not None:
            deltas.append(d)
    if not deltas:
        return {"available": False, "reason": "simulate_failed"}
    arr = np.asarray(deltas, dtype=float)
    return {
        "available": True,
        "delta_mu_std": float(np.std(arr)),
        "delta_mu_min": float(np.min(arr)),
        "delta_mu_max": float(np.max(arr)),
        "downside_gap_vs_mean": float(np.mean(arr) - np.min(arr)),
    }


def _rank_by_objective(
    rows: list[dict[str, Any]],
    objective: ObjectiveKind,
) -> list[str]:
    if objective == "maximize_expected_delta_mu":
        key = "expected_delta_mu"
    elif objective == "maximize_lower_confidence_bound_proxy":
        key = "lower_confidence_bound_proxy"
    else:
        key = "risk_adjusted_score"
    ranked = sorted(
        rows,
        key=lambda r: float(r.get(key) or -np.inf),
        reverse=True,
    )
    return [str(r["candidate_id"]) for r in ranked]


def build_robust_optimization_research(
    config: MMMConfig,
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    fit_out: dict[str, Any],
    extension_report: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Research-only robust allocation comparison using point Δμ and uncertainty proxies.

    Does not call ``optimize_budget_via_simulation`` or prod decision APIs.
    """
    roc: RobustOptimizationResearchConfig = config.extensions.robust_optimization_research
    gen = rng if rng is not None else np.random.default_rng(config.random_seed)

    base_report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "enabled": bool(roc.enabled),
        "research_only": True,
        "prod_decisioning_allowed": False,
        "decision_safe": False,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
        "warnings": list(GOVERNANCE_WARNINGS),
        "unsupported_claims": [
            "Production robust budget recommendation from this artifact.",
            "Calibrated monetary confidence intervals on allocations.",
        ],
    }

    if not roc.enabled:
        base_report["skipped"] = True
        base_report["reason"] = "robust_optimization_research_disabled"
        return base_report

    propagation = extension_report.get("uncertainty_propagation_report")
    combined_unc, has_calibrated_inputs, unc_warnings, source_snapshot = _uncertainty_inputs(
        propagation if isinstance(propagation, dict) else None
    )
    base_report["warnings"].extend(unc_warnings)
    base_report["uncertainty_input_snapshot"] = source_snapshot
    base_report["calibrated_uncertainty_inputs_available"] = has_calibrated_inputs
    base_report["uncertainty_label"] = (
        "bootstrap_or_posterior_summarized" if has_calibrated_inputs else "proxy_only"
    )

    ctx: RidgeFitContext | None = None
    if config.framework == Framework.RIDGE_BO and fit_out.get("artifacts") is not None:
        try:
            ctx = ridge_context_from_fit(panel, schema, config, fit_out)
        except Exception as exc:
            base_report["warnings"].append(f"ridge_fit_context_unavailable: {exc}")

    if ctx is None:
        base_report["warnings"].append(
            "Ridge fit context required for Δμ evaluation; robust research needs framework=ridge_bo "
            "with successful fit artifacts."
        )
        base_report["skipped"] = True
        base_report["reason"] = "no_simulate_context"
        return base_report

    names = list(schema.channel_columns)
    n = len(names)
    baseline_plan = bau_baseline_from_panel(ctx.panel, ctx.schema)
    baseline_vec = np.array([float(baseline_plan.spend_by_channel[c]) for c in names], dtype=float)
    total_budget = float(config.budget.total_budget or np.sum(baseline_vec))
    lo = np.array(
        [float(config.budget.channel_min.get(c, 0.0)) for c in names],
        dtype=float,
    )
    hi = np.array(
        [
            float(
                config.budget.channel_max.get(c, max(float(baseline_vec[i]) * 3.0, 1.0))
            )
            for i, c in enumerate(names)
        ],
        dtype=float,
    )
    coef_proxy = np.asarray(ctx.coef, dtype=float).ravel()[:n]
    if coef_proxy.size < n:
        coef_proxy = np.ones(n, dtype=float)

    candidates = _generate_candidates(
        names,
        baseline_vec,
        coef_proxy,
        n_candidates=int(roc.n_candidates),
        total_budget=total_budget,
        lo=lo,
        hi=hi,
        rng=gen,
    )

    rows: list[dict[str, Any]] = []
    for cid, spend in candidates:
        mu = _evaluate_candidate(spend, ctx=ctx, baseline_plan=baseline_plan)
        if mu is None:
            continue
        spend_vals = list(spend.values())
        spend_cv = float(np.std(spend_vals)) / (float(np.mean(spend_vals)) + 1e-9)
        u_proxy = _clip01(combined_unc * (1.0 + 0.05 * spend_cv))
        scale = max(abs(mu), 1e-6)
        lcb = float(mu - roc.lcb_z_score * u_proxy * scale)
        risk_adj = float(mu - roc.risk_lambda * u_proxy * scale)
        stab = _stability_metrics(
            spend,
            ctx=ctx,
            baseline_plan=baseline_plan,
            names=names,
            baseline_vec=baseline_vec,
            lo=lo,
            hi=hi,
            total_budget=total_budget,
            rng=gen,
            n_scenarios=int(roc.n_stability_scenarios),
            perturb_pct=float(roc.budget_perturbation_pct),
        )
        if stab.get("available"):
            downside = float(stab.get("downside_gap_vs_mean", u_proxy * scale))
        else:
            downside = u_proxy * scale
        unc_label = (
            "proxy_from_summarized_uncertainty"
            if has_calibrated_inputs
            else "proxy_only"
        )
        rows.append(
            {
                "candidate_id": cid,
                "allocation": spend,
                "expected_delta_mu": mu,
                "uncertainty_proxy": u_proxy,
                "uncertainty_proxy_label": unc_label,
                "lower_confidence_bound_proxy": lcb,
                "risk_adjusted_score": risk_adj,
                "downside_risk_proxy": downside,
                "allocation_stability": stab,
            }
        )

    if not rows:
        base_report["skipped"] = True
        base_report["reason"] = "no_successful_candidate_evaluations"
        return base_report

    baseline_row = next((r for r in rows if r["candidate_id"] == "bau_baseline"), rows[0])
    frontier: list[dict[str, Any]] = []
    for lam in roc.frontier_lambda_grid:
        lam_f = float(lam)
        def _frontier_score(r: dict[str, Any], lam: float) -> float:
            mu_r = float(r["expected_delta_mu"])
            u_r = float(r["uncertainty_proxy"])
            return mu_r - lam * u_r * max(abs(mu_r), 1e-6)

        best = max(rows, key=lambda r: _frontier_score(r, lam_f))
        frontier.append(
            {
                "risk_lambda": lam_f,
                "best_candidate_id": best["candidate_id"],
                "risk_adjusted_score": _frontier_score(best, lam_f),
                "expected_delta_mu": float(best["expected_delta_mu"]),
                "uncertainty_proxy": float(best["uncertainty_proxy"]),
            }
        )

    ranking_stability: dict[str, Any] = {}
    if len(rows) >= 2:
        rank_expected = _rank_by_objective(rows, "maximize_expected_delta_mu")
        rank_risk = _rank_by_objective(rows, "maximize_risk_adjusted_score")
        overlap = len(set(rank_expected[:3]) & set(rank_risk[:3])) / max(len(rank_expected[:3]), 1)
        ranking_stability = {
            "top3_overlap_expected_vs_risk_adjusted": float(overlap),
            "rank_by_expected_delta_mu": rank_expected,
            "rank_by_risk_adjusted": rank_risk,
            "rank_by_lcb_proxy": _rank_by_objective(rows, "maximize_lower_confidence_bound_proxy"),
            "note": "Compares allocation ranking under uncertainty scenarios (research).",
        }

    base_report.update(
        {
            "skipped": False,
            "objectives_supported": [
                "maximize_expected_delta_mu",
                "maximize_lower_confidence_bound_proxy",
                "maximize_risk_adjusted_score",
            ],
            "risk_lambda_config": float(roc.risk_lambda),
            "lcb_z_score": float(roc.lcb_z_score),
            "baseline_allocation": dict(baseline_row["allocation"]),
            "candidate_allocations": [r["allocation"] for r in rows],
            "candidate_details": rows,
            "expected_delta_mu": {r["candidate_id"]: r["expected_delta_mu"] for r in rows},
            "uncertainty_proxy": {r["candidate_id"]: r["uncertainty_proxy"] for r in rows},
            "lower_confidence_bound_proxy": {r["candidate_id"]: r["lower_confidence_bound_proxy"] for r in rows},
            "risk_adjusted_score": {r["candidate_id"]: r["risk_adjusted_score"] for r in rows},
            "risk_return_frontier": frontier,
            "allocation_stability": {r["candidate_id"]: r["allocation_stability"] for r in rows},
            "downside_risk_proxy": {r["candidate_id"]: r["downside_risk_proxy"] for r in rows},
            "ranking_stability": ranking_stability,
            "recommended_prod_allocation": None,
            "prod_optimize_budget_path_used": False,
        }
    )
    return base_report
