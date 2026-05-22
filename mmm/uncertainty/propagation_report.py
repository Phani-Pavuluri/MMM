"""PR 5A: uncertainty propagation reports (research/diagnostic; no prod monetary CIs)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.extensions import UncertaintyPropagationConfig
from mmm.config.schema import Framework, MMMConfig
from mmm.governance.uncertainty_policy import ridge_forbids_precise_monetary_ci
from mmm.uncertainty.decomposition import UncertaintyDecomposer

REPORT_VERSION = "mmm_uncertainty_propagation_v1"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Uncertainty propagation reports are research/diagnostic only.",
    "Prod decision paths do not expose calibrated monetary confidence intervals on Ridge.",
    "Source breakdown magnitudes are proxies — not decision-grade calibrated intervals.",
    "Experiment and allocation uncertainty do not justify subgeo causal claims.",
)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _ridge_bootstrap_summary(
    config: MMMConfig,
    ident: dict[str, Any],
    separability: dict[str, Any] | None,
) -> dict[str, Any]:
    up = config.extensions.uncertainty_propagation
    if not up.ridge_summarize_bootstrap:
        return {"summarized": False, "reason": "ridge_summarize_bootstrap_disabled"}
    if ident.get("skipped"):
        return {"summarized": False, "reason": "identifiability_skipped"}
    ctx = ident.get("diagnostic_context") or {}
    out: dict[str, Any] = {
        "summarized": True,
        "method": "ridge_bootstrap_refits",
        "instability_score": float(ident.get("instability_score", 0.0)),
        "identifiability_score": float(ident.get("identifiability_score", 0.0)),
        "bootstrap_rounds_configured": int(ctx.get("bootstrap_rounds_configured", 0)),
        "bootstrap_frac_configured": float(ctx.get("bootstrap_frac_configured", 0.0)),
        "coef_dispersion_proxy": float(ident.get("instability_score", 0.0)),
        "max_vif": float(ident.get("max_vif", 0.0)),
        "note": (
            "Bootstrap refits use the same ridge alpha as the selected BO trial (identifiability engine). "
            "Not a calibrated posterior for Δμ."
        ),
    }
    if isinstance(separability, dict) and not separability.get("skipped"):
        groups = separability.get("feature_groups") or []
        unstable = [
            g.get("group_name")
            for g in groups
            if isinstance(g, dict) and g.get("coefficient_stability") == "unstable"
        ]
        if unstable:
            out["unstable_feature_groups"] = unstable[:10]
    return out


def _ridge_conformal_summary(config: MMMConfig) -> dict[str, Any]:
    up = config.extensions.uncertainty_propagation
    if not up.ridge_summarize_conformal:
        return {"summarized": False, "reason": "ridge_summarize_conformal_disabled"}
    return {
        "summarized": False,
        "status": "not_implemented",
        "note": (
            "Conformal interval summarization is reserved for a future research PR; "
            "no conformal outputs are produced in this package version."
        ),
    }


def _bayesian_posterior_summary(config: MMMConfig, fit_out: dict[str, Any]) -> dict[str, Any]:
    if config.framework != Framework.BAYESIAN:
        return {"summarized": False, "reason": "not_bayesian_framework"}
    summary = fit_out.get("summary")
    ppc = fit_out.get("ppc") if isinstance(fit_out.get("ppc"), dict) else {}
    di = ppc.get("decision_inference") if isinstance(ppc.get("decision_inference"), dict) else {}
    out: dict[str, Any] = {
        "summarized": True,
        "method": "mcmc_posterior",
        "posterior_diagnostics_ok": bool(di.get("posterior_diagnostics_ok")),
        "posterior_predictive_ok": bool(di.get("posterior_predictive_ok")),
        "rhat_max": di.get("rhat_max"),
        "ess_bulk_min": di.get("ess_bulk_min"),
        "divergences": di.get("divergences"),
    }
    if summary is not None and hasattr(summary, "beta_global_q"):
        q = summary.beta_global_q
        p10 = np.asarray(q.get("p10", []), dtype=float)
        p90 = np.asarray(q.get("p90", []), dtype=float)
        if p10.size and p90.size:
            width = p90 - p10
            out["media_coef_posterior_width_mean"] = float(np.mean(np.abs(width)))
            out["media_coef_posterior_width_max"] = float(np.max(np.abs(width)))
    pp_check = ppc.get("posterior_predictive_check")
    if isinstance(pp_check, dict):
        out["posterior_predictive_check"] = {
            k: pp_check.get(k)
            for k in ("mean_abs_gap", "empirical_coverage_p90", "std_ratio_pp_over_obs")
            if k in pp_check
        }
    out["note"] = "Posterior summaries are on the modeling scale; not prod monetary CIs."
    return out


def _parameter_uncertainty_source(
    config: MMMConfig,
    *,
    ident: dict[str, Any],
    ridge_boot: dict[str, Any],
    bayes_post: dict[str, Any],
) -> dict[str, Any]:
    if config.framework == Framework.RIDGE_BO:
        mag = _clip01(float(ident.get("instability_score", 0.0)))
        return {
            "framework": "ridge_bo",
            "magnitude_proxy": mag,
            "label": "high" if mag > 0.5 else "medium" if mag > 0.25 else "low",
            "ridge_bootstrap": ridge_boot if ridge_boot.get("summarized") else None,
            "interpretation": "Parameter uncertainty from bootstrap coef dispersion and collinearity stress.",
        }
    mag = 0.35
    if bayes_post.get("summarized"):
        w = float(bayes_post.get("media_coef_posterior_width_mean") or 0.0)
        mag = _clip01(w / (w + 0.5))
        if not bayes_post.get("posterior_diagnostics_ok"):
            mag = min(1.0, mag + 0.2)
    return {
        "framework": "bayesian",
        "magnitude_proxy": mag,
        "label": "high" if mag > 0.5 else "medium" if mag > 0.25 else "low",
        "bayesian_posterior": bayes_post if bayes_post.get("summarized") else None,
        "interpretation": "Parameter uncertainty from MCMC posterior width on coefficients and sigma.",
    }


def _experiment_uncertainty_source(extension_report: dict[str, Any]) -> dict[str, Any]:
    ew = extension_report.get("evidence_weighted_replay_summary")
    bel = extension_report.get("bayesian_experiment_likelihood_report")
    parts: list[dict[str, Any]] = []
    magnitudes: list[float] = []
    if isinstance(ew, dict) and ew.get("n_evidence_units_used"):
        n = int(ew.get("n_evidence_units_used", 0))
        mean_se = float(ew.get("mean_lift_se") or ew.get("mean_standard_error") or 1.0)
        wloss = float(ew.get("weighted_replay_loss") or 0.0)
        mag = _clip01(min(1.0, wloss / (wloss + 1.0)) * min(1.0, 1.0 / max(mean_se, 0.05)))
        magnitudes.append(mag)
        parts.append(
            {
                "path": "ridge_evidence_weighted_replay",
                "n_units": n,
                "mean_lift_se": mean_se,
                "weighted_replay_loss": wloss,
            }
        )
    if isinstance(bel, dict) and bel.get("enabled"):
        ses = bel.get("adjusted_standard_errors") or []
        if ses:
            mean_adj = float(np.mean(ses))
            mag = _clip01(min(1.0, 1.0 / max(mean_adj, 0.05)))
            magnitudes.append(mag)
        parts.append(
            {
                "path": "bayesian_experiment_likelihood",
                "n_terms": int(bel.get("n_evidence_units_used", 0)),
                "n_adjusted_se": len(ses),
            }
        )
    mag = float(np.mean(magnitudes)) if magnitudes else 0.0
    return {
        "present": bool(parts),
        "magnitude_proxy": mag,
        "label": "high" if mag > 0.5 else "medium" if mag > 0.2 else "low",
        "components": parts,
        "interpretation": "Experiment uncertainty from replay SE weighting and/or Bayesian experiment likelihood.",
    }


def _hierarchy_uncertainty_source(extension_report: dict[str, Any]) -> dict[str, Any]:
    ridge_h = extension_report.get("hierarchy_diagnostics")
    bayes_h = extension_report.get("bayesian_hierarchy_report")
    parts: list[dict[str, Any]] = []
    magnitudes: list[float] = []
    if isinstance(ridge_h, dict) and ridge_h.get("hierarchy_enabled"):
        pen = float(ridge_h.get("hierarchical_penalty_at_fit") or 0.0)
        lam = float(ridge_h.get("regularization_strength") or 0.1)
        mag = _clip01(pen / (pen + lam + 1e-6))
        magnitudes.append(mag)
        parts.append(
            {
                "path": "ridge_hierarchical_penalty",
                "regularization_strength": lam,
                "n_coef_pairs": int(ridge_h.get("n_coef_pairs", 0)),
            }
        )
    if isinstance(bayes_h, dict) and bayes_h.get("enabled"):
        gv = bayes_h.get("group_variance_summary") or {}
        sg = float(gv.get("hier_sigma_group_mean") or 0.0)
        mag = _clip01(sg / (sg + 0.5))
        magnitudes.append(mag)
        parts.append(
            {
                "path": "bayesian_hier_sigma_group",
                "group_variance_summary": gv,
                "n_pairs": int(bayes_h.get("n_hierarchy_pairs", 0)),
            }
        )
    mag = float(np.mean(magnitudes)) if magnitudes else 0.0
    return {
        "present": bool(parts),
        "magnitude_proxy": mag,
        "label": "high" if mag > 0.4 else "medium" if mag > 0.15 else "low",
        "components": parts,
        "interpretation": "Hierarchy uncertainty from group-level shrinkage variance (not causal identification).",
    }


def _allocation_uncertainty_source(extension_report: dict[str, Any]) -> dict[str, Any]:
    shock = extension_report.get("counterfactual_shock_plan")
    bel = extension_report.get("bayesian_experiment_likelihood_report")
    parts: list[dict[str, Any]] = []
    mag = 0.0
    if isinstance(shock, dict):
        plans = shock.get("plans") or shock.get("shock_plans") or []
        bridge = 0
        if isinstance(plans, list):
            for p in plans:
                if isinstance(p, dict) and p.get("allocation_role") == "computational_bridge_only":
                    bridge += 1
        if bridge:
            mag = _clip01(0.3 + 0.1 * bridge)
            parts.append({"path": "counterfactual_shock_plan", "n_bridge_allocations": bridge})
    if isinstance(bel, dict) and bel.get("enabled"):
        terms = bel.get("likelihood_terms") or []
        alloc_terms = [t for t in terms if isinstance(t, dict) and t.get("allocation_required")]
        if alloc_terms:
            mag = max(mag, _clip01(0.25 + 0.05 * len(alloc_terms)))
            parts.append(
                {"path": "bayesian_experiment_likelihood", "n_allocation_inflated_terms": len(alloc_terms)}
            )
    return {
        "present": bool(parts),
        "magnitude_proxy": mag,
        "label": "high" if mag > 0.45 else "medium" if mag > 0.2 else "low",
        "components": parts,
        "interpretation": (
            "Allocation uncertainty from nationally allocated shocks and computational bridges — "
            "not experimental subgeo truth."
        ),
    }


def build_uncertainty_propagation_report(
    config: MMMConfig,
    *,
    fit_out: dict[str, Any],
    extension_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate uncertainty from existing extension artifacts (reports only; no new inference).
    """
    up: UncertaintyPropagationConfig = config.extensions.uncertainty_propagation
    ident = extension_report.get("identifiability") if isinstance(extension_report.get("identifiability"), dict) else {}
    separability = extension_report.get("feature_separability_report")
    if not isinstance(separability, dict):
        separability = None

    if up.enabled:
        ridge_boot = _ridge_bootstrap_summary(config, ident, separability)
        bayes_post = _bayesian_posterior_summary(config, fit_out)
    else:
        ridge_boot = {"summarized": False, "reason": "propagation_disabled"}
        bayes_post = {"summarized": False, "reason": "propagation_disabled"}
    if config.framework == Framework.RIDGE_BO:
        ridge_conf = _ridge_conformal_summary(config)
    else:
        ridge_conf = {"summarized": False, "reason": "not_ridge"}

    sources = {
        "parameter_uncertainty": _parameter_uncertainty_source(
            config, ident=ident, ridge_boot=ridge_boot, bayes_post=bayes_post
        ),
        "experiment_uncertainty": _experiment_uncertainty_source(extension_report),
        "hierarchy_uncertainty": _hierarchy_uncertainty_source(extension_report),
        "allocation_uncertainty": _allocation_uncertainty_source(extension_report),
    }

    prod_forbids_ci = ridge_forbids_precise_monetary_ci(config)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "enabled": bool(up.enabled),
        "research_only": True,
        "prod_decisioning_allowed": False,
        "prod_monetary_ci_allowed": False,
        "prod_monetary_ci_forbidden": prod_forbids_ci,
        "run_environment": config.run_environment.value,
        "framework": config.framework.value,
        "uncertainty_sources": sources,
        "ridge_bootstrap_summary": ridge_boot,
        "ridge_conformal_summary": ridge_conf,
        "bayesian_posterior_summary": bayes_post,
        "warnings": list(GOVERNANCE_WARNINGS),
        "governance_warnings": list(GOVERNANCE_WARNINGS),
        "unsupported_claims": [
            "Production monetary confidence intervals from this report.",
            "Decision-grade Δμ intervals on Ridge without validated bootstrap or Bayesian draws.",
        ],
    }
    if not up.enabled:
        report["warnings"].append(
            "extensions.uncertainty_propagation.enabled=false; passive source breakdown only "
            "(enable for full Ridge bootstrap / Bayesian posterior summarization)."
        )
    return report


def build_legacy_uncertainty_buckets(
    config: MMMConfig,
    propagation: dict[str, Any],
    *,
    ident: dict[str, Any],
    cv_mae_std: float | None = None,
) -> dict[str, Any]:
    """Bridge PR 5A report into existing ``UncertaintyDecomposer`` shape."""
    param = propagation.get("uncertainty_sources", {}).get("parameter_uncertainty", {})
    exp = propagation.get("uncertainty_sources", {}).get("experiment_uncertainty", {})
    posterior_width: dict[str, float] | None = None
    bayes = propagation.get("bayesian_posterior_summary") or {}
    if bayes.get("summarized"):
        posterior_width = {
            "media_coef_width_mean": float(bayes.get("media_coef_posterior_width_mean") or 0.0),
            "media_coef_width_max": float(bayes.get("media_coef_posterior_width_max") or 0.0),
        }
    bootstrap_width = {
        "coef_dispersion_proxy": float(ident.get("instability_score", 0.0)),
        "parameter_magnitude_proxy": float(param.get("magnitude_proxy", 0.0)),
    }
    exp_scale = 1.0 / max(float(exp.get("magnitude_proxy", 0.0)) + 0.05, 0.05) if exp.get("present") else 1.0
    ud = UncertaintyDecomposer.build_report(
        posterior_width=posterior_width,
        bootstrap_width=bootstrap_width,
        cv_mae_std=cv_mae_std,
        experiment_se_scale=exp_scale,
        optimization_robustness=None,
    )
    ud["hierarchy_uncertainty"] = propagation.get("uncertainty_sources", {}).get("hierarchy_uncertainty")
    ud["allocation_uncertainty"] = propagation.get("uncertainty_sources", {}).get("allocation_uncertainty")
    ud["uncertainty_propagation_report_version"] = propagation.get("report_version")
    ud["decision_safe_intervals"] = False
    ud["prod_monetary_ci_allowed"] = False
    return ud
