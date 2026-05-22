"""Canonical run manifest: orchestration entry pointing at tiered surfaces (Phase 1.2)."""

from __future__ import annotations

from typing import Any

RUN_MANIFEST_VERSION = "mmm_run_manifest_v1"

# Keys on extension_report / trainer outputs and their artifact tier for human + machine readers.
EXTENSION_REPORT_SURFACE_TIERS: dict[str, str] = {
    "decision_bundle": "research",
    "ridge_fit_summary": "decision_input",
    "governance": "decision_input",
    "model_release": "decision_input",
    "operational_health": "decision_input",
    "post_fit_validation": "diagnostic",
    "response_diagnostics": "diagnostic",
    "curve_bundle": "diagnostic",
    "curve_bundles": "diagnostic",
    "roi_summary": "diagnostic",
    "posterior_exploration_quantity": "research",
    "uncertainty_decomposition": "research",
    "uncertainty_propagation_report": "research",
    "robust_optimization_research": "research",
    "continuous_validation_report": "diagnostic",
    "decision_validation_report": "diagnostic",
    "falsification": "diagnostic",
    "identifiability": "diagnostic",
    "feature_separability_report": "diagnostic",
    "experiment_scheduler_report": "diagnostic",
    "baselines": "diagnostic",
    "panel_qa": "diagnostic",
    "calibration_summary": "diagnostic",
    "experiment_matching": "diagnostic",
    "experiment_compatibility_report": "diagnostic",
    "evidence_weighting_report": "diagnostic",
    "counterfactual_shock_plan": "diagnostic",
    "experiment_evidence_registry_coverage": "diagnostic",
    "governance_unsupported_claims": "diagnostic",
    "evidence_weighted_replay_summary": "diagnostic",
    "bayesian_experiment_likelihood_report": "research",
    "bayesian_hierarchy_report": "research",
    "hierarchy_diagnostics": "diagnostic",
    "hierarchy_effect_summary": "diagnostic",
    "hierarchy_validation_report": "diagnostic",
    "hierarchy_governance_warnings": "diagnostic",
}


def build_run_manifest(extension_report: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """
    Single machine-readable index over a training/extension_report dict.

    Decision APIs must still consume typed decision bundles only; this manifest is for
    orchestration and audit, not a substitute for decision-tier payloads.
    """
    keys = sorted(str(k) for k in extension_report if not str(k).startswith("_"))
    tiers = {k: EXTENSION_REPORT_SURFACE_TIERS.get(k, "unknown") for k in keys}
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "extension_report_top_level_keys": keys,
        "surface_tier_by_key": tiers,
        "canonical_decision_inputs": [
            "ridge_fit_summary",
            "governance",
            "model_release",
            "operational_health",
            "panel_qa",
        ],
        "notes": [
            "Curve/decomposition/ROI-like blocks remain diagnostic unless promoted by separate decision bundle policy.",
            "Use mmm.decision.service paths for prod decision JSON; do not treat this manifest as decision truth.",
        ],
    }
