"""PR 5A: uncertainty propagation reports (no optimizer changes)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError, RuntimePolicy, require_bayesian_block
from mmm.governance.uncertainty_policy import ridge_forbids_precise_monetary_ci
from mmm.uncertainty.propagation_report import (
    build_legacy_uncertainty_buckets,
    build_uncertainty_propagation_report,
)


def _ident_json() -> dict:
    return {
        "identifiability_score": 0.4,
        "instability_score": 0.25,
        "max_vif": 6.0,
        "diagnostic_context": {"bootstrap_rounds_configured": 10, "bootstrap_frac_configured": 0.85},
    }


def test_propagation_disabled_still_emits_sources() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a", "b"]},
        extensions={"uncertainty_propagation": {"enabled": False}},
    )
    rep = build_uncertainty_propagation_report(
        cfg,
        fit_out={},
        extension_report={"identifiability": _ident_json()},
    )
    assert rep["enabled"] is False
    assert "uncertainty_sources" in rep
    assert "parameter_uncertainty" in rep["uncertainty_sources"]
    assert rep["prod_monetary_ci_allowed"] is False


def test_propagation_enabled_ridge_bootstrap_summary() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a", "b"]},
        extensions={"uncertainty_propagation": {"enabled": True, "ridge_summarize_bootstrap": True}},
    )
    rep = build_uncertainty_propagation_report(
        cfg,
        fit_out={},
        extension_report={"identifiability": _ident_json()},
    )
    assert rep["ridge_bootstrap_summary"]["summarized"] is True
    assert rep["uncertainty_sources"]["parameter_uncertainty"]["framework"] == "ridge_bo"


def test_experiment_and_hierarchy_sources_present() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    ext = {
        "identifiability": _ident_json(),
        "evidence_weighted_replay_summary": {
            "n_evidence_units_used": 2,
            "mean_lift_se": 0.5,
            "weighted_replay_loss": 1.2,
        },
        "hierarchy_diagnostics": {
            "hierarchy_enabled": True,
            "hierarchical_penalty_at_fit": 0.3,
            "regularization_strength": 0.1,
            "n_coef_pairs": 2,
        },
        "counterfactual_shock_plan": {
            "plans": [{"allocation_role": "computational_bridge_only"}],
        },
    }
    rep = build_uncertainty_propagation_report(cfg, fit_out={}, extension_report=ext)
    assert rep["uncertainty_sources"]["experiment_uncertainty"]["present"]
    assert rep["uncertainty_sources"]["hierarchy_uncertainty"]["present"]
    assert rep["uncertainty_sources"]["allocation_uncertainty"]["present"]


def test_bayesian_posterior_summary_from_fit_out() -> None:
    from mmm.models.bayesian.pymc_trainer import BayesianPosteriorSummary

    summary = BayesianPosteriorSummary(
        beta_global_mean=np.array([0.1, 0.2]),
        beta_global_q={
            "p10": np.array([0.05, 0.15]),
            "p50": np.array([0.1, 0.2]),
            "p90": np.array([0.15, 0.25]),
        },
        diagnostics={},
    )
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        data={"channel_columns": ["m1", "m2"]},
        extensions={"uncertainty_propagation": {"enabled": True}},
    )
    fit_out = {
        "summary": summary,
        "ppc": {
            "decision_inference": {
                "posterior_diagnostics_ok": True,
                "posterior_predictive_ok": True,
                "rhat_max": 1.01,
                "ess_bulk_min": 400.0,
            },
            "posterior_predictive_check": {"mean_abs_gap": 0.05},
        },
    }
    rep = build_uncertainty_propagation_report(cfg, fit_out=fit_out, extension_report={})
    assert rep["bayesian_posterior_summary"]["summarized"] is True
    assert rep["uncertainty_sources"]["parameter_uncertainty"]["framework"] == "bayesian"


def test_conformal_not_implemented() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a"]},
        extensions={"uncertainty_propagation": {"enabled": True, "ridge_summarize_conformal": True}},
    )
    rep = build_uncertainty_propagation_report(cfg, fit_out={}, extension_report={})
    assert rep["ridge_conformal_summary"]["status"] == "not_implemented"


def test_legacy_buckets_bridge() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    prop = build_uncertainty_propagation_report(
        cfg,
        fit_out={},
        extension_report={"identifiability": _ident_json()},
    )
    buckets = build_legacy_uncertainty_buckets(cfg, prop, ident=_ident_json())
    assert buckets["decision_safe_intervals"] is False
    assert "hierarchy_uncertainty" in buckets


def test_prod_ridge_forbids_monetary_ci() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a"], "data_version_id": "x"},
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        extensions={"optimization_gates": {"enabled": True}},
    )
    assert ridge_forbids_precise_monetary_ci(cfg)
    rep = build_uncertainty_propagation_report(cfg, fit_out={}, extension_report={})
    assert rep["prod_monetary_ci_forbidden"] is True


def test_prod_bayesian_decisioning_still_blocked() -> None:
    policy = RuntimePolicy(
        prod=True,
        allow_bayesian_decisioning=False,
        allowed_cv_modes=["calendar"],
    )
    with pytest.raises(PolicyError):
        require_bayesian_block(Framework.BAYESIAN, policy)
