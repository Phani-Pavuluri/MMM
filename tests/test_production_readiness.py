"""Production readiness certification rollup."""

from __future__ import annotations

import pytest

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm, NormalizationProfile, RunEnvironment
from mmm.governance.policy import PolicyError
from mmm.governance.production_readiness import (
    build_production_readiness_report,
    require_production_readiness_for_prod_decide,
)
from mmm.governance.synthetic_certification import run_synthetic_certification_suite
from mmm.optimization.optimizer_certification import build_optimizer_certification_report


def _cfg(**kwargs: object) -> MMMConfig:
    base = {
        "framework": Framework.RIDGE_BO,
        "data": {"channel_columns": ["tv", "search"]},
    }
    base.update(kwargs)
    return MMMConfig(**base)  # type: ignore[arg-type]


def _minimal_extension_report(*, include_optimizer: bool = True) -> dict:
    synth = run_synthetic_certification_suite(mode="exact")
    er: dict = {
        "ridge_fit_summary": {
            "coef": [0.1, 0.2],
            "intercept": [4.6],
            "model_form": "semi_log",
            "best_params": {"decay": 0.1, "hill_half": 1e6, "hill_slope": 1.0},
        },
        "transform_policy": {"adstock": "geometric", "saturation": "hill", "policy_version": "mmm_transform_policy_v1"},
        "data_fingerprint": {"sha256_combined": "a" * 64},
        "governance": {"approved_for_optimization": True},
        "model_release": {"state": "planning_allowed"},
        "calibration_readiness_report": {"stale_calibration_warning": None},
        "calibration_summary": {"replay_generalization_gap_severity": "low"},
        "synthetic_certification_report": synth,
        "reproducibility_certification_report": {
            "self_certification": False,
            "certification_status": "pass",
            "reproducibility_evidence": True,
            "identical_output": True,
        },
    }
    if include_optimizer:
        er["optimizer_certification_report"] = build_optimizer_certification_report()
    return er


def test_production_readiness_approved_when_contract_complete() -> None:
    rep = build_production_readiness_report(_cfg(model_form=ModelForm.SEMI_LOG), _minimal_extension_report())
    assert rep["decision_contract_valid"] is True
    assert rep["synthetic_certification_level"] == "exact"
    assert rep["approved_for_prod"] is True


def test_inexact_synthetic_blocks_approval() -> None:
    er = _minimal_extension_report()
    er["synthetic_certification_report"] = {
        "certification_status": "fail",
        "certification_level": "incomplete",
    }
    rep = build_production_readiness_report(_cfg(), er)
    assert rep["approved_for_prod"] is False
    assert "synthetic_certification_not_exact" in rep["blocked_reasons"]


def test_strict_mode_requires_optimizer_cert() -> None:
    er = _minimal_extension_report(include_optimizer=False)
    rep = build_production_readiness_report(
        _cfg(governance={"require_production_certification": True}),
        er,
    )
    assert rep["approved_for_prod"] is False
    assert "optimizer_certification_missing" in rep["blocked_reasons"]


def test_self_repro_only_blocks_under_strict() -> None:
    er = _minimal_extension_report()
    er["reproducibility_certification_report"] = {
        "self_certification": True,
        "identical_output": None,
        "reproducibility_evidence": False,
    }
    rep = build_production_readiness_report(_cfg(governance={"require_production_certification": True}), er)
    assert "reproducibility_self_certification_only" in rep["blocked_reasons"]


def test_directional_fallback_warns_but_may_approve() -> None:
    er = _minimal_extension_report()
    er["optimizer_certification_report"] = {
        "certification_status": "pass",
        "certification_mode": "directional_fallback",
        "scenarios": [],
    }
    rep = build_production_readiness_report(_cfg(), er)
    assert rep["approved_for_prod"] is True
    assert any("directional_fallback" in w for w in rep["warnings"])


def test_prod_decide_surface_severe_warning_when_not_approved() -> None:
    from mmm.governance.production_readiness import production_readiness_decide_surface

    cfg = _cfg(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        objective={
            "normalization_profile": NormalizationProfile.STRICT_PROD,
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    er = _minimal_extension_report()
    er["synthetic_certification_report"] = {"certification_status": "fail", "certification_level": "incomplete"}
    surf = production_readiness_decide_surface(cfg, er)
    assert surf["severe_warning"] is not None
    assert "NOT APPROVED" in surf["severe_warning"]


def test_require_production_certification_blocks_prod_decide() -> None:
    cfg = _cfg(
        run_environment=RunEnvironment.PROD,
        model_form=ModelForm.SEMI_LOG,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        objective={
            "normalization_profile": NormalizationProfile.STRICT_PROD,
            "named_profile": "ridge_bo_standard_v1",
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        governance={"require_production_certification": True},
    )
    er = _minimal_extension_report(include_optimizer=False)
    with pytest.raises(PolicyError, match="production readiness"):
        require_production_readiness_for_prod_decide(cfg, er)
