"""Operational trust: calibration freshness and drift gates."""

from __future__ import annotations

import pytest

from mmm.config.schema import Framework, GovernanceWorkflowConfig, MMMConfig, RunEnvironment
from mmm.decision.service import _apply_runtime_policy_prechecks
from mmm.governance.calibration_readiness import build_calibration_readiness_report
from mmm.governance.model_release import ModelReleaseState
from mmm.governance.policy import PolicyError, runtime_policy_from_config


def test_stale_calibration_warning() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        governance=GovernanceWorkflowConfig(calibration_max_age_days=30),
    )
    er = {
        "continuous_validation_report": {
            "evidence_freshness_report": {"max_age_days": 120, "n_evidence": 1},
        },
        "ridge_fit_summary": {"coef": [0.1]},
    }
    rep = build_calibration_readiness_report(cfg, er)
    assert rep["stale_calibration_warning"]
    assert rep["recommended_action"] in ("experiment_refresh_required", "model_review_required")


def test_coefficient_shift_detection() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv", "search"]},
        governance=GovernanceWorkflowConfig(coefficient_shift_threshold=0.1),
    )
    er = {"ridge_fit_summary": {"coef": [1.0, 0.0]}}
    ref = {"ridge_fit_summary": {"coef": [0.1, 0.0]}}
    rep = build_calibration_readiness_report(cfg, er, historical_reference=ref)
    assert rep["coefficient_shift_score"] >= 0.1


def test_replay_trend_detection() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        governance=GovernanceWorkflowConfig(replay_miss_threshold=0.25),
    )
    er = {
        "ridge_fit_summary": {"coef": [0.1]},
        "continuous_validation_report": {
            "experiment_comparisons": [
                {"classification": "severe_miss"},
                {"classification": "aligned"},
                {"classification": "severe_miss"},
            ],
        },
    }
    rep = build_calibration_readiness_report(cfg, er)
    assert rep["replay_miss_rate"] is not None
    assert float(rep["replay_miss_rate"]) >= 0.5


def test_optional_planning_block() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        governance=GovernanceWorkflowConfig(
            require_review_on_drift=True,
            coefficient_shift_threshold=0.01,
        ),
    )
    er = {"ridge_fit_summary": {"coef": [2.0]}}
    ref = {"ridge_fit_summary": {"coef": [0.01]}}
    rep = build_calibration_readiness_report(cfg, er, historical_reference=ref)
    assert rep["blocks_planning_allowed"] is True


def test_defaults_unchanged_warnings_only() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    er = {"ridge_fit_summary": {"coef": [0.1]}}
    rep = build_calibration_readiness_report(cfg, er)
    assert rep["blocks_planning_allowed"] is False
    assert rep["require_review_on_drift"] is False


def test_prod_decide_blocked_when_drift_requires_review() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        run_environment=RunEnvironment.PROD,
        data={"channel_columns": ["tv"]},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
        cv={"mode": "rolling"},
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        governance=GovernanceWorkflowConfig(require_review_on_drift=True),
    )
    er = {
        "model_release": {"state": ModelReleaseState.PLANNING_ALLOWED.value},
        "panel_qa": {"max_severity": "info"},
        "calibration_summary": {"replay_calibration_active": True, "n_units": 1},
        "experiment_matching": {"n_matched": 1},
        "ridge_fit_summary": {"coef": [0.1], "model_form": "semi_log"},
        "calibration_readiness_report": {
            "blocks_planning_allowed": True,
            "recommended_action": "model_review_required",
        },
    }
    policy = runtime_policy_from_config(cfg)
    with pytest.raises(PolicyError, match="drift review"):
        _apply_runtime_policy_prechecks(cfg, er, policy)


def test_readiness_always_has_recommended_action() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    rep = build_calibration_readiness_report(cfg, {"ridge_fit_summary": {"coef": [0.1]}})
    assert rep["recommended_action"] in (
        "monitor",
        "recalibration_recommended",
        "experiment_refresh_required",
        "model_review_required",
    )
