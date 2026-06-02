"""Tests for frozen H5 shadow policy loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_shadow_policy import (
    H5ShadowPolicyError,
    build_transform_config_from_policy,
    load_shadow_policy,
    policy_to_shadow_runner_args,
    validate_shadow_policy,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import (
    build_shadow_run_artifact,
    validate_shadow_run_artifact_file,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    classify_convergence_status,
    evidence_promotion_allowed,
)

POLICY_PATH = Path("docs/06_investigations/h5m_sample_panel_shadow_policy.json")


def _load_policy() -> dict:
    return load_shadow_policy(POLICY_PATH)


def test_policy_loads() -> None:
    policy = _load_policy()
    assert policy["policy_id"] == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert policy["enable_h5_sandbox"] is True


def test_required_fields_enforced() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="dataset_snapshot_id"):
        validate_shadow_policy({**policy, "dataset_snapshot_id": ""})
    with pytest.raises(H5ShadowPolicyError, match="panel_id"):
        validate_shadow_policy({**policy, "panel_id": ""})
    with pytest.raises(H5ShadowPolicyError, match="geometry_config"):
        validate_shadow_policy({**policy, "geometry_config": None})
    with pytest.raises(H5ShadowPolicyError, match="transform_config"):
        validate_shadow_policy({**policy, "transform_config": {}})


def test_production_flags_true_fail() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="approved_for_prod"):
        validate_shadow_policy(
            {
                **policy,
                "production_flags": {**policy["production_flags"], "approved_for_prod": True},
            }
        )


def test_implicit_channel_dropping_fails() -> None:
    policy = _load_policy()
    bad = dict(policy["channel_policy"])
    bad.pop("explicit_dropped_channels")
    with pytest.raises(H5ShadowPolicyError, match="explicit_dropped_channels"):
        validate_shadow_policy({**policy, "channel_policy": bad})


def test_no_silent_dropping_required() -> None:
    policy = _load_policy()
    bad = {**policy["channel_policy"], "no_silent_dropping": False}
    with pytest.raises(H5ShadowPolicyError, match="no_silent_dropping"):
        validate_shadow_policy({**policy, "channel_policy": bad})


def test_wrong_model_spec_version_fails() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="model_spec_version"):
        validate_shadow_policy({**policy, "model_spec_version": "other"})


def test_unsupported_tau_in_geometry_fails() -> None:
    policy = _load_policy()
    geom = {**policy["geometry_config"], "tau_parameterization": "invalid"}
    with pytest.raises(H5ShadowPolicyError, match="tau_parameterization"):
        validate_shadow_policy({**policy, "geometry_config": geom})


def test_unsupported_sigma_policy_fails() -> None:
    policy = _load_policy()
    geom = {**policy["geometry_config"], "sigma_policy": "bad_sigma"}
    with pytest.raises(H5ShadowPolicyError, match="sigma_policy"):
        validate_shadow_policy({**policy, "geometry_config": geom})


def test_policy_maps_to_shadow_runner_args() -> None:
    policy = _load_policy()
    args = policy_to_shadow_runner_args(policy, policy_path=POLICY_PATH)
    assert args["panel_id"] == "examples_mmm_sample_panel_v1"
    assert args["extended_mcmc"] is True
    assert args["policy_id"] == policy["policy_id"]
    assert args["sandbox_model_overrides"]["h5_geometry_config"]["sigma_policy"] == "sigma_floor"
    assert "channel_policy" in args["transform_config"]


def test_transform_config_includes_panel_schema() -> None:
    policy = _load_policy()
    tc = build_transform_config_from_policy(policy)
    assert tc["panel_schema"]["week_column"] == "week_start_date"
    assert "tv" in tc["media_transforms_by_channel"]


def test_evidence_promotion_only_when_converged() -> None:
    assert evidence_promotion_allowed("converged_diagnostic_only") is True
    assert evidence_promotion_allowed("weak_convergence") is False
    status = classify_convergence_status(rhat_max=1.02, divergence_count=4)
    assert evidence_promotion_allowed(status) is False


@patch("mmm.research.bayes_h3_sandbox.h5_shadow_runner.run_sandbox_fit")
def test_policy_replay_artifact_schema(mock_fit, tmp_path) -> None:
    mock_fit.return_value = {
        "posterior_summary": {"model_kind": "bayes_h5_hierarchical_sandbox_v1"},
        "convergence_diagnostics": {"rhat_max": 1.01, "divergence_count": 0, "ess_bulk_min": 500},
        "pooling_diagnostics": {"pooling_mode": "partial"},
        "h5_transform_diagnostics": {"transform_mismatch_detected": False},
        "h5_geometry_diagnostics": {"sigma_policy": "sigma_floor"},
        "diagnostic_trust_report": {"trust_report_kind": "bayes_h3_diagnostic_stub"},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "decision_surface": None,
    }
    from mmm.research.bayes_h3_sandbox.h5_shadow_policy import policy_to_shadow_request

    policy = _load_policy()
    request = policy_to_shadow_request(
        policy,
        policy_path=POLICY_PATH,
        output_path=tmp_path / "replay.json",
        execute_fit=True,
    )
    artifact = build_shadow_run_artifact(request)
    validate_shadow_run_artifact_file(artifact)
    assert artifact["artifact_type"] == "real_panel_shadow_artifact"
    assert artifact["policy_id"] == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert artifact["source_policy_path"] == str(POLICY_PATH)
    assert artifact["convergence_status"] == "converged_diagnostic_only"
    assert artifact["evidence_promotion_allowed"] is True
    assert artifact["production_flags"]["approved_for_prod"] is False
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert artifact.get(forbidden) is None
