"""Tests for frozen H5 shadow policy loading and validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    HIERARCHY_FIXED_TAU,
    HIERARCHY_POOLED_CHANNEL,
)
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


def test_valid_policy_loads() -> None:
    policy = _load_policy()
    assert policy["policy_id"] == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert policy["policy_type"] == "research_shadow_policy"
    assert policy["enable_h5_sandbox"] is True


def test_required_fields_enforced() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="policy_id"):
        validate_shadow_policy({**policy, "policy_id": ""})
    with pytest.raises(H5ShadowPolicyError, match="dataset_snapshot_id"):
        validate_shadow_policy({**policy, "dataset_snapshot_id": ""})
    with pytest.raises(H5ShadowPolicyError, match="panel_id"):
        validate_shadow_policy({**policy, "panel_id": ""})
    with pytest.raises(H5ShadowPolicyError, match="panel_path"):
        validate_shadow_policy({**policy, "panel_path": ""})
    with pytest.raises(H5ShadowPolicyError, match="panel_schema"):
        validate_shadow_policy({**policy, "panel_schema": None})
    with pytest.raises(H5ShadowPolicyError, match="channel_policy"):
        validate_shadow_policy({**policy, "channel_policy": None})
    with pytest.raises(H5ShadowPolicyError, match="h5_geometry_config"):
        p = {**policy}
        p.pop("h5_geometry_config")
        validate_shadow_policy(p)
    with pytest.raises(H5ShadowPolicyError, match="transform_config"):
        validate_shadow_policy({**policy, "transform_config": {}})


def test_production_flags_true_fail_closed() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="approved_for_prod"):
        validate_shadow_policy(
            {
                **policy,
                "production_flags": {**policy["production_flags"], "approved_for_prod": True},
            }
        )


def test_wrong_model_spec_version_fails() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="model_spec_version"):
        validate_shadow_policy({**policy, "model_spec_version": "other"})


def test_enable_h5_sandbox_false_fails() -> None:
    policy = _load_policy()
    with pytest.raises(H5ShadowPolicyError, match="enable_h5_sandbox"):
        validate_shadow_policy({**policy, "enable_h5_sandbox": False})


def test_implicit_channel_dropping_fails() -> None:
    policy = _load_policy()
    bad = dict(policy["channel_policy"])
    bad.pop("dropped_channels")
    with pytest.raises(H5ShadowPolicyError, match="dropped_channels"):
        validate_shadow_policy({**policy, "channel_policy": bad})


def test_dropped_and_kept_channels_must_be_explicit() -> None:
    policy = _load_policy()
    bad = {**policy["channel_policy"], "kept_channels": []}
    with pytest.raises(H5ShadowPolicyError, match="kept_channels"):
        validate_shadow_policy({**policy, "channel_policy": bad})


def test_no_silent_dropping_required() -> None:
    policy = _load_policy()
    bad = {**policy["channel_policy"], "no_silent_dropping": False}
    with pytest.raises(H5ShadowPolicyError, match="no_silent_dropping"):
        validate_shadow_policy({**policy, "channel_policy": bad})


def test_unsupported_geometry_option_fails() -> None:
    policy = _load_policy()
    geom = {**policy["h5_geometry_config"], "tau_parameterization": "invalid"}
    with pytest.raises(H5ShadowPolicyError, match="tau_parameterization"):
        validate_shadow_policy({**policy, "h5_geometry_config": geom})


def test_unsupported_sigma_policy_fails() -> None:
    policy = _load_policy()
    geom = {**policy["h5_geometry_config"], "sigma_policy": "bad_sigma"}
    with pytest.raises(H5ShadowPolicyError, match="sigma_policy"):
        validate_shadow_policy({**policy, "h5_geometry_config": geom})


def test_ablation_geometry_cannot_be_frozen_policy() -> None:
    policy = _load_policy()
    for hier in (HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU):
        geom = {**policy["h5_geometry_config"], "hierarchy_policy": hier}
        with pytest.raises(H5ShadowPolicyError, match="ablation"):
            validate_shadow_policy({**policy, "h5_geometry_config": geom})


def test_policy_maps_to_shadow_runner_args() -> None:
    policy = _load_policy()
    args = policy_to_shadow_runner_args(policy, policy_path=POLICY_PATH)
    assert args["panel_id"] == "examples_mmm_sample_panel_v1"
    assert args["extended_mcmc"] is True
    assert args["sandbox_model_overrides"]["h5_geometry_config"]["sigma_policy"] == "sigma_floor"
    assert args["channel_policy_declared"]["dropped_channels"] == ["tv"]


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
def test_output_artifact_includes_policy_metadata(mock_fit, tmp_path) -> None:
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
    assert artifact["policy_id"] == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert artifact["source_policy_path"] == str(POLICY_PATH)
    assert artifact["h5_geometry_config_applied"]["sigma_floor"] == 0.05
    assert artifact["channel_policy_applied"]["kept_channels"] == ["search", "social"]
    assert artifact["convergence_status"] == "converged_diagnostic_only"
    assert artifact["evidence_promotion_allowed"] is True
    assert artifact["production_flags"]["approved_for_prod"] is False
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert artifact.get(forbidden) is None


def test_shadow_policy_from_recommendation_keep_all() -> None:
    from mmm.research.bayes_h3_sandbox.h5_shadow_policy import shadow_policy_from_recommendation
    from mmm.research.bayes_h3_sandbox.h5_shadow_runner import load_transform_config

    rec = {
        "artifact_id": "TEST-REC",
        "forbidden_claims": ["No production Bayes claim"],
        "recommended_shadow_policy": {
            "status": "recommended",
            "channel_policy": {"mode": "keep_all_channels", "no_silent_dropping": True},
            "h5_geometry_config": {
                "parameterization": "non_centered",
                "hierarchy_policy": "full_geo_channel_hierarchy",
                "hierarchy_strength_policy": "learned_tau",
                "tau_parameterization": "current",
                "beta_prior_policy": "current_default",
                "sigma_policy": "sigma_floor",
                "sigma_floor": 0.05,
                "likelihood_scale_policy": "prescaled_log_outcome",
            },
            "sampler_profile": {
                "profile": "extended_mcmc",
                "draws": 600,
                "tune": 600,
                "chains": 4,
                "target_accept": 0.95,
            },
        },
    }
    transform = load_transform_config(
        "docs/06_investigations/h5o_benchmark_geo_panel_transform_config.json"
    )
    policy = shadow_policy_from_recommendation(
        rec,
        policy_id="test_h5o_policy_v1",
        panel_path="examples/benchmark_geo_panel_v1.csv",
        panel_id="examples_mmm_benchmark_geo_panel_v1",
        dataset_snapshot_id="mmm-examples-benchmark-geo-panel-frozen-2022-v1",
        panel_schema={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "media_columns": ["search", "social", "tv"],
            "control_columns": [],
        },
        transform_config=transform,
    )
    validate_shadow_policy(policy)
    assert policy["channel_policy"]["mode"] == "keep_all_channels"


def test_h5r_sparse_radio_policy_validates() -> None:
    from mmm.research.bayes_h3_sandbox.h5_shadow_policy import load_shadow_policy

    path = Path("docs/06_investigations/h5r_examples_mmm_triangulation_geo_panel_v1_sparse_radio_policy.json")
    if not path.is_file():
        pytest.skip("H5r policy missing")
    policy = load_shadow_policy(path)
    validate_shadow_policy(policy)
    assert policy["channel_policy"]["mode"] == "drop_sparse_channels"
    assert policy["channel_policy"]["dropped_channels"] == ["radio"]
    assert policy["production_flags"]["approved_for_prod"] is False
    assert "radio" in " ".join(policy["forbidden_claims"]).lower()


def test_shadow_policy_from_do_not_run_fails() -> None:
    from mmm.research.bayes_h3_sandbox.h5_shadow_policy import shadow_policy_from_recommendation

    with pytest.raises(H5ShadowPolicyError, match="do_not_run"):
        shadow_policy_from_recommendation(
            {
                "forbidden_claims": ["x"],
                "recommended_shadow_policy": {"status": "do_not_run"},
            },
            policy_id="x",
            panel_path="p.csv",
            panel_id="p",
            dataset_snapshot_id="s",
            panel_schema={"media_columns": ["a"]},
            transform_config={
                "media_transforms_by_channel": {"a": "identity"},
                "transform_mismatch_mode": "aligned",
            },
        )
