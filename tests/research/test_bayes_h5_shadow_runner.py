"""Bayes-H5f shadow-run execution harness tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_shadow_protocol import validate_shadow_run_record
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import (
    H5ShadowRunnerError,
    DEFAULT_FIXTURE_TRANSFORM_CONFIG,
    ShadowRunRequest,
    _config_from_panel,
    build_shadow_run_artifact,
    resolve_sampler_profile,
    run_fixture_dry_run_shadow,
    validate_shadow_run_artifact_file,
    write_shadow_run_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    PANEL_CONTEXT_REAL,
    PANEL_CONTEXT_SYNTHETIC_FIXTURE,
    build_shadow_trust_diagnostics,
    classify_convergence_status,
    derive_real_panel_transform_warning_codes,
    derive_synthetic_transform_warning_codes,
    evidence_promotion_allowed,
)
from mmm.research.bayes_h3_sandbox.fixtures import toy_sandbox_bundle


def _valid_transform_config() -> dict:
    return dict(DEFAULT_FIXTURE_TRANSFORM_CONFIG)


def _base_request(**overrides: object) -> ShadowRunRequest:
    cfg, schema, df = toy_sandbox_bundle()
    base = ShadowRunRequest(
        panel_id="test_panel",
        dataset_snapshot_id="snap-test-001",
        transform_config=_valid_transform_config(),
        panel_df=df,
        execute_fit=False,
        artifact_type="real_panel_shadow_artifact",
    )
    for key, val in overrides.items():
        object.__setattr__(base, key, val)
    return base


def test_missing_dataset_snapshot_id_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="dataset_snapshot_id"):
        build_shadow_run_artifact(_base_request(dataset_snapshot_id=""))


def test_missing_panel_id_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="panel_id"):
        build_shadow_run_artifact(_base_request(panel_id=""))


def test_missing_transform_config_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="transform_config"):
        build_shadow_run_artifact(_base_request(transform_config={}))


def test_h5_disabled_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="enable_h5_sandbox"):
        build_shadow_run_artifact(_base_request(enable_h5_sandbox=False))


def test_wrong_model_spec_version_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="model_spec_version"):
        build_shadow_run_artifact(_base_request(model_spec_version="bayes_h3_hierarchical_mvp_v1"))


def test_production_flags_true_fails() -> None:
    with pytest.raises(H5ShadowRunnerError, match="approved_for_prod"):
        build_shadow_run_artifact(
            _base_request(requested_production_flags={"approved_for_prod": True}),
        )


@patch("mmm.research.bayes_h3_sandbox.h5_shadow_runner.run_sandbox_fit")
def test_dry_run_artifact_validates_against_h5e_schema(mock_fit, tmp_path) -> None:
    mock_fit.return_value = {
        "posterior_summary": {"model_kind": "bayes_h5_hierarchical_sandbox_v1"},
        "convergence_diagnostics": {"rhat_max": 1.01},
        "pooling_diagnostics": {"pooling_mode": "partial"},
        "h5_transform_diagnostics": {
            "transform_mismatch_detected": False,
            "transforms_aligned": True,
        },
        "diagnostic_trust_report": {"trust_report_kind": "bayes_h3_diagnostic_stub"},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "decision_surface": None,
    }
    out = tmp_path / "dry.json"
    artifact = run_fixture_dry_run_shadow(
        execute_fit=True,
        fast_mcmc=True,
        output_path=out,
    )
    validate_shadow_run_artifact_file(artifact)
    assert artifact["artifact_type"] == "dry_run_shadow_artifact"
    shadow = artifact["shadow_run"]
    assert shadow["dataset_snapshot_id"] == "synthetic_fixture_only"
    assert shadow["panel_id"] == "synthetic_h5_shadow_fixture"
    assert shadow["production_flags"]["hard_gate"] is False
    trust = shadow["trust_report_candidate_diagnostics"]
    assert trust.get("production_trust_report") is None
    assert "warning_codes" in trust


def test_dry_run_no_fit_schema_path(tmp_path) -> None:
    artifact = run_fixture_dry_run_shadow(execute_fit=False, output_path=tmp_path / "nfit.json")
    assert artifact["execute_fit"] is False
    validate_shadow_run_record(artifact["shadow_run"])


def test_panel_schema_from_transform_config() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "geo_id": ["G0", "G0"],
            "week_start_date": ["2022-01-03", "2022-01-10"],
            "revenue": [100.0, 101.0],
            "search": [1.0, 2.0],
            "social": [1.0, 2.0],
        }
    )
    cfg = dict(DEFAULT_FIXTURE_TRANSFORM_CONFIG)
    cfg["media_transforms_by_channel"] = {"search": "identity", "social": "identity"}
    cfg["panel_schema"] = {
        "geo_column": "geo_id",
        "week_column": "week_start_date",
        "target_column": "revenue",
    }
    artifact = build_shadow_run_artifact(
        ShadowRunRequest(
            panel_id="schema_test_panel",
            dataset_snapshot_id="snap-schema-test",
            transform_config=cfg,
            panel_df=df,
            execute_fit=False,
            artifact_type="real_panel_shadow_artifact",
        ),
    )
    assert artifact["artifact_type"] == "real_panel_shadow_artifact"
    validate_shadow_run_record(artifact["shadow_run"])


def test_production_flags_false_on_envelope() -> None:
    artifact = build_shadow_run_artifact(
        _base_request(artifact_type="dry_run_shadow_artifact", execute_fit=False),
    )
    flags = artifact.get("production_flags") or {}
    assert flags.get("approved_for_prod") is False
    assert flags.get("prod_decisioning_allowed") is False
    assert flags.get("hard_gate") is False
    assert artifact["shadow_run"]["production_flags"]["approved_for_prod"] is False


def test_no_optimizer_fields_on_envelope() -> None:
    artifact = build_shadow_run_artifact(
        _base_request(artifact_type="dry_run_shadow_artifact", execute_fit=False),
    )
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation", "recommendation"):
        assert forbidden not in artifact
        assert forbidden not in artifact["shadow_run"]


def test_real_panel_no_synthetic_transform_mismatch_by_default() -> None:
    cfg = {
        **DEFAULT_FIXTURE_TRANSFORM_CONFIG,
        "media_transforms_by_channel": {"search": "identity", "social": "identity"},
    }
    codes = derive_real_panel_transform_warning_codes(cfg)
    assert "h5:transform_mismatch:adstock" not in codes
    assert "h5:transform_mismatch:saturation" not in codes
    assert "h5:transform_unknown:real_panel" in codes
    assert "h5:transform_assumption:identity" in codes


def test_real_panel_transform_assumption_diagnostics() -> None:
    cfg = {
        "transform_registry_id": "bayes_h5_media_transform_registry_v1",
        "media_transforms_by_channel": {"search": "identity", "tv": "geometric_adstock"},
        "transform_mismatch_mode": "aligned",
    }
    codes = derive_real_panel_transform_warning_codes(cfg)
    assert "h5:transform_assumption:identity" in codes
    assert "h5:transform_assumption:adstock" in codes


def test_synthetic_mismatch_world_still_emits_mismatch() -> None:
    cfg = dict(DEFAULT_FIXTURE_TRANSFORM_CONFIG)
    cfg["transform_mismatch_mode"] = "intentional_mismatch"
    h5_diag = {
        "transform_mismatch_detected": True,
        "generative_transform_expected": "adstock_then_saturation",
    }
    codes = derive_synthetic_transform_warning_codes(cfg, h5_diag)
    assert "h5:transform_mismatch:adstock" in codes or "h5:transform_mismatch:saturation" in codes


def test_frozen_policy_no_fit_includes_policy_metadata() -> None:
    policy_path = Path("docs/06_investigations/h5m_sample_panel_shadow_policy.json")
    if not policy_path.is_file():
        pytest.skip("frozen policy file missing")
    from mmm.research.bayes_h3_sandbox.h5_shadow_policy import load_shadow_policy, policy_to_shadow_request

    policy = load_shadow_policy(policy_path)
    request = policy_to_shadow_request(
        policy,
        policy_path=policy_path,
        execute_fit=False,
    )
    artifact = build_shadow_run_artifact(request)
    assert artifact.get("policy_id") == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert artifact.get("h5_geometry_config_applied", {}).get("sigma_policy") == "sigma_floor"
    assert artifact.get("sampler_profile_applied", {}).get("draws") == 600
    assert artifact["channel_policy_applied"]["kept_channels"] == ["search", "social"]
    assert artifact["production_flags"]["approved_for_prod"] is False


def test_policy_path_cli_rejects_conflicting_args() -> None:
    from mmm.research.bayes_h3_sandbox.h5_shadow_runner import main

    with pytest.raises(SystemExit, match="policy-path fully specifies"):
        main(
            [
                "--policy-path",
                "docs/06_investigations/h5m_sample_panel_shadow_policy.json",
                "--panel-path",
                "examples/sample_panel.csv",
            ]
        )


def test_convergence_status_classification() -> None:
    assert classify_convergence_status(rhat_max=1.02, divergence_count=0) == "converged_diagnostic_only"
    assert classify_convergence_status(rhat_max=1.08, divergence_count=2) == "weak_convergence"
    assert classify_convergence_status(rhat_max=2.09, divergence_count=9) == "failed_convergence"
    assert evidence_promotion_allowed("converged_diagnostic_only") is True
    assert evidence_promotion_allowed("failed_convergence") is False


def test_failed_convergence_blocks_evidence_promotion() -> None:
    artifact = {
        "convergence_diagnostics": {"rhat_max": 2.09, "divergence_count": 9},
        "h5_transform_diagnostics": {
            "transform_mismatch_detected": False,
            "generative_transform_expected": "unknown",
        },
        "posterior_summary": {},
    }
    trust = build_shadow_trust_diagnostics(
        artifact,
        DEFAULT_FIXTURE_TRANSFORM_CONFIG,
        panel_context=PANEL_CONTEXT_REAL,
    )
    assert trust["trust_report_candidate_fields"]["convergence_status"] == "failed_convergence"
    assert trust["trust_report_candidate_fields"]["evidence_promotion_allowed"] is False
    assert "h5:convergence:failed" in trust["warning_codes"]
    assert "h5:evidence:blocked" in trust["warning_codes"]


def test_extended_mcmc_maps_to_extended_sampler_profile() -> None:
    _, schema, df = toy_sandbox_bundle()
    cfg, profile = _config_from_panel(df, schema, fast_mcmc=False, extended_mcmc=True)
    assert profile == "extended"
    assert cfg.bayesian.draws == 600
    assert cfg.bayesian.tune == 600
    assert cfg.bayesian.chains == 4
    assert resolve_sampler_profile(fast_mcmc=False, extended_mcmc=True)[0] == "extended"


@patch("mmm.research.bayes_h3_sandbox.h5_shadow_runner.run_sandbox_fit")
def test_real_panel_shadow_record_has_real_panel_diagnostics(mock_fit) -> None:
    mock_fit.return_value = {
        "posterior_summary": {"mu_channel_mean": {"search": 0.1}},
        "convergence_diagnostics": {"rhat_max": 1.02, "divergence_count": 0, "ess_bulk_min": 200},
        "pooling_diagnostics": {},
        "h5_transform_diagnostics": {
            "transform_mismatch_detected": False,
            "generative_transform_expected": "unknown",
            "panel_context": "real_panel",
        },
    }
    _, schema, df = toy_sandbox_bundle()
    artifact = build_shadow_run_artifact(
        ShadowRunRequest(
            panel_id="real_test",
            dataset_snapshot_id="snap-1",
            transform_config=dict(DEFAULT_FIXTURE_TRANSFORM_CONFIG),
            panel_df=df,
            execute_fit=True,
            fast_mcmc=True,
            artifact_type="real_panel_shadow_artifact",
        ),
    )
    shadow = artifact["shadow_run"]
    assert "real_panel_diagnostics" in shadow
    assert shadow["real_panel_diagnostics"]["panel_context"] == PANEL_CONTEXT_REAL
    codes = shadow["trust_report_candidate_diagnostics"]["warning_codes"]
    assert "h5:transform_mismatch:adstock" not in codes


@patch("mmm.research.bayes_h3_sandbox.h5_shadow_runner.run_sandbox_fit")
def test_synthetic_fixture_uses_synthetic_trust_codes(mock_fit, tmp_path) -> None:
    mock_fit.return_value = {
        "posterior_summary": {},
        "convergence_diagnostics": {"rhat_max": 1.01, "divergence_count": 0},
        "pooling_diagnostics": {},
        "h5_transform_diagnostics": {
            "transform_mismatch_detected": False,
            "generative_transform_expected": "linear",
        },
    }
    artifact = run_fixture_dry_run_shadow(
        execute_fit=True, fast_mcmc=True, output_path=tmp_path / "dry.json"
    )
    codes = artifact["shadow_run"]["trust_report_candidate_diagnostics"]["warning_codes"]
    assert "h5:transform_unknown:real_panel" not in codes
    assert "h5:recovery_candidate:stable_research_only" in codes


@pytest.mark.slow
@pytest.mark.pymc
def test_fixture_shadow_run_fast_mcmc(tmp_path) -> None:
    out = tmp_path / "pymc_shadow.json"
    artifact = run_fixture_dry_run_shadow(execute_fit=True, fast_mcmc=True, output_path=out)
    validate_shadow_run_artifact_file(artifact)
    assert artifact["shadow_run"]["model_spec_version"] == H5_MODEL_SPEC_VERSION
    assert artifact["shadow_run"]["trust_report_candidate_diagnostics"]["warning_codes"]
