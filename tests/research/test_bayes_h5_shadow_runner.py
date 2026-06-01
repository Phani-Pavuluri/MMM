"""Bayes-H5f shadow-run execution harness tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_shadow_protocol import validate_shadow_run_record
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import (
    H5ShadowRunnerError,
    DEFAULT_FIXTURE_TRANSFORM_CONFIG,
    ShadowRunRequest,
    build_shadow_run_artifact,
    run_fixture_dry_run_shadow,
    validate_shadow_run_artifact_file,
    write_shadow_run_artifact,
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


@pytest.mark.slow
@pytest.mark.pymc
def test_fixture_shadow_run_fast_mcmc(tmp_path) -> None:
    out = tmp_path / "pymc_shadow.json"
    artifact = run_fixture_dry_run_shadow(execute_fit=True, fast_mcmc=True, output_path=out)
    validate_shadow_run_artifact_file(artifact)
    assert artifact["shadow_run"]["model_spec_version"] == H5_MODEL_SPEC_VERSION
    assert artifact["shadow_run"]["trust_report_candidate_diagnostics"]["warning_codes"]
