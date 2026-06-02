"""Bayes-H5i real-panel convergence diagnostics tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics import (
    H5ConvergenceDiagnosticError,
    build_convergence_diagnostics_artifact,
    inspect_collinearity_diagnostics,
    inspect_panel_shape,
    inspect_scale_diagnostics,
    inspect_sparsity_diagnostics,
    run_convergence_experiment_matrix,
    validate_convergence_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import classify_convergence_status


def _sample_schema_and_df() -> tuple[pd.DataFrame, object]:
    from mmm.data.schema import PanelSchema

    df = pd.DataFrame(
        {
            "geo_id": ["G0", "G0", "G1", "G1"],
            "week_start_date": ["2022-01-03", "2022-01-10", "2022-01-03", "2022-01-10"],
            "revenue": [100.0, 110.0, 90.0, 95.0],
            "search": [1.0, 2.0, 1.5, 2.5],
            "social": [2.0, 2.0, 3.0, 3.0],
        }
    )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search", "social"), ())
    return df, schema


def test_panel_diagnostics_missing_column_fails() -> None:
    from mmm.research.bayes_h3_sandbox.h5_shadow_runner import H5ShadowRunnerError, _infer_schema

    df, _ = _sample_schema_and_df()
    bad_config = {
        "transform_registry_id": "bayes_h5_media_transform_registry_v1",
        "media_transforms_by_channel": {"search": "identity", "social": "identity"},
        "transform_mismatch_mode": "aligned",
        "panel_schema": {"geo_column": "geo_id", "week_column": "week_start_date", "target_column": "missing_y"},
    }
    with pytest.raises(H5ShadowRunnerError, match="missing"):
        _infer_schema(df, bad_config)


def test_scale_diagnostics_output_fields() -> None:
    df, schema = _sample_schema_and_df()
    out = inspect_scale_diagnostics(df, schema)
    assert "outcome_level" in out
    assert "outcome_log" in out
    assert "search" in out["media_level_by_channel"]


def test_collinearity_diagnostics_output_fields() -> None:
    df, schema = _sample_schema_and_df()
    out = inspect_collinearity_diagnostics(df, schema)
    assert "pairwise_correlations" in out
    assert "max_abs_correlation" in out


def test_sparsity_diagnostics_output_fields() -> None:
    df, schema = _sample_schema_and_df()
    out = inspect_sparsity_diagnostics(df, schema)
    assert "by_channel" in out
    assert "search" in out["by_channel"]


def test_convergence_classifier_matches_h5h_rules() -> None:
    assert classify_convergence_status(rhat_max=1.02, divergence_count=0) == "converged_diagnostic_only"
    assert classify_convergence_status(rhat_max=2.09, divergence_count=9) == "failed_convergence"


def test_artifact_production_flags_false() -> None:
    diag = build_convergence_diagnostics_artifact()
    validate_convergence_artifact(diag)
    assert diag["approved_for_prod"] is False
    assert diag["prod_decisioning_allowed"] is False


def test_no_optimizer_fields_in_artifacts() -> None:
    diag = build_convergence_diagnostics_artifact()
    matrix = run_convergence_experiment_matrix(execute_fit=False)
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert forbidden not in diag
        assert matrix.get(forbidden) is None


def test_investigation_matrix_schema_valid() -> None:
    matrix = run_convergence_experiment_matrix(execute_fit=False)
    validate_convergence_artifact(matrix)
    assert matrix["artifact_id"].startswith("BAYES_H5I")
    ids = {row["variant_id"] for row in matrix["experiments"]}
    assert "H5I-REF-H5G-FAST" in ids
    assert "H5I-REF-H5H-EXTENDED" in ids


def test_panel_shape_on_sample_panel() -> None:
    from mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics import _load_panel_and_schema
    from mmm.research.bayes_h3_sandbox.h5_shadow_runner import load_transform_config

    cfg = load_transform_config("docs/06_investigations/h5g_sample_panel_transform_config.json")
    df, schema = _load_panel_and_schema("examples/sample_panel.csv", cfg)
    summary = inspect_panel_shape(df, schema)
    assert summary["n_geos"] == 3
    assert summary["n_rows"] == 123
