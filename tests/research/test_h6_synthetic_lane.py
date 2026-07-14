"""Tests for Bayes-H6 production-shaped synthetic validation lane."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mmm.research.h6_synthetic import (
    H6_PILOT_WORLD_IDS,
    get_h6_world,
    list_h6_world_ids,
    materialize_h6_panel,
)
from mmm.research.h6_synthetic.benchmark_harness import (
    build_h6_confounding_comparison,
    run_h6_benchmark_pair,
    run_ridge_h6_benchmark,
)
from mmm.research.h6_synthetic.benchmark_matrix import (
    H6F_ARTIFACT_KIND_MATRIX,
    H6F_PRODUCTION_FLAGS,
    build_h6f_benchmark_matrix,
    build_h6f_control_confounding_summary,
)
from mmm.research.h6_synthetic.production_shapes import (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    build_production_shaped_world,
    forbidden_claims_for_h6_world,
    materialize_h6_truth_artifact,
)
from mmm.research.h6_synthetic.vertical_controls import get_vertical_profile


def test_h6_pilot_world_registry() -> None:
    ids = list_h6_world_ids()
    assert ids == H6_PILOT_WORLD_IDS
    assert len(ids) == 5


def test_pilot_panel_shape_and_columns() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    assert len(panel) == spec.n_geos * spec.n_weeks
    assert set(spec.channels).issubset(panel.columns)
    assert "geo_id" in panel.columns
    assert "week_start_date" in panel.columns
    assert "revenue" in panel.columns
    for ctrl in spec.active_controls:
        assert ctrl in panel.columns


def test_production_scale_spec_dims() -> None:
    spec = build_production_shaped_world(
        world_id="WORLD-H6-PROD-SMOKE",
        scale="production",
        vertical_id="retail",
        stress_variant="full_controls",
        panel_seed=99,
    )
    assert spec.n_geos == 200
    assert 104 <= spec.n_weeks <= 156
    assert 5 <= len(spec.channels) <= 10


def test_omitted_controls_stress_has_forbidden_claims() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_OMITTED)
    truth = materialize_h6_truth_artifact(spec)
    assert truth["production_flags"]["approved_for_prod"] is False
    assert truth["forbidden_claims"]
    profile = get_vertical_profile("retail")
    omitted = set(spec.control_truth.get("omitted_controls") or [])
    assert set(profile.required_controls).issubset(omitted)


def test_vertical_profiles_have_required_controls() -> None:
    for vid in ("retail", "cpg", "auto"):
        p = get_vertical_profile(vid)
        assert p.required_controls
        assert p.control_effects_log


def test_ridge_benchmark_on_pilot_panel() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    ridge = run_ridge_h6_benchmark(spec, panel)
    assert ridge["model"] == "ridge_bo"
    assert ridge["approved_for_prod"] is False
    assert ridge["optimizer_enabled"] is False
    assert ridge["prediction_rmse"] is not None


@patch("mmm.research.h6_synthetic.benchmark_harness.run_sandbox_fit")
def test_benchmark_pair_mocks_h5(mock_fit) -> None:
    from mmm.research.bayes_h3_sandbox.labels import apply_research_only_envelope

    mock_fit.return_value = apply_research_only_envelope(
        {
            "convergence_diagnostics": {"divergences": 0, "rhat_max": 1.01},
            "idata": None,
        }
    )
    out = run_h6_benchmark_pair(WORLD_H6_PILOT_RETAIL_FULL, run_h5=True, fast_mcmc=True)
    assert out["ridge_benchmark"]["prediction_rmse"] is not None
    assert out["h5_benchmark"] is not None
    assert out["production_flags"]["ridge_remains_production_baseline"] is True
    mock_fit.assert_called_once()


def test_confounding_comparison_ridge_only() -> None:
    out = build_h6_confounding_comparison(run_h5=False)
    assert out["artifact_kind"] == "BAYES_H6_CONFOUNDING_STRESS"
    assert len(out["world_comparisons"]) == 4
    variants = {c["stress_variant"] for c in out["world_comparisons"]}
    assert "full_controls" in variants
    assert "omitted_controls" in variants


def test_panel_max_correlation_collinear_block() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    cols = [c for c in ("display", "ctv") if c in panel.columns]
    if len(cols) >= 2:
        corr = panel[cols].corr().iloc[0, 1]
        assert abs(corr) > 0.5


@patch("mmm.research.h6_synthetic.benchmark_matrix.run_h5_h6_benchmark")
@patch("mmm.research.h6_synthetic.benchmark_matrix.run_ridge_h6_benchmark")
def test_h6f_matrix_schema(mock_ridge, mock_h5) -> None:
    mock_ridge.return_value = {
        "prediction_rmse": 1.0,
        "prediction_wmape": 0.1,
        "ridge_coef_recovery": {"coef_sign_match_rate_vs_mu": 0.8},
        "ridge_lift_recovery": {},
        "max_channel_correlation": 0.9,
        "collinearity_sensitive": True,
    }
    mock_h5.return_value = None
    matrix = build_h6f_benchmark_matrix(h5_world_ids=())
    assert matrix["artifact_kind"] == H6F_ARTIFACT_KIND_MATRIX
    assert matrix["production_flags"] == H6F_PRODUCTION_FLAGS
    assert len(matrix["matrix_rows"]) == 5
    omitted = next(
        r for r in matrix["matrix_rows"] if r["world_id"] == WORLD_H6_PILOT_RETAIL_OMITTED
    )
    assert omitted["forbidden_claims"]
    assert omitted["ridge"]["recommendations_emitted"] is False
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_OMITTED)
    assert forbidden_claims_for_h6_world(spec)


def test_h6f_confounding_summary_forbidden_claims() -> None:
    with patch("mmm.research.h6_synthetic.benchmark_matrix.run_ridge_h6_benchmark") as mock_ridge:
        mock_ridge.return_value = {
            "prediction_rmse": 1.0,
            "ridge_coef_recovery": {"coef_mu_mae_vs_true_mu": 0.5},
        }
        summary = build_h6f_control_confounding_summary(run_h5=False)
    assert summary["artifact_kind"] == "BAYES_H6F_CONTROL_CONFOUNDING_SUMMARY"
    omitted = next(
        c for c in summary["world_comparisons"] if c["control_variant"] == "omitted_controls"
    )
    assert omitted["forbidden_claims"]


@pytest.mark.slow
def test_h5_benchmark_pilot_integration() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    from mmm.research.h6_synthetic.benchmark_harness import run_h5_h6_benchmark

    h5 = run_h5_h6_benchmark(spec, panel, fast_mcmc=True)
    assert h5["model"] == "bayes_h5_sandbox"
    assert h5["approved_for_prod"] is False
    rec = h5.get("h5_recovery") or {}
    assert rec.get("outputs_are_diagnostic_only") is True
