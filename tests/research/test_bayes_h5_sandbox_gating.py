"""Bayes-H5 sandbox gating — fail closed without explicit flag."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION, BayesSandboxGuardError
from mmm.research.bayes_h3_sandbox.fixtures import toy_sandbox_bundle
from mmm.research.bayes_h3_sandbox.model import H5_MODEL_KIND, MODEL_KIND


def test_h5_cannot_run_without_enable_flag() -> None:
    cfg, schema, df = toy_sandbox_bundle()
    with pytest.raises(BayesSandboxGuardError, match="enable_h5_sandbox"):
        run_sandbox_fit(
            cfg,
            schema,
            df,
            model_spec_version=H5_MODEL_SPEC_VERSION,
            enable_h5_sandbox=False,
        )


def test_enable_h5_without_spec_version_rejected() -> None:
    cfg, schema, df = toy_sandbox_bundle()
    with pytest.raises(BayesSandboxGuardError, match="model_spec_version"):
        run_sandbox_fit(cfg, schema, df, enable_h5_sandbox=True)


@patch("mmm.research.bayes_h3_sandbox.entrypoint.fit_h3_sandbox_hierarchical")
def test_default_path_uses_h3_fit(mock_h3) -> None:
    cfg, schema, df = toy_sandbox_bundle()
    mock_h3.return_value = {
        "model_kind": MODEL_KIND,
        "posterior_summary": {"model_kind": MODEL_KIND},
        "convergence_diagnostics": {},
        "hierarchy_evidence_diagnostics": {},
        "pooling_diagnostics": {},
        "calibration_signal_slots": {"reserved": True, "signals": []},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "production_recommendation": False,
        "decision_surface": None,
        "optimizer_ready_curves": None,
        "budget_recommendation": None,
    }
    art = run_sandbox_fit(cfg, schema, df)
    mock_h3.assert_called_once()
    assert art.get("model_spec_version") is None
    assert art["posterior_summary"]["model_kind"] == MODEL_KIND


@patch("mmm.research.bayes_h3_sandbox.entrypoint.fit_h5_sandbox_hierarchical")
def test_h5_path_sets_spec_and_research_flags(mock_h5) -> None:
    cfg, schema, df = toy_sandbox_bundle()
    mock_h5.return_value = {
        "model_kind": H5_MODEL_KIND,
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "posterior_summary": {"model_kind": H5_MODEL_KIND},
        "convergence_diagnostics": {},
        "hierarchy_evidence_diagnostics": {},
        "pooling_diagnostics": {},
        "h5_transform_diagnostics": {"transform_mismatch_detected": False},
        "calibration_signal_slots": {"reserved": True, "signals": []},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "production_recommendation": False,
        "decision_surface": None,
        "optimizer_ready_curves": None,
        "budget_recommendation": None,
    }
    art = run_sandbox_fit(
        cfg,
        schema,
        df,
        model_spec_version=H5_MODEL_SPEC_VERSION,
        enable_h5_sandbox=True,
    )
    mock_h5.assert_called_once()
    assert art["model_spec_version"] == H5_MODEL_SPEC_VERSION
    assert art["enable_h5_sandbox"] is True
    assert art["approved_for_prod"] is False
    assert art.get("decision_surface") is None
    assert art.get("budget_recommendation") is None


@pytest.mark.slow
@pytest.mark.pymc
def test_h5_aligned_adstock_pymc_smoke() -> None:
    from mmm.research.bayes_h3_sandbox.recovery_runner import run_h5_recovery_world

    report = run_h5_recovery_world("WORLD-BAYES-H5-ADSTOCK-ALIGNED", fast_mcmc=True)
    assert report["model_spec_version"] == H5_MODEL_SPEC_VERSION
    assert report.get("decision_surface") is None


@pytest.mark.slow
@pytest.mark.pymc
def test_h5_mismatch_adstock_pymc_smoke() -> None:
    from mmm.research.bayes_h3_sandbox.recovery_runner import run_h5_recovery_world

    report = run_h5_recovery_world("WORLD-BAYES-H5-ADSTOCK-MISMATCH", fast_mcmc=True)
    warnings = (report.get("h4_recovery") or {}).get("h5_diagnostic_warnings") or []
    assert any("transform_mismatch" in w for w in warnings)
