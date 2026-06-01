"""Bayes-H4 recovery worlds for H3 research sandbox (research validation only)."""

from __future__ import annotations

import pytest

from mmm.research.bayes_h3_sandbox import validate_research_only_artifact
from mmm.research.bayes_h3_sandbox.fencing import BayesSandboxGuardError, assert_optimizer_input_not_bayes_sandbox
from mmm.research.bayes_h3_sandbox.model import MODEL_KIND
from mmm.research.bayes_h3_sandbox.recovery_runner import (
    build_h4_recovery_report,
    compute_conflict_warnings,
    run_h4_recovery_world,
    validate_world_catalog,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
    WORLD_BAYES_H4_SIMPLE_POOLING,
    WORLD_BAYES_H4_SPARSE_GEO,
    get_recovery_world,
    list_recovery_worlds,
    materialize_recovery_panel,
)


@pytest.mark.parametrize("world_id", H4_WORLD_IDS)
def test_recovery_world_panel_is_deterministic(world_id: str) -> None:
    spec = get_recovery_world(world_id)
    a = materialize_recovery_panel(spec)
    b = materialize_recovery_panel(spec)
    assert a.equals(b)
    assert len(a) > 0


@pytest.mark.parametrize("world_id", H4_WORLD_IDS)
def test_recovery_world_has_known_truth_fields(world_id: str) -> None:
    spec = get_recovery_world(world_id)
    truth = spec.known_truth
    assert "true_mu_c" in truth
    assert "true_tau_c" in truth
    assert "true_beta_gc" in truth
    assert set(truth["true_mu_c"]) == set(spec.channels)
    for geo in spec.geo_order:
        assert geo in truth["true_beta_gc"]


def test_validate_world_catalog_passes() -> None:
    summary = validate_world_catalog()
    assert summary["status"] == "pass"
    assert len(summary["worlds"]) == len(H4_WORLD_IDS)


def test_conflict_world_has_conflict_warnings_without_fit() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_CONFLICTING_EVIDENCE)
    warnings = compute_conflict_warnings(spec)
    assert warnings
    assert spec.expected_diagnostic_behavior.get("conflict_warning_expected") is True


def test_recovery_report_research_only_and_no_prod_paths() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SIMPLE_POOLING)
    mock_artifact = {
        "model_kind": MODEL_KIND,
        "posterior_summary": {
            "mu_channel_mean": dict(spec.true_mu_c),
            "tau_channel_mean": dict(spec.true_tau_c),
        },
        "hierarchy_evidence_diagnostics": {
            "beta_geo_channel_mean": {
                str(i): {ch: spec.true_beta_gc[geo][ch] for ch in spec.channels} for i, geo in enumerate(spec.geo_order)
            },
        },
        "pooling_diagnostics": {"tau_channel_mean": dict(spec.true_tau_c)},
        "convergence_diagnostics": {"rhat_max": 1.01},
        "diagnostic_trust_report": {"trust_report_kind": "bayes_h3_diagnostic_stub"},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "production_recommendation": False,
        "decision_surface": None,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "decision_grade": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "bayes_h3_sandbox": True,
    }
    report = build_h4_recovery_report(spec, mock_artifact)
    validate_research_only_artifact(report)
    assert report["h4_recovery"]["beta_gc_mae"] is not None
    assert report.get("decision_surface") is None
    assert report.get("budget_recommendation") is None
    with pytest.raises(BayesSandboxGuardError):
        assert_optimizer_input_not_bayes_sandbox(report)


def test_sparse_world_expected_shrinkage_flag() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    assert spec.expected_diagnostic_behavior.get("shrinkage_expected") is True
    assert spec.sparse_geos == ("dma_sparse",)


def test_list_recovery_worlds_covers_catalog() -> None:
    ids = {w.world_id for w in list_recovery_worlds()}
    assert ids == set(H4_WORLD_IDS)


@pytest.mark.pymc
@pytest.mark.slow
def test_run_h4_recovery_simple_pooling_sample() -> None:
    pytest.importorskip("pymc")
    report = run_h4_recovery_world(WORLD_BAYES_H4_SIMPLE_POOLING)
    validate_research_only_artifact(report)
    rec = report["h4_recovery"]
    assert rec["beta_gc_mae"] is not None
    assert rec["approved_for_prod"] is False
    assert report["production_decision_surface"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_run_h4_sparse_world_shrinkage_direction() -> None:
    pytest.importorskip("pymc")
    report = run_h4_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    rec = report["h4_recovery"]
    ratio = rec.get("shrinkage_ratio_sparse")
    if ratio is not None:
        assert ratio < 1.0, f"expected shrinkage toward mu on sparse geo, got ratio={ratio}"


@pytest.mark.pymc
@pytest.mark.slow
def test_run_h4_conflict_world_emits_trust_warning() -> None:
    pytest.importorskip("pymc")
    report = run_h4_recovery_world(WORLD_BAYES_H4_CONFLICTING_EVIDENCE)
    rec = report["h4_recovery"]
    assert rec["conflict_warnings"]
    dtr = report.get("diagnostic_trust_report") or {}
    assert dtr.get("conflict_warnings")
