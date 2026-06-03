"""MIP-C1 — CalibrationSignal → Ridge diagnostic attachment contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.diagnostics.calibration_signal_attachment import (
    FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS,
    attach_calibration_evidence_context,
    build_calibration_evidence_summary,
    build_calibration_forbidden_claims,
    evaluate_signal_alignment,
    evaluate_signal_conflict,
    normalize_signal_attachment,
)
from mmm.diagnostics.ridge_diagnostic_summary import (
    format_ridge_diagnostics_cli_block,
    format_ridge_diagnostics_markdown,
)
from mmm.diagnostics.ridge_diagnostics import FORBIDDEN_OUTPUT_FIELDS

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_calibration_signal_attachment"

FIXTURE_FILES = tuple(sorted(FIXTURE_DIR.glob("*.json")))


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(params=FIXTURE_FILES, ids=lambda p: p.stem)
def mip_fixture(request: pytest.FixtureRequest) -> dict:
    return _load_fixture(request.param)


def test_fixture_attachment_matches_contract(mip_fixture: dict) -> None:
    """Each fixture: signal attaches per policy without overriding MMM."""
    stub = mip_fixture["ridge_diagnostic_stub"]
    signal = mip_fixture["calibration_signal_stub"]
    expected = mip_fixture["expected_attachment"]

    report = attach_calibration_evidence_context(stub, [signal])
    ctx = report["calibration_evidence_context"]
    attached = ctx["signals"][0]

    assert attached["alignment_status"] == expected["alignment_status"]
    assert attached["conflict_status"] == expected["conflict_status"]
    assert attached["trust_report_disposition"] == expected["trust_report_disposition"]
    for use in expected.get("allowed_use_contains") or []:
        assert use in attached["allowed_use"]

    report_forbidden = set(report.get("forbidden_claims") or [])
    for claim in mip_fixture.get("expected_forbidden_claims") or []:
        assert claim in report_forbidden or claim in attached["forbidden_claims"]

    boundary = ctx.get("trust_report_boundary") or ""
    assert "TrustReport" in boundary
    assert "override" in boundary.lower() or "bypass" in boundary.lower()

    if expected.get("collinearity_calibration_evidence_available"):
        assert report["collinearity"]["calibration_evidence_available"] is True


def test_aligned_signal_attaches_as_context() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_aligned_positive_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    ctx = report["calibration_evidence_context"]
    assert ctx["context_only"] is True
    assert ctx["signals"][0]["alignment_status"] == "aligned"
    assert report["coefficient_stability"] == fx["ridge_diagnostic_stub"]["coefficient_stability"]


def test_conflict_does_not_override_mmm() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_conflict_positive_vs_mmm_negative.json")
    stub = fx["ridge_diagnostic_stub"]
    report = attach_calibration_evidence_context(stub, [fx["calibration_signal_stub"]])
    assert report["coefficient_stability"]["media_coef_by_channel"]["tv"] == -0.08
    assert report["calibration_evidence_context"]["signals"][0]["conflict_status"] == "directional_conflict"
    assert "mmm_direction_validated_by_external_evidence" in report["forbidden_claims"]


def test_stale_signal_context_only() -> None:
    fx = _load_fixture(FIXTURE_DIR / "stale_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    sig = report["calibration_evidence_context"]["signals"][0]
    assert sig["trust_report_disposition"] == "context_only_stale"
    assert "fresh_calibration_evidence_claim" in sig["forbidden_claims"]


def test_estimand_mismatch_trust_report_only() -> None:
    fx = _load_fixture(FIXTURE_DIR / "estimand_mismatch_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    sig = report["calibration_evidence_context"]["signals"][0]
    assert sig["trust_report_disposition"] == "trust_report_only"
    assert sig["conflict_status"] == "scope_mismatch"


def test_missing_uncertainty_blocks_calibration_use() -> None:
    fx = _load_fixture(FIXTURE_DIR / "missing_uncertainty_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    forbidden = set(report["forbidden_claims"])
    assert "production_calibration_use_without_se" in forbidden
    assert "calibration_informed_attribution_without_uncertainty" in forbidden


def test_sparse_channel_forbids_clean_mmm_only_claim() -> None:
    fx = _load_fixture(FIXTURE_DIR / "sparse_channel_external_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    forbidden = set(report["forbidden_claims"])
    assert "no_clean_mmm_only_channel_claim_for_radio_despite_external_signal" in forbidden
    assert "no_separate_channel_effect_claim_for_radio" in forbidden


def test_no_optimizer_decision_surface_recommendation_fields() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_aligned_positive_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    forbidden_keys = FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
    for key in forbidden_keys:
        assert key not in report
        assert key not in report.get("calibration_evidence_context", {})


def test_trust_report_boundary_preserved() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_conflict_positive_vs_mmm_negative.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    boundary = report["calibration_evidence_context"]["trust_report_boundary"]
    assert "TrustReport" in boundary
    assert "optimizer" in boundary.lower() or "DecisionSurface" in boundary


def test_summary_and_markdown_surface_context_only() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_conflict_positive_vs_mmm_negative.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    ctx = report["calibration_evidence_context"]
    summary = build_calibration_evidence_summary(ctx)
    assert summary["context_only"] is True
    assert summary["mmm_coefficients_unchanged"] is True
    assert summary["directional_conflict_count"] == 1

    md = format_ridge_diagnostics_markdown(report)
    assert "Calibration evidence context" in md
    assert "does not override Ridge" in md

    cli = format_ridge_diagnostics_cli_block(report)
    assert any("Calibration context" in line for line in cli)


def test_build_calibration_forbidden_claims_aggregates() -> None:
    fx = _load_fixture(FIXTURE_DIR / "missing_uncertainty_signal.json")
    report = attach_calibration_evidence_context(fx["ridge_diagnostic_stub"], [fx["calibration_signal_stub"]])
    ctx = report["calibration_evidence_context"]
    aggregated = build_calibration_forbidden_claims(ctx)
    assert "automatic_mmm_recalibration_from_calibration_signal" in aggregated


def test_evaluate_helpers() -> None:
    fx = _load_fixture(FIXTURE_DIR / "geox_aligned_positive_signal.json")
    stub = fx["ridge_diagnostic_stub"]
    sig = fx["calibration_signal_stub"]
    assert evaluate_signal_alignment(sig, stub) == "aligned"
    assert evaluate_signal_conflict(sig, stub) == "none"
    row = normalize_signal_attachment(sig, stub)
    assert row["signal_id"] == sig["signal_id"]
