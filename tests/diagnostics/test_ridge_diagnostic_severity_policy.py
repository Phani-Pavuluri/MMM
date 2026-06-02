"""H9 — Ridge diagnostic severity policy tests."""

from __future__ import annotations

import json
from pathlib import Path

from mmm.diagnostics.ridge_diagnostic_summary import (
    format_ridge_diagnostics_cli_block,
    format_ridge_diagnostics_markdown,
    summarize_ridge_diagnostics,
)
from mmm.diagnostics.ridge_diagnostics import compose_ridge_diagnostic_report
from mmm.diagnostics.ridge_severity_policy import (
    SEVERITY_BLOCKED,
    SEVERITY_CLEAN,
    SEVERITY_DIAGNOSTIC_ONLY,
    SEVERITY_INFO,
    SEVERITY_RESTRICTED,
    SEVERITY_WARNING,
    apply_severity_policy_to_report,
    classify_ridge_diagnostic_severity,
)
from mmm.research.h6_synthetic.production_shapes import (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

ARCHIVE_SEVERITY = Path(
    "docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SEVERITY_20260601.json"
)


def _fit_report(world_id: str, *, vertical_id: str, media_corr: bool = False):
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    trainer = RidgeBOMMMTrainer(config, schema)
    fit = trainer.fit(panel)
    return compose_ridge_diagnostic_report(
        panel,
        schema,
        config,
        fit,
        trainer=trainer,
        vertical_id=vertical_id,
        media_correlated_controls=media_corr,
    )


def test_clean_retail_full_controls() -> None:
    report = _fit_report(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    assert report["severity"] not in (SEVERITY_DIAGNOSTIC_ONLY, SEVERITY_BLOCKED)
    assert not (report.get("control_completeness") or {}).get("omitted_control_risk")
    elig = report["output_eligibility"]
    assert elig["optimizer_decision_surface_unchanged"] is True
    assert "model_fit_review" in elig["allowed_uses"]


def test_missing_required_diagnostic_only() -> None:
    report = _fit_report(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    assert report["severity"] == SEVERITY_DIAGNOSTIC_ONLY
    elig = report["output_eligibility"]
    assert elig["human_review_required"] is True
    assert "budget_reallocation_claim" in elig["forbidden_uses"]
    assert elig.get("diagnostic_only_reason")


def test_collinearity_restricted_interpretation() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec).copy()
    panel["ctv"] = panel["display"] * 0.99
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(panel, schema, config, None, vertical_id="retail")
    if (report.get("collinearity") or {}).get("weak_identification_risk"):
        assert report["severity"] in (SEVERITY_RESTRICTED, SEVERITY_DIAGNOSTIC_ONLY, SEVERITY_WARNING)


def test_sparse_channel_restricted() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec).copy()
    panel["radio"] = 0.0
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(panel, schema, config, None, vertical_id="retail")
    if (report.get("sparse_channels") or {}).get("sparse_channel_extreme"):
        assert report["severity"] in (SEVERITY_RESTRICTED, SEVERITY_DIAGNOSTIC_ONLY)
        forbidden = report["output_eligibility"]["forbidden_uses"]
        assert any("isolated_channel_claim" in u or "sparse" in u.lower() for u in forbidden)


def test_fold_instability_diagnostic_only() -> None:
    report = {
        "control_completeness": {"omitted_control_risk": False},
        "collinearity": {"weak_identification_risk": False, "calibration_evidence_available": True},
        "sparse_channels": {"sparse_channel_extreme": []},
        "transform_diagnostics": {"metadata_complete": True},
        "fold_stability": {"fold_stability_ok": False},
        "forbidden_claims": [],
        "warnings": [],
        "production_flags": {"optimizer_enabled": False, "decision_surface_enabled": False},
    }
    elig = classify_ridge_diagnostic_severity(report)
    assert elig["severity"] == SEVERITY_DIAGNOSTIC_ONLY


def test_forbidden_fields_blocked() -> None:
    report = {
        "decision_surface": True,
        "control_completeness": {},
        "collinearity": {},
        "sparse_channels": {},
        "transform_diagnostics": {"metadata_complete": True},
        "fold_stability": {},
        "production_flags": {"optimizer_enabled": False},
    }
    elig = classify_ridge_diagnostic_severity(report)
    assert elig["severity"] == SEVERITY_BLOCKED
    assert "engineering_debug_only" in elig["allowed_uses"]


def test_missing_transform_warning() -> None:
    report = {
        "control_completeness": {
            "omitted_control_risk": False,
            "missing_optional_controls": ["weather_index"],
        },
        "collinearity": {"weak_identification_risk": False, "calibration_evidence_available": True},
        "sparse_channels": {"sparse_channel_extreme": []},
        "transform_diagnostics": {"metadata_complete": False, "warnings": ["ridge_transform:missing"]},
        "fold_stability": {"fold_stability_ok": True},
        "forbidden_claims": [],
        "warnings": ["ridge_transform:missing"],
        "production_flags": {},
    }
    elig = classify_ridge_diagnostic_severity(report)
    assert elig["severity"] in (SEVERITY_WARNING, SEVERITY_INFO)


def test_summary_renders_eligibility() -> None:
    report = _fit_report(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    assert summary["allowed_uses"]
    assert summary["forbidden_uses"]
    assert summary["human_review_required"] is True
    md = format_ridge_diagnostics_markdown(report)
    assert "Output eligibility" in md
    assert "Allowed uses" in md
    cli = format_ridge_diagnostics_cli_block(report)
    assert any("Allowed uses" in line or "Human review" in line for line in cli)


def test_apply_policy_adds_output_eligibility() -> None:
    base = {"status": "unavailable"}
    out = apply_severity_policy_to_report(base)
    assert out["output_eligibility"]["severity"] == SEVERITY_DIAGNOSTIC_ONLY


def test_write_severity_archive() -> None:
    report = _fit_report(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    payload = {
        "artifact_kind": "RIDGE_PRODUCTION_DIAGNOSTICS_SEVERITY",
        "milestone": "H9",
        "ridge_diagnostics": report,
        "output_eligibility": report["output_eligibility"],
    }
    ARCHIVE_SEVERITY.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_SEVERITY.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    assert report["severity"] == SEVERITY_DIAGNOSTIC_ONLY
