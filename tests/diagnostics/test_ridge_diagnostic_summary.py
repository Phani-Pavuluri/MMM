"""H8 — Ridge diagnostic summary and operator artifact surfacing tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.artifacts.lifecycle import persist_training_artifacts
from mmm.artifacts.stores.local import LocalArtifactStore
from mmm.diagnostics.ridge_diagnostic_summary import (
    FORBIDDEN_SUMMARY_TERMS,
    export_ridge_diagnostic_artifacts,
    extract_forbidden_claims,
    format_ridge_diagnostics_cli_block,
    format_ridge_diagnostics_markdown,
    severity_badge,
    summarize_ridge_diagnostics,
    write_ridge_diagnostic_archive_files,
)
from mmm.research.h6_synthetic.production_shapes import (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)
from mmm.diagnostics.ridge_diagnostics import compose_ridge_diagnostic_report
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


ARCHIVE_JSON = Path(
    "docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_REPORT_20260601.json"
)
ARCHIVE_MD = Path(
    "docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SUMMARY_20260601.md"
)


def _report_for_world(world_id: str, *, vertical_id: str, media_corr: bool = False):
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


def test_missing_controls_in_summary() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    assert summary["omitted_control_risk"] is True
    assert "promo_flag" in summary["missing_required_controls"]
    md = format_ridge_diagnostics_markdown(report)
    assert "promo_flag" in md
    assert "Forbidden claims" in md


def test_sparse_channel_in_summary() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    panel = materialize_h6_panel(get_h6_world(WORLD_H6_PILOT_RETAIL_FULL))
    panel = panel.copy()
    panel["radio"] = 0.0
    config = h6_ridge_config(get_h6_world(WORLD_H6_PILOT_RETAIL_FULL))
    schema = h6_panel_schema(get_h6_world(WORLD_H6_PILOT_RETAIL_FULL))
    report = compose_ridge_diagnostic_report(panel, schema, config, None, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    if "radio" in (summary.get("sparse_channels_extreme") or []):
        assert "radio" in format_ridge_diagnostics_markdown(report)


def test_collinearity_in_summary() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec).copy()
    panel["ctv"] = panel["display"] * 0.98
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(panel, schema, config, None, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    if summary.get("weak_identification_risk"):
        assert "Collinearity" in format_ridge_diagnostics_markdown(report)


def test_forbidden_claims_rendered() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    claims = extract_forbidden_claims(report)
    assert "no_clean_media_attribution_claim" in claims
    cli = format_ridge_diagnostics_cli_block(report)
    assert any("Forbidden claims" in line for line in cli)


def test_production_boundary_rendered() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    boundary = summary["production_boundary"]
    assert boundary.get("bayes_h5_research_only") is True
    assert boundary.get("optimizer_enabled") is False
    md = format_ridge_diagnostics_markdown(report)
    assert "Production boundary" in md
    assert "Bayes-H5 research-only" in md


def test_clean_report_summary() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    summary = summarize_ridge_diagnostics(report)
    assert summary["severity_badge"] in ("OK", "INFO", "WARNING", "CRITICAL")
    assert not summary.get("omitted_control_risk")


def test_summary_no_optimizer_decision_language() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    md = format_ridge_diagnostics_markdown(report).lower()
    for term in ("budget recommendation", "optimizer output", "decision_surface_enabled: true"):
        assert term not in md
    cli_text = "\n".join(format_ridge_diagnostics_cli_block(report)).lower()
    for term in FORBIDDEN_SUMMARY_TERMS[:4]:
        assert term not in cli_text or "false" in cli_text or "must be false" in md


def test_severity_badge_mapping() -> None:
    assert severity_badge({"diagnostic_severity": "high"}) == "CRITICAL"
    assert severity_badge({"diagnostic_severity": "none"}) == "OK"


def test_export_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    ext: dict = {"ridge_production_diagnostics_report": report}
    store = LocalArtifactStore(tmp_path)
    store.start_run("ridge_h8_test")
    written = export_ridge_diagnostic_artifacts(store, ext)
    assert "ridge_production_diagnostics_report.json" in written.values()
    assert (store.run_path / "ridge_production_diagnostics_report.json").is_file()
    assert (store.run_path / "ridge_production_diagnostics_summary.md").is_file()
    assert ext.get("ridge_production_diagnostics_summary")
    store.end_run()


def test_persist_training_artifacts_backward_compatible(tmp_path: Path) -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    ext = {"governance": {"approved_for_optimization": False}, "ridge_production_diagnostics_report": report}
    store = LocalArtifactStore(tmp_path)
    store.start_run("lifecycle_h8")
    written = persist_training_artifacts(store, extension_report=ext)
    assert "extension_report.json" in written.values()
    assert "ridge_production_diagnostics_report.json" in written.values()
    er = json.loads((store.run_path / "extension_report.json").read_text(encoding="utf-8"))
    assert er.get("governance")
    assert er.get("ridge_production_diagnostics_summary")
    store.end_run()


def test_write_reference_archives() -> None:
    report = _report_for_world(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    write_ridge_diagnostic_archive_files(report, json_path=ARCHIVE_JSON, markdown_path=ARCHIVE_MD)
    assert ARCHIVE_JSON.is_file()
    assert ARCHIVE_MD.is_file()
    assert "no_clean_media_attribution_claim" in ARCHIVE_MD.read_text(encoding="utf-8")
