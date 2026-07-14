"""MIP-C2 — CalibrationSignal ingestion at Ridge train/extension boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.config.schema import Framework
from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
from mmm.diagnostics.calibration_signal_ingestion import (
    ingest_calibration_signals_into_report,
    load_calibration_signals_from_path,
)
from mmm.diagnostics.ridge_diagnostic_summary import (
    format_ridge_diagnostics_cli_block,
    format_ridge_diagnostics_markdown,
    summarize_ridge_diagnostics,
)
from mmm.diagnostics.ridge_diagnostics import (
    FORBIDDEN_OUTPUT_FIELDS,
    attach_ridge_diagnostics_to_extension_report,
)
from mmm.diagnostics.ridge_real_bundle_hardening import BENCHMARK_BUNDLE_SPEC, run_real_bundle_ridge_diagnostics
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_calibration_signal_ingestion"
ARCHIVE_PATH = Path(
    "docs/05_validation/archives/"
    "MIP_C2_RIDGE_DIAGNOSTICS_WITH_CALIBRATION_SIGNAL_CONTEXT_20260601.json"
)


def _benchmark_fit(n_trials: int = 2) -> tuple:
    result = run_real_bundle_ridge_diagnostics({**BENCHMARK_BUNDLE_SPEC, "n_trials": n_trials})
    from mmm.diagnostics.ridge_real_bundle_hardening import load_bundle_panel

    panel, schema, config = load_bundle_panel({**BENCHMARK_BUNDLE_SPEC, "n_trials": n_trials})
    trainer = RidgeBOMMMTrainer(config, schema)
    fit_out = trainer.fit(panel)
    fit_out["_ridge_trainer"] = trainer
    return panel, schema, config, fit_out, result


def test_no_signal_path_preserves_h11_absent_lineage() -> None:
    if not Path(BENCHMARK_BUNDLE_SPEC["panel_path"]).is_file():
        pytest.skip("benchmark panel missing")
    panel, schema, config, fit_out, _ = _benchmark_fit()
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=fit_out.get("_ridge_trainer"),
    )
    lineage = ext["ridge_production_diagnostics_report"]["evidence_attachment_lineage"]
    assert lineage["attempted"] is False
    assert lineage["source_type"] == "none"
    assert lineage["calibration_evidence_context_present"] is False


def test_aligned_signal_file_attaches_context() -> None:
    if not Path(BENCHMARK_BUNDLE_SPEC["panel_path"]).is_file():
        pytest.skip("benchmark panel missing")
    path = FIXTURE_DIR / "aligned_signals.json"
    panel, schema, config, fit_out, _ = _benchmark_fit()
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=fit_out.get("_ridge_trainer"),
        calibration_signals_path=str(path),
    )
    report = ext["ridge_production_diagnostics_report"]
    lineage = report["evidence_attachment_lineage"]
    assert lineage["attempted"] is True
    assert lineage["source_type"] == "file"
    assert lineage["attached_count"] >= 1
    assert report.get("calibration_evidence_context")
    assert lineage["context_only"] is True


def test_conflict_signal_attaches_forbidden_claims_on_stub() -> None:
    stub = {
        "report_version": "mmm_ridge_production_diagnostics_v1",
        "coefficient_stability": {"media_coef_by_channel": {"tv": -0.1, "search": 0.05}},
        "collinearity": {},
        "forbidden_claims": [],
        "warnings": [],
    }
    path = FIXTURE_DIR / "conflict_signals.json"
    report = ingest_calibration_signals_into_report(stub, signals_path=str(path))
    assert report["calibration_evidence_context"]
    assert "mmm_direction_validated_by_external_evidence" in report["forbidden_claims"]
    assert report["evidence_attachment_lineage"]["attached_count"] == 1


def test_malformed_signal_file_records_errors_without_changing_fit() -> None:
    if not Path(BENCHMARK_BUNDLE_SPEC["panel_path"]).is_file():
        pytest.skip("benchmark panel missing")
    path = FIXTURE_DIR / "malformed_signals.json"
    panel, schema, config, fit_out, _ = _benchmark_fit()
    art_before = fit_out["artifacts"]
    params_before = dict(art_before.best_params)
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=fit_out.get("_ridge_trainer"),
        calibration_signals_path=str(path),
    )
    assert fit_out["artifacts"].best_params == params_before
    lineage = ext["ridge_production_diagnostics_report"]["evidence_attachment_lineage"]
    assert lineage["attempted"] is True
    assert lineage["rejected_count"] >= 2
    assert lineage["attachment_errors"]
    assert lineage["attached_count"] == 1


def test_empty_signal_file_attempted_zero_attached() -> None:
    path = FIXTURE_DIR / "empty_signals.json"
    valid, errors, meta = load_calibration_signals_from_path(path)
    assert valid == []
    assert errors == []
    stub = {"coefficient_stability": {"media_coef_by_channel": {"tv": 0.1}}, "forbidden_claims": []}
    report = ingest_calibration_signals_into_report(stub, signals_path=str(path))
    lineage = report["evidence_attachment_lineage"]
    assert lineage["attempted"] is True
    assert lineage["attached_count"] == 0
    assert not report.get("calibration_evidence_context")


def test_extension_report_embeds_summary_with_signals() -> None:
    if not Path(BENCHMARK_BUNDLE_SPEC["panel_path"]).is_file():
        pytest.skip("benchmark panel missing")
    path = FIXTURE_DIR / "aligned_signals.json"
    panel, schema, config, fit_out, _ = _benchmark_fit()
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=fit_out.get("_ridge_trainer"),
        calibration_signals_path=str(path),
    )
    report = ext["ridge_production_diagnostics_report"]
    ext["ridge_production_diagnostics_summary"] = summarize_ridge_diagnostics(report)
    assert ext["ridge_production_diagnostics_summary"]["calibration_evidence_context_present"] is True


def test_markdown_cli_surface_evidence_context() -> None:
    stub = {
        "coefficient_stability": {"media_coef_by_channel": {"search": 0.05}},
        "forbidden_claims": [],
        "warnings": [],
    }
    path = FIXTURE_DIR / "aligned_signals.json"
    report = ingest_calibration_signals_into_report(stub, signals_path=str(path))
    md = format_ridge_diagnostics_markdown(report)
    cli = format_ridge_diagnostics_cli_block(report)
    assert "Calibration evidence context" in md
    assert any("Calibration context" in line for line in cli)


def test_no_optimizer_decision_surface_recommendation_fields() -> None:
    if not Path(BENCHMARK_BUNDLE_SPEC["panel_path"]).is_file():
        pytest.skip("benchmark panel missing")
    path = FIXTURE_DIR / "aligned_signals.json"
    panel, schema, config, fit_out, _ = _benchmark_fit()
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        calibration_signals_path=str(path),
    )
    report = ext["ridge_production_diagnostics_report"]
    flags = report["production_flags"]
    assert flags["optimizer_enabled"] is False
    assert flags["decision_surface_enabled"] is False
    assert flags["recommendations_enabled"] is False
    lineage = report["evidence_attachment_lineage"]
    assert lineage["optimizer_unchanged"] is True
    assert lineage["decision_surface_unchanged"] is True
    assert lineage["recommendations_unchanged"] is True
    for key in FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
        assert key not in report


def test_mip_c2_archive_fixture_exists() -> None:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("MIP-C2 archive not materialized — run write_mip_c2_archive")
    payload = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    assert payload["milestone"] == "MIP-C2"
    lineage = payload["ridge_production_diagnostics_report"]["evidence_attachment_lineage"]
    assert lineage["attempted"] is True
    assert lineage["calibration_evidence_context_present"] is True


def test_non_ridge_framework_skips_ridge_diagnostics_attachment() -> None:
    from mmm.config.schema import MMMConfig, ModelForm, PoolingMode, RunEnvironment

    config = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        run_environment=RunEnvironment.RESEARCH,
        data={"channel_columns": ["tv"], "target_column": "y", "geo_column": "g", "week_column": "w"},
    )
    ext = attach_ridge_diagnostics_to_extension_report(
        {"existing": True},
        __import__("pandas").DataFrame(),
        __import__("mmm.data.schema", fromlist=["PanelSchema"]).PanelSchema(
            geo_column="g",
            week_column="w",
            target_column="y",
            channel_columns=("tv",),
            control_columns=(),
        ),
        config,
        {},
    )
    assert ext.get("existing") is True
    assert "ridge_production_diagnostics_report" not in ext
