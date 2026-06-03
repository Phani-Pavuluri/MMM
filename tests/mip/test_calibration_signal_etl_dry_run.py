"""MIP-C4 — CalibrationSignal ETL dry-run tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mmm.diagnostics.calibration_signal_etl import (
    ETL_ARTIFACT_TYPE,
    adapt_export_to_signals,
    load_geox_cls_export,
    prove_c2_ingest,
    run_dry_run_etl,
    validate_signal_artifact,
)
from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
from mmm.diagnostics.calibration_signal_ingestion import parse_calibration_signals_payload
from mmm.diagnostics.ridge_diagnostics import (
    FORBIDDEN_OUTPUT_FIELDS,
    attach_ridge_diagnostics_to_extension_report,
)
from mmm.data.loader import DatasetBuilder
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

FIXTURE_EXPORT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "mip_calibration_signal_adapters"
    / "mixed_batch.json"
)
DRY_RUN_ARCHIVE = Path(
    "docs/05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json"
)
TRAIN_PROOF_ARCHIVE = Path(
    "docs/05_validation/archives/MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json"
)


@pytest.fixture
def mixed_export_bundle() -> dict:
    data = json.loads(FIXTURE_EXPORT.read_text(encoding="utf-8"))
    return data["export_bundle"]


def test_load_geox_cls_export_unwraps_fixture() -> None:
    export = load_geox_cls_export(FIXTURE_EXPORT)
    assert "records" in export


def test_mixed_export_becomes_c2_compatible_signals(mixed_export_bundle: dict) -> None:
    signals, errors, lineage = adapt_export_to_signals(mixed_export_bundle)
    assert not errors
    assert len(signals) == 2
    assert lineage["geox_count"] == 1
    assert lineage["cls_count"] == 1
    valid, parse_errs = parse_calibration_signals_payload({"signals": signals})
    assert not parse_errs
    assert len(valid) == 2


def test_run_dry_run_etl_writes_artifact(tmp_path: Path, mixed_export_bundle: dict) -> None:
    inp = tmp_path / "export.json"
    inp.write_text(json.dumps(mixed_export_bundle), encoding="utf-8")
    out = tmp_path / "signals.json"
    artifact = run_dry_run_etl(inp, out)
    assert out.is_file()
    assert artifact["artifact_type"] == ETL_ARTIFACT_TYPE
    lineage = artifact["etl_lineage"]
    assert lineage["signals_written"] == 2
    assert "geox" in lineage["source_systems"]
    assert lineage["inconclusive_count"] >= 1


def test_disposition_counts_in_lineage(tmp_path: Path) -> None:
    full = json.loads(FIXTURE_EXPORT.read_text(encoding="utf-8"))
    bundle_path = tmp_path / "full_fixture.json"
    bundle_path.write_text(json.dumps(full), encoding="utf-8")
    out = tmp_path / "out.json"
    artifact = run_dry_run_etl(bundle_path, out)
    lineage = artifact["etl_lineage"]
    assert lineage["records_seen"] >= 2
    assert lineage["context_only"] is True
    assert lineage["optimizer_unchanged"] is True


def test_output_ingests_through_c2_path(tmp_path: Path) -> None:
    out = tmp_path / "artifact.json"
    run_dry_run_etl(FIXTURE_EXPORT, out)
    artifact = json.loads(out.read_text(encoding="utf-8"))
    report = prove_c2_ingest(artifact)
    assert report["evidence_attachment_lineage"]["attempted"] is True
    assert report.get("calibration_evidence_context")


def test_production_flags_false(tmp_path: Path) -> None:
    out = tmp_path / "artifact.json"
    artifact = run_dry_run_etl(FIXTURE_EXPORT, out)
    flags = artifact["production_flags"]
    assert flags["approved_for_prod"] is False
    assert flags["decisioning_allowed"] is False


def test_validate_rejects_bad_artifact() -> None:
    ok, errs = validate_signal_artifact({"artifact_type": "wrong", "signals": []})
    assert not ok
    assert errs


def test_no_optimizer_decision_surface_fields(tmp_path: Path) -> None:
    out = tmp_path / "artifact.json"
    artifact = run_dry_run_etl(FIXTURE_EXPORT, out)
    for sig in artifact["signals"]:
        for key in FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
            assert key not in sig


def test_cli_module_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "cli_out.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmm.diagnostics.calibration_signal_etl",
            "--input",
            str(FIXTURE_EXPORT),
            "--output",
            str(out),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == ETL_ARTIFACT_TYPE


def test_train_boundary_consumes_dry_run_artifact(tmp_path: Path) -> None:
    """Prove C2 train/extension path accepts ETL artifact (fast attach, same as train)."""
    signals_path = tmp_path / "etl_signals.json"
    run_dry_run_etl(FIXTURE_EXPORT, signals_path)
    if not Path("examples/sample_panel.csv").is_file():
        pytest.skip("sample panel missing")
    from mmm.config.load import load_config

    config = load_config("examples/minimal_train.yaml")
    config.ridge_bo.n_trials = 2
    builder = DatasetBuilder(config.data)
    panel = builder.build()
    schema = builder.schema()
    trainer = RidgeBOMMMTrainer(config, schema)
    fit_out = trainer.fit(panel)
    fit_out["_ridge_trainer"] = trainer
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        calibration_signals_path=str(signals_path),
    )
    report = ext["ridge_production_diagnostics_report"]
    assert report["evidence_attachment_lineage"]["attempted"] is True
    assert report["evidence_attachment_lineage"]["source_type"] == "file"
    assert report.get("calibration_evidence_context")


def test_mip_c4_dry_run_archive_exists() -> None:
    if not DRY_RUN_ARCHIVE.is_file():
        pytest.skip("run ETL CLI to materialize MIP-C4 dry-run archive")
    artifact = json.loads(DRY_RUN_ARCHIVE.read_text(encoding="utf-8"))
    ok, _ = validate_signal_artifact(artifact)
    assert ok


def test_mip_c4_train_proof_archive_exists() -> None:
    if not TRAIN_PROOF_ARCHIVE.is_file():
        pytest.skip("train proof archive not materialized")
    payload = json.loads(TRAIN_PROOF_ARCHIVE.read_text(encoding="utf-8"))
    assert payload["calibration_evidence_context_present"] is True
