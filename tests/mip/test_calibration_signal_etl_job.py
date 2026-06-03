"""MIP-C5 — Scheduled CalibrationSignal ETL wrapper tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
from mmm.diagnostics.calibration_signal_etl import prove_c2_ingest, validate_signal_artifact
from mmm.diagnostics.calibration_signal_etl_job import (
    MANIFEST_ARTIFACT_TYPE,
    discover_export_files,
    run_etl_for_file,
    run_scheduled_etl_job,
    summarize_job_run,
    validate_job_outputs,
    write_job_manifest,
)
from mmm.diagnostics.ridge_diagnostics import FORBIDDEN_OUTPUT_FIELDS

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_calibration_signal_adapters"
C5_OUTPUT_DIR = Path("docs/05_validation/archives/mip_c5_etl_outputs")
TRAIN_PROOF_ARCHIVE = Path(
    "docs/05_validation/archives/MIP_C5_TRAIN_WITH_SCHEDULED_ETL_SIGNALS_20260601.json"
)


def test_discover_matching_files() -> None:
    found = discover_export_files(FIXTURE_DIR, "mixed_batch.json")
    assert len(found) == 1
    assert found[0].name == "mixed_batch.json"


def test_discover_ignores_nonmatching(tmp_path: Path) -> None:
    (tmp_path / "match.json").write_text("{}", encoding="utf-8")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    found = discover_export_files(tmp_path, "match.json")
    assert len(found) == 1
    assert discover_export_files(tmp_path, "missing.json") == []


def test_run_etl_for_file_writes_c2_output(tmp_path: Path) -> None:
    inp = FIXTURE_DIR / "mixed_batch.json"
    row = run_etl_for_file(inp, tmp_path, "TEST_RUN")
    assert row["success"]
    assert Path(row["output_path"]).is_file()
    artifact = json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))
    ok, _ = validate_signal_artifact(artifact)
    assert ok


def test_scheduled_job_writes_manifest_and_summary(tmp_path: Path) -> None:
    manifest = run_scheduled_etl_job(
        input_dir=FIXTURE_DIR,
        output_dir=tmp_path,
        run_id="JOB_TEST",
        pattern="mixed_batch.json",
    )
    assert manifest["artifact_type"] == MANIFEST_ARTIFACT_TYPE
    assert (tmp_path / "JOB_TEST_manifest.json").is_file()
    assert (tmp_path / "JOB_TEST_summary.md").is_file()
    assert manifest["files_processed"] == 1
    assert manifest["live_api_used"] is False
    assert manifest["production_scheduler"] is False


def test_continue_on_error_processes_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    good = tmp_path / "mixed_batch.json"
    good.write_text((FIXTURE_DIR / "mixed_batch.json").read_text(), encoding="utf-8")
    manifest = run_scheduled_etl_job(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        run_id="BATCH",
        pattern="*.json",
        continue_on_error=True,
    )
    assert manifest["files_seen"] == 2
    assert manifest["files_processed"] >= 1
    assert manifest["files_failed"] >= 1


def test_fail_fast_stops_batch(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    (tmp_path / "good.json").write_text(
        (FIXTURE_DIR / "mixed_batch.json").read_text(), encoding="utf-8"
    )
    manifest = run_scheduled_etl_job(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        run_id="FF",
        pattern="*.json",
        continue_on_error=False,
    )
    assert manifest["files_processed"] + manifest["files_failed"] <= manifest["files_seen"]


def test_validate_job_outputs(tmp_path: Path) -> None:
    manifest = run_scheduled_etl_job(
        input_dir=FIXTURE_DIR,
        output_dir=tmp_path,
        run_id="VAL",
        pattern="mixed_batch.json",
    )
    ok, errs = validate_job_outputs(manifest, tmp_path)
    assert ok, errs


def test_output_ingests_through_c2(tmp_path: Path) -> None:
    row = run_etl_for_file(FIXTURE_DIR / "mixed_batch.json", tmp_path, "C2")
    artifact = json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))
    report = prove_c2_ingest(artifact)
    assert report.get("calibration_evidence_context")


def test_no_decision_fields_in_signals(tmp_path: Path) -> None:
    row = run_etl_for_file(FIXTURE_DIR / "mixed_batch.json", tmp_path, "ND")
    artifact = json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))
    for sig in artifact["signals"]:
        for key in FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
            assert key not in sig


def test_cli_job_module(tmp_path: Path) -> None:
    out = tmp_path / "cli_out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmm.diagnostics.calibration_signal_etl_job",
            "--input-dir",
            str(FIXTURE_DIR),
            "--pattern",
            "mixed_batch.json",
            "--output-dir",
            str(out),
            "--run-id",
            "CLI_JOB",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "CLI_JOB_manifest.json").is_file()
    assert list(out.glob("CLI_JOB_*_signals.json"))


def test_mip_c5_materialized_outputs_exist() -> None:
    if not C5_OUTPUT_DIR.is_dir():
        pytest.skip("run MIP-C5 CLI to materialize outputs")
    manifest_path = C5_OUTPUT_DIR / "MIP_C5_DRY_RUN_20260601_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok, _ = validate_job_outputs(manifest, C5_OUTPUT_DIR)
    assert ok


def test_train_proof_archive_exists() -> None:
    if not TRAIN_PROOF_ARCHIVE.is_file():
        pytest.skip("train proof archive not materialized")
    payload = json.loads(TRAIN_PROOF_ARCHIVE.read_text(encoding="utf-8"))
    assert payload["calibration_evidence_context_present"] is True


def test_summarize_job_run_renders_markdown() -> None:
    manifest = {
        "run_id": "X",
        "started_at": "t0",
        "completed_at": "t1",
        "input_dir": "/in",
        "pattern": "*.json",
        "output_dir": "/out",
        "files_seen": 1,
        "files_processed": 1,
        "files_failed": 0,
        "records_seen": 2,
        "signals_written": 2,
        "blocked_count": 0,
        "inconclusive_count": 1,
        "stale_count": 0,
        "context_only": True,
        "live_api_used": False,
        "production_scheduler": False,
        "errors": [],
        "output_artifacts": [],
    }
    md = summarize_job_run(manifest)
    assert "Live API used" in md
