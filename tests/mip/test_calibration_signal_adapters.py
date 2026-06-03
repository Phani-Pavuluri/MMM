"""MIP-C3 — GeoX/CLS → CalibrationSignal adapter tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.diagnostics.calibration_signal_adapters import (
    ADAPTER_VERSION,
    adapt_mixed_batch_export,
    cls_record_to_calibration_signal,
    geox_record_to_calibration_signal,
    normalize_calibration_signal_batch,
    validate_adapter_output,
)
from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
from mmm.diagnostics.calibration_signal_ingestion import ingest_calibration_signals_into_report
from mmm.diagnostics.ridge_diagnostics import FORBIDDEN_OUTPUT_FIELDS

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_calibration_signal_adapters"
ARCHIVE_PATH = Path("docs/05_validation/archives/MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json")


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_geox_valid_converts_to_c2_compatible_signal() -> None:
    fx = _load_fixture("geox_valid_signal.json")
    sig = geox_record_to_calibration_signal(fx["source_record"])
    ok, errs = validate_adapter_output(sig)
    assert ok, errs
    exp = fx["expected"]
    assert sig["source_system"] == exp["source_system"]
    assert sig["eligibility_status"] == exp["eligibility_status"]
    assert sig["freshness_status"] == exp["freshness_status"]
    assert sig["standard_error"] == pytest.approx(0.021)
    assert sig["measurement_instrument_id"] == exp["measurement_instrument_id"]
    assert sig["geo_scope"]["kind"] == "dma"


def test_cls_valid_converts_to_c2_compatible_signal() -> None:
    fx = _load_fixture("cls_valid_signal.json")
    sig = cls_record_to_calibration_signal(fx["source_record"])
    ok, errs = validate_adapter_output(sig)
    assert ok, errs
    assert sig["source_system"] == "cls"
    assert sig["study_id"] == "CLS-STUDY-2201"
    assert sig["eligibility_status"] == "eligible"


def test_geox_missing_uncertainty_marked_blocked() -> None:
    fx = _load_fixture("geox_missing_uncertainty.json")
    sig = geox_record_to_calibration_signal(fx["source_record"])
    assert sig["standard_error"] is None
    assert sig["eligibility_status"] == "blocked"
    notes = sig["adapter_metadata"]["adapter_notes"]
    assert any("missing_uncertainty" in n for n in notes)


def test_cls_stale_marked_stale() -> None:
    fx = _load_fixture("cls_stale_signal.json")
    sig = cls_record_to_calibration_signal(fx["source_record"])
    assert sig["freshness_status"] == "stale"


def test_geox_estimand_mismatch_preserved_for_trust_report() -> None:
    fx = _load_fixture("geox_estimand_mismatch.json")
    sig = geox_record_to_calibration_signal(fx["source_record"])
    assert sig["estimand_id"] == "brand_lift"
    notes = sig["adapter_metadata"]["adapter_notes"]
    assert any("estimand_mismatch" in n for n in notes)
    assert "TrustReport" in sig["adapter_metadata"]["trust_report_boundary"]


def test_cls_inconclusive_eligibility() -> None:
    fx = _load_fixture("cls_inconclusive_signal.json")
    sig = cls_record_to_calibration_signal(fx["source_record"])
    assert sig["eligibility_status"] == "inconclusive"


def test_mixed_batch_normalizes_safely() -> None:
    fx = _load_fixture("mixed_batch.json")
    signals, errors, lineage = adapt_mixed_batch_export(fx["export_bundle"])
    assert not errors
    assert len(signals) == fx["expected"]["signal_count"]
    assert lineage["geox_count"] == 1
    assert lineage["cls_count"] == 1
    assert {s["source_system"] for s in signals} == {"geox", "cls"}


def test_adapter_output_ingests_via_c2_path() -> None:
    fx = _load_fixture("geox_valid_signal.json")
    sig = geox_record_to_calibration_signal(fx["source_record"])
    stub = {
        "coefficient_stability": {"media_coef_by_channel": {"search": 0.05}},
        "forbidden_claims": [],
        "world_metadata": {"estimand_id": "incremental_sales"},
    }
    report = ingest_calibration_signals_into_report(stub, signals=[sig])
    assert report["evidence_attachment_lineage"]["attempted"] is True
    assert report.get("calibration_evidence_context")


def test_conflict_geox_ingests_with_forbidden_claims() -> None:
    geox = geox_record_to_calibration_signal(
        {
            "geox_experiment_id": "EXP-C",
            "channel": "tv",
            "incremental_lift": 0.2,
            "incremental_lift_se": 0.05,
            "estimand_type": "incremental_sales",
            "evidence_as_of": "2026-05-01",
        }
    )
    stub = {"coefficient_stability": {"media_coef_by_channel": {"tv": -0.1}}, "forbidden_claims": []}
    report = ingest_calibration_signals_into_report(stub, signals=[geox])
    assert "mmm_direction_validated_by_external_evidence" in report["forbidden_claims"]


def test_no_optimizer_decision_surface_recommendation_fields() -> None:
    fx = _load_fixture("cls_valid_signal.json")
    sig = cls_record_to_calibration_signal(fx["source_record"])
    for key in FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
        assert key not in sig


def test_normalize_batch_geox() -> None:
    fx = _load_fixture("geox_valid_signal.json")
    signals, errors = normalize_calibration_signal_batch([fx["source_record"]], "geox")
    assert not errors
    assert len(signals) == 1


def test_mip_c3_archive_exists() -> None:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("MIP-C3 archive not materialized")
    payload = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    assert payload["adapter_version"] == ADAPTER_VERSION
    assert len(payload["signals"]) >= 2
    for sig in payload["signals"]:
        ok, _ = validate_adapter_output(sig)
        assert ok
