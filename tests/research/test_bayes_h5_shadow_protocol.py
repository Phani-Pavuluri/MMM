"""Bayes-H5e real-panel shadow-run protocol (schema only — no execution)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h5_shadow_protocol import (
    FORBIDDEN_OUTPUT_FIELDS,
    REQUIRED_LINEAGE_FIELDS,
    H5ShadowProtocolError,
    build_dry_run_shadow_run_record,
    build_shadow_run_schema_document,
    load_shadow_run_schema,
    validate_shadow_run_record,
    validate_shadow_run_schema_document,
    write_shadow_run_schema_artifact,
)


def test_schema_artifact_valid_on_disk() -> None:
    write_shadow_run_schema_artifact()
    schema = load_shadow_run_schema()
    validate_shadow_run_schema_document(schema)
    assert schema["production_flags"]["hard_gate"] is False
    assert schema["production_flags"]["approved_for_prod"] is False


def test_schema_contains_required_lineage_fields() -> None:
    schema = build_shadow_run_schema_document()
    lineage = set(schema["required_lineage_fields"])
    assert lineage >= REQUIRED_LINEAGE_FIELDS


def test_production_flags_false() -> None:
    schema = build_shadow_run_schema_document()
    for key in ("hard_gate", "production_promotion", "approved_for_prod", "prod_decisioning_allowed"):
        assert schema["production_flags"][key] is False
    record = schema["example_shadow_run_record"]
    assert record["production_flags"]["hard_gate"] is False


def test_excluded_production_fields_listed() -> None:
    schema = build_shadow_run_schema_document()
    excluded = set(schema["excluded_fields"])
    assert "decision_surface" in excluded
    assert "optimizer_ready_curves" in excluded
    assert "budget_recommendation" in excluded
    assert excluded <= FORBIDDEN_OUTPUT_FIELDS


def test_forbidden_fields_fail_validation() -> None:
    record = build_dry_run_shadow_run_record()
    record["decision_surface"] = {"blocked": True}
    with pytest.raises(H5ShadowProtocolError, match="forbidden"):
        validate_shadow_run_record(record)


def test_ridge_comparison_diagnostic_only() -> None:
    record = build_dry_run_shadow_run_record()
    assert record["ridge_comparison"]["decision_grade"] is False
    assert record["ridge_comparison"]["used_for_optimizer"] is False
    record["ridge_comparison"]["used_for_optimizer"] = True
    with pytest.raises(H5ShadowProtocolError, match="used_for_optimizer"):
        validate_shadow_run_record(record)


def test_trust_report_candidates_research_only() -> None:
    record = build_dry_run_shadow_run_record()
    trust = record["trust_report_candidate_diagnostics"]
    assert trust.get("production_trust_report") is None
    assert "h5:production:block" in trust["warning_codes"]


def test_missing_dataset_snapshot_id_fails_closed() -> None:
    record = build_dry_run_shadow_run_record()
    record["dataset_snapshot_id"] = ""
    with pytest.raises(H5ShadowProtocolError, match="dataset_snapshot_id"):
        validate_shadow_run_record(record)


def test_missing_transform_config_fails_closed() -> None:
    record = build_dry_run_shadow_run_record()
    record["transform_config"] = {}
    with pytest.raises(H5ShadowProtocolError, match="transform_config"):
        validate_shadow_run_record(record)


def test_dry_run_record_has_no_optimizer_fields() -> None:
    record = build_dry_run_shadow_run_record()
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation", "recommendation"):
        assert forbidden not in record
    validate_shadow_run_record(record)


def test_write_schema_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    doc = write_shadow_run_schema_artifact(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    validate_shadow_run_schema_document(loaded)
    assert loaded["schema_id"] == doc["schema_id"]
