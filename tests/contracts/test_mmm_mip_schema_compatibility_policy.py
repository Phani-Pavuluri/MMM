"""Evidence checks for the MMM-owned schema compatibility policy registry."""

from __future__ import annotations

import json
from pathlib import Path

from mmm.contracts.calibration_treatment import MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION
from mmm.contracts.diagnostics_limitations import MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION
from mmm.contracts.mip_export import MMMExportBundle, SCHEMA_VERSION
from mmm.contracts.mip_failure import MMM_FAILURE_SCHEMA_VERSION, MMMExportOutcome, MMMFailurePacket
from mmm.contracts.run_manifest import (
    MMM_RUN_MANIFEST_SCHEMA_VERSION,
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunManifest,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "docs/05_validation/archives/MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001_registry.json"
GOLDEN_INDEX_PATH = ROOT / "tests/fixtures/mip_export/golden_v1/index.json"

EXPECTED_CONTRACTS = {
    "mmm_export_bundle",
    "mmm_failure_packet",
    "mmm_export_outcome",
    "mmm_run_manifest",
    "mmm_export_manifest_outcome",
    "mmm_artifact_reference",
    "mmm_calibration_treatment_lineage",
    "mmm_diagnostics_limitations",
}
EXPECTED_VERSIONS = {
    "mmm_export_bundle": SCHEMA_VERSION,
    "mmm_failure_packet": MMM_FAILURE_SCHEMA_VERSION,
    "mmm_run_manifest": MMM_RUN_MANIFEST_SCHEMA_VERSION,
    "mmm_calibration_treatment_lineage": MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION,
    "mmm_diagnostics_limitations": MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION,
}
EXPECTED_RULES = {
    "optional_field_addition",
    "required_field_addition",
    "field_removal",
    "field_rename",
    "field_type_or_nullability_change",
    "semantic_change",
    "enum_expansion",
    "unknown_field",
    "missing_field",
    "fixture_scenario_addition",
    "fixture_scenario_removal",
    "version_reuse",
}


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _contracts(registry: dict[str, object]) -> dict[str, dict[str, object]]:
    return {record["contract_id"]: record for record in registry["public_contracts"]}  # type: ignore[index]


def test_registry_is_deterministic_complete_and_has_no_invented_versions() -> None:
    registry = _registry()
    assert registry["policy_id"] == "MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001"
    assert registry["policy_schema_version"] == "mmm_mip_schema_compatibility_policy_v1"
    assert registry["audited_base_commit"] == "fdc69f9"
    contracts = _contracts(registry)
    assert set(contracts) == EXPECTED_CONTRACTS
    assert len(contracts) == len(registry["public_contracts"])
    for contract_id, version in EXPECTED_VERSIONS.items():
        record = contracts[contract_id]
        assert record["current_schema_version"] == version
        assert record["supported_versions"] == [version]
    for contract_id in {"mmm_export_outcome", "mmm_export_manifest_outcome", "mmm_artifact_reference"}:
        assert contracts[contract_id]["current_schema_version"] is None
        assert contracts[contract_id]["supported_versions"] == []


def test_registry_records_current_per_contract_parser_behavior_without_runtime_change() -> None:
    registry = _registry()
    contracts = _contracts(registry)
    assert MMMExportBundle.model_config["extra"] == "allow"
    assert contracts["mmm_export_bundle"]["unknown_field_policy"]["top_level"] == "CURRENT_RUNTIME_ALLOW_AND_PRESERVE"  # type: ignore[index]
    for model, contract_id in (
        (MMMFailurePacket, "mmm_failure_packet"),
        (MMMExportOutcome, "mmm_export_outcome"),
        (MMMRunManifest, "mmm_run_manifest"),
        (MMMExportManifestOutcome, "mmm_export_manifest_outcome"),
        (MMMArtifactReference, "mmm_artifact_reference"),
    ):
        assert model.model_config["extra"] == "forbid"
        assert contracts[contract_id]["unknown_field_policy"]["top_level"] == "CURRENT_RUNTIME_REJECT"  # type: ignore[index]
    for record in contracts.values():
        missing = record["missing_field_policy"]
        assert {"required", "optional", "newly_introduced_field", "schema_version"} <= set(missing)
        unknown = record["unknown_field_policy"]
        assert {"top_level", "nested", "optional_metadata", "required_semantic"} <= set(unknown)
        assert record["enum_expansion_policy"]


def test_policy_classifies_required_changes_and_deprecation_lifecycle() -> None:
    registry = _registry()
    rules = {rule["rule_id"]: rule for rule in registry["compatibility_rules"]}  # type: ignore[index]
    assert set(rules) == EXPECTED_RULES
    assert rules["optional_field_addition"]["classification"] == "CONDITIONALLY_COMPATIBLE"
    for rule_id in {"required_field_addition", "field_removal", "field_rename", "field_type_or_nullability_change", "semantic_change", "fixture_scenario_removal"}:
        assert rules[rule_id]["classification"] == "BREAKING"
    assert rules["version_reuse"]["classification"] == "PROHIBITED_UNTIL_AUTHORIZED"
    lifecycle = registry["deprecation_lifecycle"]
    assert lifecycle["states"] == ["ACTIVE", "DEPRECATED", "UNSUPPORTED", "REMOVED"]
    assert lifecycle["current_versions_state"] == "ACTIVE"
    assert lifecycle["removal_authorized_by_this_policy"] is False
    assert "separate_removal_authorization" in lifecycle["removal_requires"]


def test_golden_fixture_set_support_and_safety_status_are_evidence_backed() -> None:
    registry = _registry()
    fixture = registry["fixture_sets"][0]  # type: ignore[index]
    index = json.loads(GOLDEN_INDEX_PATH.read_text(encoding="utf-8"))
    assert fixture["fixture_set_id"] == index["fixture_set_id"]
    assert fixture["fixture_set_version"] == index["schema_version"] == "mmm_producer_golden_fixture_set_v1"
    assert fixture["required_scenarios"] == sorted(item["scenario_id"] for item in index["scenarios"])
    assert fixture["represented_contract_versions"] == {
        "mmm_run_manifest": MMM_RUN_MANIFEST_SCHEMA_VERSION,
        "mmm_failure_packet": MMM_FAILURE_SCHEMA_VERSION,
        "mmm_calibration_treatment_lineage": MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION,
        "mmm_diagnostics_limitations": MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION,
    }
    assert fixture["removal_policy"] == "BREAKING_FOR_FIXTURE_CONSUMERS_AND_PROHIBITED_UNTIL_AUTHORIZED"
    assert "interface freeze" in fixture["consumer_regression_expectation"]


def test_fail_closed_and_authorization_boundaries_remain_explicit() -> None:
    registry = _registry()
    fail_closed = set(registry["fail_closed_rules"])
    assert {
        "unsupported_required_schema_version",
        "missing_schema_version_evidence",
        "unknown_terminal_run_status",
        "unknown_failure_code_with_remediation_effect",
        "unknown_promotion_status",
        "unknown_claim_disposition",
        "unresolved_required_artifact_reference",
        "inconsistent_cross_artifact_run_id",
        "contradictory_success_and_failure_terminal_state",
        "research_only_artifact_presented_as_production_supported",
    } <= fail_closed
    assert all(value is False for value in registry["authorization_flags"].values())
    assert registry["interface_freeze_status"] == "unauthorized"
    assert registry["consumer_readiness_status"] == "blocked"
    statuses = registry["status_record"]
    assert statuses["R6_CALIBRATION_LINEAGE"] == "IMPLEMENTED"
    assert statuses["R7_TYPED_DIAGNOSTICS_LIMITATIONS"] == "IMPLEMENTED"
    assert statuses["R9_TYPED_RUN_MANIFEST"] == "IMPLEMENTED"
    assert statuses["R10_TYPED_FAILURE_PACKET"] == "IMPLEMENTED"
    assert statuses["R13_PRODUCER_GOLDEN_FIXTURES"] == "IMPLEMENTED"
    assert statuses["R15_SCHEMA_COMPATIBILITY_POLICY"] == "IMPLEMENTED_WITH_POLICY_EVIDENCE"
    assert statuses["R11_FULL_PANEL_DELTA_MU_SIMULATION"] == "PARTIAL"
    assert statuses["R12_RESPONSE_SURFACE_EVIDENCE"] == "PARTIAL"
    assert statuses["R16_MIP_CONSUMER_READINESS"] == "BLOCKED"
