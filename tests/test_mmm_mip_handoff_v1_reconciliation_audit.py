"""Governance contract checks for the MMM-to-MIP handoff reconciliation audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/05_validation/MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001.md"
SUMMARY = ROOT / "docs/05_validation/archives/MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001_summary.json"
ROADMAP = ROOT / "docs/05_validation/platform_roadmap.md"
REQUIREMENTS = {
    "R1_CANONICAL_EXPORT_BUNDLE", "R2_RUN_IDENTITY_LINEAGE", "R3_MEASUREMENT_SCOPE",
    "R4_MODEL_PROMOTION_STATUS", "R5_RESULTS_UNCERTAINTY", "R6_CALIBRATION_LINEAGE",
    "R7_DIAGNOSTICS", "R8_TECHNICAL_CLAIM_GOVERNANCE", "R9_RUN_MANIFEST",
    "R10_TYPED_FAILURE_PACKET", "R11_FULL_PANEL_DELTA_MU_SIMULATION",
    "R12_RESPONSE_SURFACE_EVIDENCE", "R13_GOLDEN_ENGINE_FIXTURES",
    "R14_INTERNAL_MODEL_ISOLATION", "R15_VERSIONING_COMPATIBILITY", "R16_MIP_CONSUMER_READINESS",
}
STATUSES = {"complete", "partial", "missing", "misowned", "duplicated", "blocked"}
CLASSIFICATIONS = {
    "KEEP_IN_MMM", "KEEP_BUT_RENAME_AS_PRODUCER_ELIGIBILITY", "SPLIT_MMM_AND_MIP_RESPONSIBILITIES",
    "MIP_OWNED_POLICY_REMOVE_FROM_MMM", "DUPLICATES_EXISTING_MMM_CONTRACT", "INSUFFICIENT_EVIDENCE",
}


def test_reconciliation_audit_contract() -> None:
    assert AUDIT.is_file()
    assert SUMMARY.is_file()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["artifact_id"] == "MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001"
    assert summary["artifact_id"] in AUDIT.read_text(encoding="utf-8")
    requirements = summary["requirements"]
    assert len(requirements) == len(REQUIREMENTS)
    assert {row["id"] for row in requirements} == REQUIREMENTS
    assert {row["status"] for row in requirements} <= STATUSES
    assert all(row["status"] in STATUSES for row in requirements)
    assert {row["classification"] for row in summary["a803da2_classifications"]} <= CLASSIFICATIONS
    assert summary["next_task"] == "MMM_MIP_HANDOFF_V1_PRODUCER_GOLDEN_FIXTURES_001"
    assert summary["producer_boundary_cleanup"] == {
        "completed": True,
        "interface_freeze_recommended": False,
        "r10_typed_failure_packet": "missing",
        "r16_mip_consumer_readiness": "blocked",
    }
    assert summary["typed_failure_packet_followup"] == {
        "r10_typed_failure_packet": "implemented",
        "r16_mip_consumer_readiness": "blocked",
        "interface_freeze_recommended": False,
    }
    assert summary["typed_run_manifest_followup"] == {
        "r9_typed_run_manifest": "implemented",
        "r10_typed_failure_packet": "implemented",
        "r16_mip_consumer_readiness": "blocked",
        "interface_freeze_recommended": False,
    }
    assert summary["calibration_treatment_lineage_followup"] == {
        "r6_calibration_treatment_lineage": "implemented",
        "r9_typed_run_manifest": "implemented",
        "r10_typed_failure_packet": "implemented",
        "r16_mip_consumer_readiness": "blocked",
        "interface_freeze_recommended": False,
    }
    assert summary["non_authorizations"]
    assert "MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001" in ROADMAP.read_text(encoding="utf-8")
