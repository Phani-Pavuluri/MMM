"""Governance evidence checks for post-compatibility R11/R12 task selection."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/05_validation/MMM_MIP_HANDOFF_V1_POST_COMPATIBILITY_GAP_SELECTION_AUDIT_001_REPORT.md"
SUMMARY = ROOT / "docs/05_validation/archives/MMM_MIP_HANDOFF_V1_POST_COMPATIBILITY_GAP_SELECTION_AUDIT_001_summary.json"
ROADMAP = ROOT / "docs/05_validation/platform_roadmap.md"
INVENTORY = ROOT / "docs/DOCUMENTATION_INVENTORY.md"


def test_post_compatibility_gap_selection_audit_is_complete_and_conservative() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert REPORT.is_file()
    assert summary["audit_id"] == "MMM_MIP_HANDOFF_V1_POST_COMPATIBILITY_GAP_SELECTION_AUDIT_001"
    assert summary["audited_base_commit"] == "c96848e"
    assert summary["prerequisite_commits"] == ["30d4543", "8d73e0c", "7f615fb", "949d7a6", "fdc69f9", "c96848e"]
    assert summary["established_completed_requirements"] == ["R6", "R7", "R9", "R10", "R13", "R15"]
    assert summary["r11_assessment"]["status"] == "PARTIAL"
    assert summary["r12_assessment"]["status"] == "PARTIAL"
    models = summary["dependency_models_considered"]
    assert set(models) == {"A_R12_before_R11", "B_R11_before_R12", "C_narrow_shared_prerequisite_first", "D_parallel_independent_work"}
    assert summary["selected_dependency_model"] == "C_narrow_shared_prerequisite_first"
    assert summary["narrower_prerequisite_findings"]["exists"] is True
    assert summary["selected_next_task"]["task_id"] == "MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001"
    assert len(summary["deferred_alternatives"]) == 3
    assert summary["r16_status"] == "BLOCKED"
    assert summary["interface_freeze_status"] == "unauthorized"
    assert all(value is False for value in summary["authorization_flags"].values())
    assert summary["deterministic_verdict"] in REPORT.read_text(encoding="utf-8")


def test_governance_documents_preserve_the_selected_boundary() -> None:
    text = "\n".join((ROADMAP.read_text(encoding="utf-8"), INVENTORY.read_text(encoding="utf-8")))
    assert "MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001" in text
    assert "R16 remains blocked" in text
    assert "interface freeze remains unauthorized" in text
