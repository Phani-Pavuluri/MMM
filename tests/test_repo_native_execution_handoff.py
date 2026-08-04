"""Semantic checks for MMM's Git-durable execution protocol."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
TASK = (ROOT / "docs/execution/ACTIVE_TASK.md").read_text(encoding="utf-8")
CONTEXT = (ROOT / "docs/execution/REPOSITORY_CONTEXT_INDEX.md").read_text(encoding="utf-8")
LEAN = (ROOT / "docs/program/LEAN_REPOSITORY_DELIVERY_STANDARD.md").read_text(encoding="utf-8")
STANDARD = (ROOT / "docs/execution/TASK_EXECUTION_STANDARD.md").read_text(encoding="utf-8")
STATE = json.loads((ROOT / "docs/execution/EXECUTION_STATE.json").read_text(encoding="utf-8"))


def test_lean_definition_ready_delivery_is_adopted() -> None:
    for phrase in (
        "one independently mergeable outcome",
        "Split when a",
        "public contract, migration",
        "surface-appropriate resolved decisions",
        "inputs/outputs/invariants/failures",
        "compatibility or migration policy",
        "named deterministic evidence",
        "deferred successor",
        "unresolved execution-blocking design questions: none",
    ):
        assert phrase in AGENTS or phrase in LEAN
    assert "Why this task cannot be split further" in TASK
    assert "Failure semantics" in TASK


def test_invocation_only_and_terminal_outcomes_are_adopted() -> None:
    assert "Synchronize from Git and execute the active task." in AGENTS
    assert "Approved exact remote head: <FULL_SHA>" in AGENTS
    assert "Prompts cannot repair" in AGENTS
    assert "successful orientation is non-terminal" in AGENTS
    assert "Git-durable `ready_for_review` or `blocked`" in AGENTS
    assert "continue through implementation, validation, publication, and push" in AGENTS
    assert "Stop externally only when no safe authorized branch exists" in AGENTS
    launcher = "Work in <local repository path>."
    assert launcher in AGENTS
    assert "Approved exact remote head: <FULL_SHA>" in AGENTS
    assert "chat output is diagnostic context only" in AGENTS


def test_resumed_branch_and_exact_tree_receipt_are_adopted() -> None:
    for phrase in (
        "Main authorizes",
        "exact remote branch",
        "freeze the task-owned tree",
        "exact-commit-tree",
        "implementation parent",
        "Docker `make validate` count/disposition",
        "worktree state",
        "evidence source",
        "Any post-receipt change",
    ):
        assert phrase in AGENTS or phrase in STANDARD
    assert "Fail-closed conflicts" in TASK
    assert "authorization-head ancestry" in AGENTS
    assert "completion report is evidence only" in STANDARD or "is evidence only" in AGENTS


def test_risk_tiers_preserve_mmm_full_validation() -> None:
    assert all(f"| {tier} |" in LEAN for tier in ("1", "2", "3"))
    assert "never waive MMM's" in AGENTS
    assert "make validate" in AGENTS and STATE["full_suite_validation_required"] is True
    for phrase in (
        "Tier 3",
        "analytical/public/package surface",
        "required category that cannot run is `blocked`",
        "Do not start duplicate validation containers",
    ):
        assert phrase in AGENTS or phrase in LEAN


def test_live_overlay_coordination_is_adopted() -> None:
    for phrase in (
        "pinned MIP coordination protocol",
        "live `origin/main`",
        "exact remote feature-branch execution files",
        "stale shared snapshot",
        "duplicate ownership",
        "overlapping implementation",
        "producer completion",
        "consumer verification",
        "dependency/blocker transitions",
        "validation debt",
        "authority impact",
    ):
        assert phrase in AGENTS or phrase in STANDARD
    assert STATE["repository"] == "Phani-Pavuluri/MMM"
    assert STATE["task_id"] in TASK
    assert STATE["coordination_capability_owner"].replace("_", " ") in TASK.lower()
    assert STATE["affected_repositories"] == ["Phani-Pavuluri/MMM"]


def test_exact_head_merge_closure_and_pr19_history_are_preserved() -> None:
    for phrase in (
        "exact remote feature-branch SHA",
        "pr_creation_authorized",
        "git merge --ff-only",
        "pre-merge approval commit",
        "exactly one closure commit",
        "Historical PR #19",
    ):
        assert phrase in AGENTS
    history = STATE["historical_nonconforming_merge"]
    assert STATE["pr_creation_authorized"] is False
    assert history["external_merge_pr_number"] == 19
    assert history["external_branch_head_sha"] == "ea16ab7e7b1089f5de479eeffb236fad2767edf1"
    assert history["external_merge_commit_sha"] == "ad55fef6799a8bd717108781ad44fc88fa116df7"
    assert history["external_merge_was_authorized"] is False
    assert history["conforming_exact_head_approval_record_exists"] is False


def test_repository_context_index_is_navigation_only() -> None:
    assert "CROSS_REPOSITORY_COORDINATION_PROTOCOL.md" in CONTEXT
    assert "current feature branch" not in CONTEXT
    assert "active_task_status" not in CONTEXT


def test_authority_hierarchy_and_current_task_consistency_are_adopted() -> None:
    assert "Synchronized `main/docs/execution/EXECUTION_STATE.json` owns" in AGENTS
    assert "The verified remote feature branch state owns" in AGENTS
    assert "`ACTIVE_TASK.md` owns" in AGENTS
    assert "`LATEST_COMPLETION_REPORT.md` is evidence only" in AGENTS
    assert STATE["task_id"] == "MMM_EXECUTION_AUTHORITY_AND_OPERATIONAL_LAUNCHER_ALIGNMENT_001"
    assert STATE["task_id"] in TASK
    assert STATE["task_id"] in (ROOT / "docs/execution/LATEST_COMPLETION_REPORT.md").read_text()
    assert STATE["feature_branch"] in TASK


def test_fail_closed_conflicts_and_handoff_semantics_are_adopted() -> None:
    for phrase in (
        "never chooses whichever file appears newer",
        "Git-durable blocked state",
        "exact mismatch",
        "attempted evidence",
        "validation categories",
        "live resolution condition",
        "Chat, cached prompts",
        "Progress reports are non-terminal",
        "only repository, feature branch, and exact remote head SHA",
    ):
        assert phrase in AGENTS or phrase in STANDARD
    assert STATE["task_execution_authorized"] is True
    assert STATE["merge_authorized"] is False
    assert STATE["pr_creation_authorized"] is False
    assert STATE["blockers"] == []


def test_full_validation_and_closure_controls_remain_preserved() -> None:
    for phrase in (
        "make validate",
        "exact-tree receipt",
        "exact-head",
        "git merge --ff-only",
        "one closure commit",
        "branch cleanup",
        "Historical PR #19",
    ):
        assert phrase in AGENTS or phrase in STANDARD or phrase in TASK
    assert STATE["full_suite_validation_required"] is True
    assert STATE["mmm_analytical_authority_changed"] is False
    assert STATE["sibling_authority_changed"] is False
    assert STATE["capability_authorizations_changed"] is False


def test_launcher_does_not_duplicate_task_meaning() -> None:
    launcher_start = AGENTS.index("Work in <local repository path>.")
    launcher_end = AGENTS.index("```", launcher_start + 3)
    launcher = AGENTS[launcher_start:launcher_end]
    for forbidden in ("MMM_EXECUTION_AUTHORITY", "ACTIVE_TASK.md", "tests/", "objective", "KPI"):
        assert forbidden not in launcher
    assert "Synchronize main from Git" in launcher
    assert "remote feature branch" in launcher
    assert "Progress reports are non-terminal" in launcher
