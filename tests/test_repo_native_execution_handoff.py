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
        "compatibility or migration policy",
        "deferred successor",
        "unresolved execution-blocking design questions: none",
    ):
        assert phrase in AGENTS or phrase in LEAN
    assert "Why this task cannot be split further" in TASK


def test_invocation_only_and_terminal_outcomes_are_adopted() -> None:
    assert "Synchronize from Git and execute the active task." in AGENTS
    assert "Approved exact remote head: <SHA>." in AGENTS
    assert "Prompts cannot repair" in AGENTS
    assert "successful orientation is non-terminal" in AGENTS
    assert "Git-durable `ready_for_review` or `blocked`" in AGENTS


def test_resumed_branch_and_exact_tree_receipt_are_adopted() -> None:
    for phrase in (
        "Main authorizes",
        "exact remote branch",
        "freeze the task-owned tree",
        "exact-commit-tree",
        "Any post-receipt change",
    ):
        assert phrase in AGENTS or phrase in STANDARD


def test_risk_tiers_preserve_mmm_full_validation() -> None:
    assert all(f"| {tier} |" in LEAN for tier in ("1", "2", "3"))
    assert "never waive MMM's" in AGENTS
    assert "make validate" in AGENTS and STATE["full_suite_validation_required"] is True


def test_live_overlay_coordination_is_adopted() -> None:
    for phrase in (
        "pinned MIP coordination protocol",
        "live `origin/main`",
        "stale shared snapshot",
        "producer completion",
        "consumer verification",
    ):
        assert phrase in AGENTS or phrase in STANDARD
    assert STATE["coordination_workstream_id"] == "WS-MMM-PROTOCOL-ADOPTION-001"


def test_exact_head_merge_closure_and_pr19_history_are_preserved() -> None:
    for phrase in (
        "exact remote feature-branch SHA",
        "git merge --ff-only",
        "pre-merge approval commit",
        "exactly one closure commit",
        "Historical PR #19",
    ):
        assert phrase in AGENTS
    assert STATE["historical_nonconforming_merge"]["external_merge_was_authorized"] is False


def test_repository_context_index_is_navigation_only() -> None:
    assert "CROSS_REPOSITORY_COORDINATION_PROTOCOL.md" in CONTEXT
    assert "current feature branch" not in CONTEXT
    assert "active_task_status" not in CONTEXT
