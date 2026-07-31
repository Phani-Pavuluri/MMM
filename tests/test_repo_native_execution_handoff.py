"""Reusable fail-closed checks for MMM's V2 execution handoff files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
EXECUTION = ROOT / "docs" / "execution"
STATE_PATH = EXECUTION / "EXECUTION_STATE.json"
TASK_PATH = EXECUTION / "ACTIVE_TASK.md"
REPORT_PATH = EXECUTION / "LATEST_COMPLETION_REPORT.md"
CONTEXT_PATH = EXECUTION / "REPOSITORY_CONTEXT_INDEX.md"
MIP_V2_PIN = "38f88467f55d5bc4cc64e5a58b0f08f1639a40d0"
ALLOWED_STATUSES = {
    "idle",
    "proposed",
    "authorized",
    "in_progress",
    "blocked",
    "ready_for_review",
    "changes_requested",
    "merged",
    "superseded",
}
SHA = re.compile(r"^[0-9a-f]{40}$")


def _state() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(STATE_PATH.read_text(encoding="utf-8")))


def test_v2_execution_handoff_files_are_consistent_and_fail_closed() -> None:
    state = _state()
    task = TASK_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")
    context = CONTEXT_PATH.read_text(encoding="utf-8")

    assert all(path.is_file() for path in (AGENTS, STATE_PATH, TASK_PATH, REPORT_PATH, CONTEXT_PATH))
    assert state["schema_version"] == "mmm_repo_execution_state_v2"
    assert state["status"] in ALLOWED_STATUSES
    assert state["task_id"] in task and state["task_id"] in report
    assert state["canonical_mip_standard_commit"] == MIP_V2_PIN
    assert MIP_V2_PIN in task and MIP_V2_PIN in report and MIP_V2_PIN in context
    assert "approved_for_merge" not in state
    assert isinstance(state["task_execution_authorized"], bool)
    assert isinstance(state["merge_authorized"], bool)
    assert isinstance(state["capability_authorizations_changed"], bool)
    assert state["capability_authorizations_changed"] is False
    assert SHA.fullmatch(str(state["base_sha"]))
    assert SHA.fullmatch(str(state["authorization_head_sha"]))

    bootstrap = (
        "git status --porcelain=v1 --untracked-files=all",
        "`.codex/` and `docs/tasks/`",
        "git fetch --prune origin",
        "git fetch --unshallow origin",
        "git switch main",
        "git pull --ff-only origin main",
        "git rev-parse main",
        "git rev-parse origin/main",
    )
    assert all(command in agents for command in bootstrap)
    assert agents.index("git status --porcelain=v1") < agents.index("git fetch --prune origin")
    assert agents.index("git fetch --prune origin") < agents.index("git switch main")
    assert agents.index("git switch main") < agents.index("EXECUTION_STATE.json")
    assert "Fresh Chat Bootstrap" in context
    assert "Only then read MMM `EXECUTION_STATE.json`" in context

    merge_rules = (
        "exact remote feature-branch head SHA",
        "pre-merge approval-metadata commit",
        "git merge --ff-only",
        "exactly one\npost-merge closure commit",
    )
    assert all(rule in agents for rule in merge_rules)

    if state["status"] == "authorized":
        assert state["task_execution_authorized"] is True
        assert state["merge_authorized"] is False
        assert state["implementation_commit_sha"] is None
        assert state["reviewed_head_sha"] is None
        assert state["approval_commit_sha"] is None
    if state["status"] == "ready_for_review":
        assert state["task_execution_authorized"] is True
        assert state["merge_authorized"] is False
        assert SHA.fullmatch(str(state["implementation_commit_sha"]))
        assert state["reviewed_head_sha"] is None
        assert state["approval_commit_sha"] is None
    if state["status"] == "merged":
        assert state["task_execution_authorized"] is False
        assert state["merge_authorized"] is False
        assert SHA.fullmatch(str(state["reviewed_head_sha"]))
        assert state["approval_commit_sha"] is None
