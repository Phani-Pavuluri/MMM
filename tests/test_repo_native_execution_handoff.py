"""Reusable consistency checks for MMM's stable execution handoff files."""

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
ALLOWED_STATUSES = {
    "idle",
    "proposed",
    "authorized",
    "in_progress",
    "blocked",
    "ready_for_review",
    "changes_requested",
    "approved_for_merge",
    "merged",
    "superseded",
}
SHA = re.compile(r"^[0-9a-f]{40}$")


def _state() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(STATE_PATH.read_text(encoding="utf-8")))


def test_stable_execution_handoff_files_are_consistent_and_fail_closed() -> None:
    state = _state()
    task = TASK_PATH.read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")
    context = CONTEXT_PATH.read_text(encoding="utf-8")

    assert all(path.is_file() for path in (AGENTS, STATE_PATH, TASK_PATH, REPORT_PATH, CONTEXT_PATH))
    assert state["schema_version"] == "mmm_repo_execution_state_v1"
    assert state["status"] in ALLOWED_STATUSES
    assert state["task_id"] in task and state["task_id"] in report
    assert f"**Status:** {state['status']}" in task
    assert isinstance(state["task_execution_authorized"], bool)
    assert isinstance(state["merge_authorized"], bool)
    assert isinstance(state["capability_authorizations_changed"], bool)
    assert agents.index("EXECUTION_STATE.json") < agents.index("ACTIVE_TASK.md") < agents.index(
        "REPOSITORY_CONTEXT_INDEX.md"
    )
    assert "Fresh Chat Bootstrap" in context
    assert "connected GitHub as the source of truth" in context
    assert "5eebba6750a3754e4026397d6762c601b1d6a708" in context

    if state["status"] == "ready_for_review":
        assert state["task_execution_authorized"] is True
        assert state["merge_authorized"] is False
        assert SHA.fullmatch(str(state["implementation_commit_sha"]))
        assert state["reviewed_head_sha"] is None
        assert state["approval_commit_sha"] is None
    if state["status"] == "approved_for_merge":
        assert state["task_execution_authorized"] is True
        assert state["merge_authorized"] is True
        assert SHA.fullmatch(str(state["reviewed_head_sha"]))
        assert SHA.fullmatch(str(state["approval_commit_sha"]))
    if state["task_id"] == "MMM_REPO_NATIVE_EXECUTION_HANDOFF_ADOPTION_001":
        assert state["capability_authorizations_changed"] is False
