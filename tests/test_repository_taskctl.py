"""Focused semantic tests for the canonical MMM taskctl controller."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from mmm.execution import taskctl

ROOT = Path(__file__).resolve().parents[1]


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs/execution").mkdir(parents=True)
    v2 = json.loads((ROOT / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    for key in (
        "rejected_implementation_commit_sha",
        "correction_cycles_completed",
        "correction_cycles_remaining",
        "live_resolution_condition",
        "local_feature_branch_cleanup",
        "remote_feature_branch_cleanup",
    ):
        v2.pop(key, None)
    v2.update(
        {
            "schema_version": taskctl.SCHEMA_V2,
            "status": "authorized",
            "review_decision": "authorized",
            "feature_branch_created": False,
            "task_execution_authorized": True,
            "correction_execution_authorized": False,
            "implementation_commit_sha": None,
            "reviewed_head_sha": None,
            "rejected_review_head_sha": None,
            "approval_commit_sha": None,
            "blockers": [],
        }
    )
    for relative in ("docs/execution/ACTIVE_TASK.md", "docs/execution/LATEST_COMPLETION_REPORT.md"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        content = re.sub(
            r"(?ms)^<!-- BEGIN MMM TASKCTL EXECUTION VIEW -->.*?^<!-- END MMM TASKCTL EXECUTION VIEW -->\n?",
            "",
            content,
        )
        if relative.endswith("ACTIVE_TASK.md"):
            content = content.replace(
                "# Active Task\n\n", "# Active Task\n\n**Lifecycle:** `authorized` after migration\n", 1
            )
        (root / relative).write_text(content, encoding="utf-8")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "taskctl@example.invalid")
    _git(root, "config", "user.name", "taskctl")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "fixture base")
    base = _git(root, "rev-parse", "HEAD")

    def replace_sha(value: object) -> object:
        if isinstance(value, dict):
            return {key: replace_sha(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace_sha(child) for child in value]
        if isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value):
            return base
        return value

    v2 = replace_sha(v2)
    assert isinstance(v2, dict)
    (root / taskctl.STATE_PATH).write_text(json.dumps(v2, indent=2) + "\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture state")
    branch = _git(root, "branch", "--show-current")
    assert branch == "main"
    _git(root, "remote", "add", "origin", "git@github.com:Phani-Pavuluri/MMM.git")
    # A local origin/main is sufficient to exercise synchronized-main checks.
    _git(root, "update-ref", "refs/remotes/origin/main", _git(root, "rev-parse", "HEAD"))
    return root


def test_exact_v2_migration_and_idempotent_sync(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    with pytest.raises(taskctl.TaskControlError, match="E_MIGRATION_REQUIRED"):
        taskctl.check(root)
    taskctl.sync(root)
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert state["schema_version"] == taskctl.SCHEMA_V3
    assert state["correction_cycles_completed"] == 0
    assert state["correction_cycles_remaining"] == state["max_correction_cycles"]
    first = {path: (root / path).read_bytes() for path in (taskctl.STATE_PATH, taskctl.TASK_PATH, taskctl.REPORT_PATH)}
    taskctl.sync(root)
    assert first == {path: (root / path).read_bytes() for path in first}


@pytest.mark.parametrize(
    ("field", "value"),
    (("repository", "other/repo"), ("status", "unknown"), ("schema_version", "v9"), ("max_correction_cycles", -1)),
)
def test_malformed_state_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    root = _fixture(tmp_path)
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    state[field] = value
    (root / taskctl.STATE_PATH).write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError):
        taskctl.sync(root)


def test_marker_divergence_is_fail_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    path = root / taskctl.TASK_PATH
    path.write_text(path.read_text(encoding="utf-8").replace("Status", "Status changed", 1), encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError, match="E_VIEW_DIVERGENCE"):
        taskctl.check(root)


def test_transition_requires_branch_and_evidence(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    with pytest.raises(taskctl.TaskControlError, match="E_TRANSITION_BRANCH"):
        taskctl.transition(
            root,
            type(
                "Args",
                (),
                {
                    "to": "in_progress",
                    "implementation_sha": None,
                    "rejected_review_head_sha": None,
                    "rejected_implementation_commit_sha": None,
                    "reviewed_head_sha": None,
                    "blocker": None,
                    "live_resolution_condition": None,
                    "complete_correction": False,
                    "local_feature_branch_cleanup": None,
                    "remote_feature_branch_cleanup": None,
                },
            )(),
        )
