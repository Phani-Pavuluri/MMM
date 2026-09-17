"""Focused semantic tests for the canonical MMM taskctl controller."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mmm.execution import taskctl

ROOT = Path(__file__).resolve().parents[1]


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
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
    _git(root, "update-ref", "refs/remotes/origin/main", _git(root, "rev-parse", "HEAD"))
    return root


def _feature_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    _git(root, "switch", "-c", state["feature_branch"])
    return root, _git(root, "rev-parse", "HEAD")


def _args(to: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "to": to,
        "implementation_sha": None,
        "rejected_review_head_sha": None,
        "rejected_implementation_commit_sha": None,
        "reviewed_head_sha": None,
        "blocker": None,
        "clear_blockers": False,
        "live_resolution_condition": None,
        "complete_correction": False,
        "local_feature_branch_cleanup": None,
        "remote_feature_branch_cleanup": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _to_ready_then_changes_requested(root: Path, sha: str) -> None:
    taskctl.transition(root, _args("in_progress"))
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha))
    taskctl.transition(
        root,
        _args(
            "changes_requested",
            implementation_sha=sha,
            rejected_review_head_sha=sha,
            rejected_implementation_commit_sha=sha,
        ),
    )


def test_dynamic_task_and_branch_identity_are_state_driven(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    state["task_id"] = "MMM_FUTURE_TASK_002"
    state["feature_branch"] = "feat/future-task-002"
    (root / taskctl.STATE_PATH).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    taskctl.sync(root)
    taskctl.check(root)
    state["feature_branch"] = "main"
    (root / taskctl.STATE_PATH).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError, match="E_BRANCH"):
        taskctl.check(root)


def test_correction_direct_and_resumed_paths_preserve_rejection_and_consume_once(tmp_path: Path) -> None:
    root, sha = _feature_fixture(tmp_path / "direct")
    _to_ready_then_changes_requested(root, sha)
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha, complete_correction=True))
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert state["correction_cycles_completed"] == 1
    assert state["correction_cycles_remaining"] == state["max_correction_cycles"] - 1
    assert state["rejected_review_head_sha"] == sha

    root, sha = _feature_fixture(tmp_path / "resumed")
    _to_ready_then_changes_requested(root, sha)
    taskctl.transition(root, _args("in_progress"))
    mid = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert mid["implementation_commit_sha"] is None
    assert mid["rejected_implementation_commit_sha"] == sha
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha, complete_correction=True))
    final = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert final["correction_cycles_completed"] == 1
    assert final["correction_cycles_remaining"] == final["max_correction_cycles"] - 1


def test_blocker_evidence_requires_explicit_unambiguous_clearing(tmp_path: Path) -> None:
    root, _ = _feature_fixture(tmp_path)
    taskctl.transition(root, _args("blocked", blocker=["needs evidence"], live_resolution_condition="supply evidence"))
    with pytest.raises(taskctl.TaskControlError, match="E_BLOCKER_CLEAR_REQUIRED"):
        taskctl.transition(root, _args("in_progress"))
    with pytest.raises(taskctl.TaskControlError, match="E_EVIDENCE"):
        taskctl.transition(root, _args("in_progress", clear_blockers=True, blocker=["ambiguous"]))
    taskctl.transition(root, _args("in_progress", clear_blockers=True))
    assert json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))["blockers"] == []


def test_candidate_validation_happens_before_any_write(tmp_path: Path) -> None:
    root, _ = _feature_fixture(tmp_path)
    before = {path: (root / path).read_bytes() for path in (taskctl.STATE_PATH, taskctl.TASK_PATH, taskctl.REPORT_PATH)}
    with pytest.raises(taskctl.TaskControlError):
        taskctl.transition(root, _args("blocked", blocker=[]))
    assert before == {path: (root / path).read_bytes() for path in before}


def test_cli_failure_has_stable_code_and_exit_status(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    env = {"PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "mmm.execution.taskctl", "--root", str(root), "check"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "E_MIGRATION_REQUIRED" in result.stderr


def test_git_origin_branch_missing_commit_and_ancestry_fail_closed(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    _git(root, "remote", "set-url", "origin", "git@github.com:other/repo.git")
    with pytest.raises(taskctl.TaskControlError, match="E_ORIGIN"):
        taskctl.check(root)
    _git(root, "remote", "set-url", "origin", "git@github.com:Phani-Pavuluri/MMM.git")
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    state["base_sha"] = "f" * 40
    (root / taskctl.STATE_PATH).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError, match="E_GIT"):
        taskctl.sync(root)


def test_marker_corruption_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    path = root / taskctl.TASK_PATH
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace(taskctl.BEGIN, f"{taskctl.BEGIN}\n{taskctl.BEGIN}"), encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError, match="E_MARKERS"):
        taskctl.check(root)


def test_protected_authority_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    taskctl.sync(root)
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    state["merge_authorized"] = True
    (root / taskctl.STATE_PATH).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(taskctl.TaskControlError, match="E_PROTECTED_AUTHORITY"):
        taskctl.check(root)


def test_correction_exhaustion_and_merged_evidence(tmp_path: Path) -> None:
    root, sha = _feature_fixture(tmp_path)
    _to_ready_then_changes_requested(root, sha)
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha, complete_correction=True))
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert state["correction_cycles_remaining"] == 1
    taskctl.transition(
        root,
        _args(
            "changes_requested",
            implementation_sha=sha,
            rejected_review_head_sha=sha,
            rejected_implementation_commit_sha=sha,
        ),
    )
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha, complete_correction=True))
    with pytest.raises(taskctl.TaskControlError, match="E_CORRECTION"):
        taskctl.transition(
            root,
            _args(
                "changes_requested",
                implementation_sha=sha,
                rejected_review_head_sha=sha,
                rejected_implementation_commit_sha=sha,
            ),
        )

    root, sha = _feature_fixture(tmp_path / "merged")
    taskctl.transition(root, _args("in_progress"))
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha))
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture ready state")
    _git(root, "branch", "-f", "main", "HEAD")
    _git(root, "switch", "main")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "update-ref", "-d", "refs/heads/feat/mmm-repository-single-source-taskctl-adoption-001")
    _git(root, "update-ref", "-d", "refs/remotes/origin/feat/mmm-repository-single-source-taskctl-adoption-001")
    taskctl.transition(
        root,
        _args(
            "merged",
            reviewed_head_sha=sha,
            local_feature_branch_cleanup="observed_deleted",
            remote_feature_branch_cleanup="observed_deleted",
        ),
    )
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert state["status"] == "merged"


@pytest.mark.parametrize("surviving_ref", ("local", "remote"))
def test_merged_cleanup_rejects_surviving_refs(tmp_path: Path, surviving_ref: str) -> None:
    root, sha = _feature_fixture(tmp_path / surviving_ref)
    taskctl.transition(root, _args("in_progress"))
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha))
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture ready state")
    feature = "feat/mmm-repository-single-source-taskctl-adoption-001"
    _git(root, "branch", "-f", "main", "HEAD")
    _git(root, "switch", "main")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    if surviving_ref == "local":
        pass
    else:
        _git(root, "update-ref", "-d", f"refs/heads/{feature}")
        _git(root, "update-ref", f"refs/remotes/origin/{feature}", "HEAD")
    with pytest.raises(taskctl.TaskControlError, match="E_CLEANUP"):
        taskctl.transition(
            root,
            _args(
                "merged",
                reviewed_head_sha=sha,
                local_feature_branch_cleanup="observed_deleted",
                remote_feature_branch_cleanup="observed_deleted",
            ),
        )


def test_every_declared_edge_has_a_representative_executable_path(tmp_path: Path) -> None:
    root, sha = _feature_fixture(tmp_path)
    taskctl.transition(root, _args("in_progress"))
    taskctl.transition(root, _args("blocked", blocker=["pause"], live_resolution_condition="resume"))
    taskctl.transition(root, _args("in_progress", clear_blockers=True))
    taskctl.transition(root, _args("ready_for_review", implementation_sha=sha))
    state = json.loads((root / taskctl.STATE_PATH).read_text(encoding="utf-8"))
    assert state["status"] == "ready_for_review"
    with pytest.raises(taskctl.TaskControlError, match="E_TRANSITION"):
        taskctl.transition(root, _args("authorized"))
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
