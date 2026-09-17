"""Fail-closed controller for MMM's canonical execution lifecycle state."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_V2 = "mmm_repo_execution_state_v2"
SCHEMA_V3 = "mmm_repo_execution_state_v3"
REPOSITORY = "Phani-Pavuluri/MMM"
STATE_PATH = Path("docs/execution/EXECUTION_STATE.json")
TASK_PATH = Path("docs/execution/ACTIVE_TASK.md")
REPORT_PATH = Path("docs/execution/LATEST_COMPLETION_REPORT.md")
BEGIN = "<!-- BEGIN MMM TASKCTL EXECUTION VIEW -->"
END = "<!-- END MMM TASKCTL EXECUTION VIEW -->"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TASK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STATUSES = {
    "proposed",
    "authorized",
    "in_progress",
    "blocked",
    "changes_requested",
    "ready_for_review",
    "merged",
}
EDGES = {
    "proposed": {"authorized"},
    "authorized": {"in_progress", "blocked", "ready_for_review"},
    "in_progress": {"blocked", "ready_for_review"},
    "blocked": {"in_progress", "ready_for_review"},
    "changes_requested": {"in_progress", "blocked", "ready_for_review"},
    "ready_for_review": {"changes_requested", "merged"},
    "merged": set(),
}
CLEANUP_VALUES = {"not_started", "not_required", "observed_deleted"}

V2_KEYS = {
    "schema_version",
    "repository",
    "task_id",
    "status",
    "execution_mode",
    "base_branch",
    "base_sha",
    "task_authoring_head_sha",
    "authorization_head_sha",
    "feature_branch",
    "feature_branch_created",
    "task_path",
    "completion_report_path",
    "canonical_mip_standard_repository",
    "canonical_mip_standard_commit",
    "mip_launcher_standard_dependency",
    "task_execution_authorized",
    "correction_execution_authorized",
    "merge_authorized",
    "pr_creation_authorized",
    "reviewed_head_sha",
    "rejected_review_head_sha",
    "implementation_commit_sha",
    "approval_commit_sha",
    "capability_authorizations_changed",
    "last_updated",
    "blockers",
    "max_correction_cycles",
    "risk_tier",
    "full_suite_validation_required",
    "validation_scope",
    "unresolved_execution_blocking_design_questions",
    "affected_repositories",
    "modified_repositories",
    "coordination_workstream_id",
    "coordination_capability_owner",
    "prior_task",
    "mmm_analytical_authority_changed",
    "sibling_authority_changed",
    "historical_nonconforming_merge",
    "review_decision",
    "review_decision_source",
    "task_authoring_note",
}
V3_KEYS = V2_KEYS | {
    "rejected_implementation_commit_sha",
    "correction_cycles_completed",
    "correction_cycles_remaining",
    "live_resolution_condition",
    "local_feature_branch_cleanup",
    "remote_feature_branch_cleanup",
}


class TaskControlError(Exception):
    """A stable, expected task-control failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def fail(code: str, message: str) -> NoReturn:
    raise TaskControlError(code, message)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail("E_GIT", f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def discover_root(value: str | None) -> Path:
    candidate = Path(value or ".").resolve()
    if not candidate.exists() or not candidate.is_dir():
        fail("E_ROOT", f"repository root is not a directory: {candidate}")
    try:
        root = Path(_git(candidate, "rev-parse", "--show-toplevel")).resolve()
    except TaskControlError as error:
        fail("E_ROOT", f"not a Git worktree: {candidate} ({error.message})")
    return root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail("E_STATE_MISSING", f"canonical state is missing: {path}")
    except json.JSONDecodeError as error:
        fail("E_JSON", f"malformed canonical JSON: {error.msg}")
    if not isinstance(value, dict):
        fail("E_STATE_TYPE", "canonical state must be a JSON object")
    return value


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _require_keys(state: dict[str, Any], expected: set[str]) -> None:
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    if missing or extra:
        fail("E_SCHEMA_KEYS", f"state keys differ; missing={missing}, extra={extra}")


def _validate_sha_values(value: Any, path: str = "state") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                key.endswith("_sha")
                and child is not None
                and (not isinstance(child, str) or not SHA_RE.fullmatch(child))
            ):
                fail("E_SHA", f"{child_path} must be null or a lowercase 40-character commit SHA")
            _validate_sha_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_sha_values(child, f"{path}[{index}]")


def _validate_common_types(state: dict[str, Any]) -> None:
    strings = {
        "repository",
        "task_id",
        "status",
        "execution_mode",
        "base_branch",
        "base_sha",
        "task_authoring_head_sha",
        "authorization_head_sha",
        "feature_branch",
        "task_path",
        "completion_report_path",
        "mip_launcher_standard_dependency",
        "last_updated",
        "risk_tier",
        "validation_scope",
        "coordination_workstream_id",
        "coordination_capability_owner",
        "review_decision",
        "review_decision_source",
        "task_authoring_note",
    }
    nullable_strings = {
        "canonical_mip_standard_repository",
        "canonical_mip_standard_commit",
        "reviewed_head_sha",
        "rejected_review_head_sha",
        "implementation_commit_sha",
        "approval_commit_sha",
    }
    booleans = {
        "feature_branch_created",
        "task_execution_authorized",
        "correction_execution_authorized",
        "merge_authorized",
        "pr_creation_authorized",
        "capability_authorizations_changed",
        "full_suite_validation_required",
        "mmm_analytical_authority_changed",
        "sibling_authority_changed",
    }
    for key in strings:
        if not isinstance(state[key], str) or not state[key]:
            fail("E_TYPE", f"{key} must be a nonempty string")
    for key in nullable_strings:
        if state[key] is not None and not isinstance(state[key], str):
            fail("E_TYPE", f"{key} must be a string or null")
    for key in booleans:
        if not _is_bool(state[key]):
            fail("E_TYPE", f"{key} must be a boolean")
    if type(state["max_correction_cycles"]) is not int or state["max_correction_cycles"] < 0:
        fail("E_TYPE", "max_correction_cycles must be a non-negative integer")
    for key in (
        "blockers",
        "unresolved_execution_blocking_design_questions",
        "affected_repositories",
        "modified_repositories",
    ):
        if not isinstance(state[key], list):
            fail("E_TYPE", f"{key} must be a list")
    if not isinstance(state["prior_task"], dict) or not isinstance(state["historical_nonconforming_merge"], dict):
        fail("E_TYPE", "prior_task and historical_nonconforming_merge must be objects")
    _validate_sha_values(state)


def _validate_identity_fields(state: dict[str, Any]) -> None:
    if state["repository"] != REPOSITORY:
        fail("E_REPOSITORY", f"repository must be {REPOSITORY}")
    if not TASK_RE.fullmatch(state["task_id"]):
        fail("E_TASK", "task_id must be a nonempty safe identifier")
    if state["base_branch"] != "main" or state["execution_mode"] != "branch_and_fast_forward":
        fail("E_IDENTITY", "base branch and execution mode do not match MMM execution policy")
    branch = state["feature_branch"]
    if branch == "main" or not branch or ".." in branch or "//" in branch or "@{" in branch:
        fail("E_BRANCH", "feature branch must be a safe non-main branch name")
    if subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{branch}"], check=False, capture_output=True
    ).returncode:
        fail("E_BRANCH", "feature branch is not valid Git branch syntax")
    if state["task_path"] != str(TASK_PATH) or state["completion_report_path"] != str(REPORT_PATH):
        fail("E_PATH", "state view paths do not match the canonical MMM views")


def _validate_lifecycle(state: dict[str, Any]) -> None:
    status = state["status"]
    if status not in STATUSES or state["review_decision"] not in STATUSES:
        fail("E_STATUS", "status and review_decision must be supported lifecycle values")
    if any(
        state[key] is not False
        for key in (
            "merge_authorized",
            "pr_creation_authorized",
            "mmm_analytical_authority_changed",
            "sibling_authority_changed",
            "capability_authorizations_changed",
        )
    ):
        fail("E_PROTECTED_AUTHORITY", "merge, PR, analytical, sibling, and capability authority must remain false")
    if not all(isinstance(item, str) and item.strip() for item in state["blockers"]):
        fail("E_BLOCKERS", "blockers must contain only nonempty strings")
    completed = state["correction_cycles_completed"]
    remaining = state["correction_cycles_remaining"]
    if type(completed) is not int or type(remaining) is not int or completed < 0 or remaining < 0:
        fail("E_CORRECTION_COUNTER", "correction counters must be non-negative integers")
    if completed + remaining != state["max_correction_cycles"]:
        fail("E_CORRECTION_COUNTER", "completed plus remaining corrections must equal max_correction_cycles")
    for key in ("local_feature_branch_cleanup", "remote_feature_branch_cleanup"):
        if state[key] not in CLEANUP_VALUES:
            fail("E_CLEANUP", f"{key} has an unsupported cleanup value")
    resolution = state["live_resolution_condition"]
    if resolution is not None and (not isinstance(resolution, str) or not resolution.strip()):
        fail("E_RESOLUTION", "live_resolution_condition must be null or a nonempty string")
    implementation = state["implementation_commit_sha"]
    rejected_review = state["rejected_review_head_sha"]
    rejected_implementation = state["rejected_implementation_commit_sha"]
    if (rejected_review is None) != (rejected_implementation is None):
        fail("E_REJECTION_PAIR", "rejected review and implementation SHAs must be paired")
    if status == "proposed":
        if any(
            (
                state["task_execution_authorized"],
                state["correction_execution_authorized"],
                state["feature_branch_created"],
                implementation,
                state["reviewed_head_sha"],
                rejected_review,
                state["approval_commit_sha"],
                state["blockers"],
            )
        ):
            fail("E_LIFECYCLE", "proposed state contains execution, branch, or review evidence")
    elif status in {"authorized", "in_progress"}:
        if not state["task_execution_authorized"] or state["correction_execution_authorized"] or state["blockers"]:
            fail("E_LIFECYCLE", f"{status} has invalid execution or blocker evidence")
        if (
            implementation is not None
            or state["reviewed_head_sha"] is not None
            or state["approval_commit_sha"] is not None
        ):
            fail("E_LIFECYCLE", f"{status} has review-ready or approval evidence")
    elif status == "blocked":
        if (
            state["task_execution_authorized"]
            or state["correction_execution_authorized"]
            or not state["blockers"]
            or not resolution
            or state["review_decision"] != "blocked"
        ):
            fail("E_LIFECYCLE", "blocked requires closed execution, blockers, resolution, and blocked decision")
    elif status == "changes_requested":
        if (
            state["task_execution_authorized"]
            or not state["correction_execution_authorized"]
            or not implementation
            or not rejected_review
            or state["reviewed_head_sha"] is not None
            or state["approval_commit_sha"] is not None
            or state["review_decision"] != status
        ):
            fail("E_LIFECYCLE", "changes_requested requires correction authority and paired rejection provenance")
    elif status == "ready_for_review":
        if (
            not state["task_execution_authorized"]
            or state["correction_execution_authorized"]
            or not implementation
            or state["blockers"]
            or state["reviewed_head_sha"] is not None
            or state["approval_commit_sha"] is not None
            or state["review_decision"] != status
        ):
            fail("E_LIFECYCLE", "ready_for_review requires implementation and closed review authority")
    elif status == "merged":
        if (
            state["task_execution_authorized"]
            or state["correction_execution_authorized"]
            or not implementation
            or not state["reviewed_head_sha"]
            or state["blockers"]
            or state["review_decision"] != status
        ):
            fail("E_LIFECYCLE", "merged requires closed execution and implementation/review evidence")
        if (
            state["local_feature_branch_cleanup"] != "observed_deleted"
            or state["remote_feature_branch_cleanup"] != "observed_deleted"
        ):
            fail("E_CLEANUP", "merged requires observed local and remote feature-branch deletion")
    if status != "blocked" and resolution is not None:
        fail("E_RESOLUTION", "only blocked state may retain a live resolution condition")


def _validate_v2_migration_source(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_V2:
        fail("E_SCHEMA", f"sync can migrate only {SCHEMA_V2}")
    _require_keys(state, V2_KEYS)
    _validate_common_types(state)
    _validate_identity_fields(state)
    if state["status"] != "authorized" or state["review_decision"] != "authorized":
        fail("E_MIGRATION_AMBIGUOUS", "only the exact current authorized v2 state is migratable")
    if (
        state["feature_branch_created"]
        or not state["task_execution_authorized"]
        or state["correction_execution_authorized"]
        or state["blockers"]
    ):
        fail("E_MIGRATION_AMBIGUOUS", "v2 execution evidence is not the exact authorized migration source")
    if any(
        state[key] is not None
        for key in ("implementation_commit_sha", "reviewed_head_sha", "rejected_review_head_sha", "approval_commit_sha")
    ):
        fail("E_MIGRATION_AMBIGUOUS", "v2 review evidence is not migratable")


def validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_V3:
        if state.get("schema_version") == SCHEMA_V2:
            fail("E_MIGRATION_REQUIRED", "v2 state requires `taskctl sync` exact migration")
        fail("E_SCHEMA", f"unsupported schema version: {state.get('schema_version')!r}")
    _require_keys(state, V3_KEYS)
    _validate_common_types(state)
    _validate_identity_fields(state)
    for key in ("rejected_implementation_commit_sha", "live_resolution_condition"):
        if state[key] is not None and not isinstance(state[key], str):
            fail("E_TYPE", f"{key} must be a string or null")
    if type(state["correction_cycles_completed"]) is not int or type(state["correction_cycles_remaining"]) is not int:
        fail("E_TYPE", "correction lifecycle counters must be integers")
    if not isinstance(state["local_feature_branch_cleanup"], str) or not isinstance(
        state["remote_feature_branch_cleanup"], str
    ):
        fail("E_TYPE", "cleanup evidence must be strings")
    _validate_sha_values(state)
    _validate_lifecycle(state)


def _commit_fields(value: Any, key: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(_commit_fields(child, child_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_commit_fields(child))
    elif key.endswith("_sha") and isinstance(value, str):
        found.append(value)
    return found


def validate_git_identity(root: Path, state: dict[str, Any]) -> None:
    origin = _git(root, "config", "--get", "remote.origin.url")
    normalized = (
        origin.removesuffix(".git")
        .removeprefix("git@github.com:")
        .removeprefix("https://github.com/")
        .removeprefix("ssh://git@github.com/")
    )
    if normalized != REPOSITORY:
        fail("E_ORIGIN", f"origin must identify {REPOSITORY}; found {origin}")
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch not in {"main", state["feature_branch"]}:
        fail("E_CURRENT_BRANCH", f"taskctl only permits main or {state['feature_branch']}; found {branch}")
    for sha in _commit_fields(state):
        _git(root, "cat-file", "-e", f"{sha}^{{commit}}")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", state["authorization_head_sha"], "HEAD"], check=False
    ).returncode:
        fail("E_AUTHORIZATION_ANCESTRY", "authorization_head_sha is not an ancestor of checked HEAD")
    if branch == "main":
        origin_main = _git(root, "rev-parse", "origin/main")
        if _git(root, "rev-parse", "HEAD") != origin_main:
            fail("E_MAIN_SYNC", "main must equal origin/main")
    if state["status"] == "merged":
        for ref in (f"refs/heads/{state['feature_branch']}", f"refs/remotes/origin/{state['feature_branch']}"):
            if subprocess.run(
                ["git", "-C", str(root), "show-ref", "--verify", "--quiet", ref],
                check=False,
            ).returncode == 0:
                fail("E_CLEANUP", f"merged task branch ref still exists: {ref}")


def render_view(state: dict[str, Any]) -> str:
    def value(key: str) -> str:
        item = state[key]
        if isinstance(item, bool):
            return str(item).lower()
        if item is None:
            return "null"
        if isinstance(item, list):
            return "; ".join(item) if item else "[]"
        return str(item)

    fields = (
        ("Task ID", "task_id"),
        ("Repository", "repository"),
        ("Status", "status"),
        ("Decision", "review_decision"),
        ("Execution mode", "execution_mode"),
        ("Base SHA", "base_sha"),
        ("Task authoring SHA", "task_authoring_head_sha"),
        ("Authorization SHA", "authorization_head_sha"),
        ("Feature branch", "feature_branch"),
        ("Feature branch created", "feature_branch_created"),
        ("Task execution authorized", "task_execution_authorized"),
        ("Correction execution authorized", "correction_execution_authorized"),
        ("Merge authorized", "merge_authorized"),
        ("PR creation authorized", "pr_creation_authorized"),
        ("Implementation SHA", "implementation_commit_sha"),
        ("Reviewed head SHA", "reviewed_head_sha"),
        ("Rejected review head SHA", "rejected_review_head_sha"),
        ("Rejected implementation SHA", "rejected_implementation_commit_sha"),
        ("Approval SHA", "approval_commit_sha"),
        ("Blockers", "blockers"),
        ("Live resolution condition", "live_resolution_condition"),
        ("Correction cycles completed", "correction_cycles_completed"),
        ("Correction cycles remaining", "correction_cycles_remaining"),
        ("Local feature branch cleanup", "local_feature_branch_cleanup"),
        ("Remote feature branch cleanup", "remote_feature_branch_cleanup"),
        ("Analytical authority changed", "mmm_analytical_authority_changed"),
        ("Sibling authority changed", "sibling_authority_changed"),
        ("Capability authorizations changed", "capability_authorizations_changed"),
    )
    lines = [BEGIN, "<!-- Generated by mmm.execution.taskctl; do not edit. -->"]
    lines.extend(f"- {label}: `{value(key)}`" for label, key in fields)
    lines.append(END)
    return "\n".join(lines) + "\n"


def _extract_block(text: str, path: Path) -> str:
    begins = len(re.findall(rf"(?m)^{re.escape(BEGIN)}$", text))
    ends = len(re.findall(rf"(?m)^{re.escape(END)}$", text))
    if begins != 1 or ends != 1:
        fail("E_MARKERS", f"{path} must contain exactly one begin/end marker pair")
    start, finish = text.index(BEGIN), text.index(END)
    if start > finish:
        fail("E_MARKERS", f"{path} has reversed taskctl markers")
    finish += len(END)
    if finish < len(text) and text[finish] == "\n":
        finish += 1
    return text[start:finish]


def _replace_existing_block(text: str, path: Path, block: str) -> str:
    existing = _extract_block(text, path)
    return text.replace(existing, block, 1)


def _insert_after_h1(text: str, path: Path, block: str) -> str:
    if re.search(rf"(?m)^{re.escape(BEGIN)}$|^{re.escape(END)}$", text):
        fail("E_MARKERS", f"{path} contains a partial taskctl marker pair")
    match = re.match(r"^# [^\n]+\n", text)
    if not match:
        fail("E_MARKERS", f"{path} must begin with one H1 for migration")
    return text[: match.end()] + "\n" + block + text[match.end() :]


def _migrate_task_text(text: str) -> str:
    pattern = r"^(# [^\n]+\n\n)\*\*Lifecycle:\*\* [^\n]*\n"
    migrated, count = re.subn(pattern, r"\1", text, count=1)
    if count != 1:
        fail("E_MIGRATION_MARKER", "ACTIVE_TASK.md lacks the exact leading manual lifecycle line")
    return migrated


def _view_candidates(root: Path, state: dict[str, Any], migration: bool) -> dict[Path, str]:
    block = render_view(state)
    task_text = (root / TASK_PATH).read_text(encoding="utf-8")
    report_text = (root / REPORT_PATH).read_text(encoding="utf-8")
    if migration:
        task_text = _migrate_task_text(task_text)
        return {
            root / TASK_PATH: _insert_after_h1(task_text, TASK_PATH, block),
            root / REPORT_PATH: _insert_after_h1(report_text, REPORT_PATH, block),
        }
    return {
        root / TASK_PATH: _replace_existing_block(task_text, TASK_PATH, block),
        root / REPORT_PATH: _replace_existing_block(report_text, REPORT_PATH, block),
    }


def _atomic_write(path: Path, text: str) -> None:
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temp = Path(handle.name)
        os.replace(temp, path)
    except OSError as error:
        if temp is not None:
            with suppress(OSError):
                temp.unlink(missing_ok=True)
        fail("E_WRITE", f"atomic replacement failed for {path}: {error}")


def _migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(state)
    candidate["schema_version"] = SCHEMA_V3
    candidate["rejected_implementation_commit_sha"] = None
    candidate["correction_cycles_completed"] = 0
    candidate["correction_cycles_remaining"] = candidate["max_correction_cycles"]
    candidate["live_resolution_condition"] = None
    candidate["local_feature_branch_cleanup"] = "not_started"
    candidate["remote_feature_branch_cleanup"] = "not_started"
    return candidate


def check(root: Path) -> None:
    state = _read_json(root / STATE_PATH)
    validate_state(state)
    validate_git_identity(root, state)
    expected = render_view(state)
    for path in (TASK_PATH, REPORT_PATH):
        actual = _extract_block((root / path).read_text(encoding="utf-8"), path)
        if actual != expected:
            fail("E_VIEW_DIVERGENCE", f"generated execution view diverges in {path}")


def sync(root: Path) -> None:
    original = _read_json(root / STATE_PATH)
    migration = original.get("schema_version") == SCHEMA_V2
    if migration:
        _validate_v2_migration_source(original)
        candidate = _migrate_state(original)
    else:
        candidate = original
    validate_state(candidate)
    validate_git_identity(root, candidate)
    candidates = _view_candidates(root, candidate, migration)
    state_text = json.dumps(candidate, indent=2) + "\n"
    if migration or (root / STATE_PATH).read_text(encoding="utf-8") != state_text:
        _atomic_write(root / STATE_PATH, state_text)
    for path, text in candidates.items():
        if path.read_text(encoding="utf-8") != text:
            _atomic_write(path, text)
    check(root)


def _require_sha(value: str | None, label: str) -> str:
    if value is None or not SHA_RE.fullmatch(value):
        fail("E_EVIDENCE", f"{label} must be a lowercase 40-character commit SHA")
    return value


def transition(root: Path, args: argparse.Namespace) -> None:
    current = _read_json(root / STATE_PATH)
    validate_state(current)
    validate_git_identity(root, current)
    check(root)
    target = args.to
    if target not in EDGES[current["status"]]:
        fail("E_TRANSITION", f"{current['status']} -> {target} is not an allowed lifecycle transition")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    if (current["status"], target) in {("proposed", "authorized"), ("ready_for_review", "merged")}:
        if branch != "main":
            fail("E_TRANSITION_BRANCH", "this lifecycle transition is main-only")
    elif branch != current["feature_branch"]:
        fail("E_TRANSITION_BRANCH", "mutable lifecycle transitions are feature-branch-only")
    candidate = deepcopy(current)
    candidate["status"] = target
    candidate["review_decision"] = target
    candidate["feature_branch_created"] = target != "proposed"
    candidate["task_execution_authorized"] = target in {"authorized", "in_progress", "ready_for_review"}
    candidate["correction_execution_authorized"] = target == "changes_requested"
    if args.implementation_sha is not None:
        candidate["implementation_commit_sha"] = _require_sha(args.implementation_sha, "implementation SHA")
    if args.rejected_review_head_sha is not None:
        candidate["rejected_review_head_sha"] = _require_sha(args.rejected_review_head_sha, "rejected review head SHA")
    if args.rejected_implementation_commit_sha is not None:
        candidate["rejected_implementation_commit_sha"] = _require_sha(
            args.rejected_implementation_commit_sha, "rejected implementation SHA"
        )
    if args.reviewed_head_sha is not None:
        candidate["reviewed_head_sha"] = _require_sha(args.reviewed_head_sha, "reviewed head SHA")
    if args.local_feature_branch_cleanup is not None:
        candidate["local_feature_branch_cleanup"] = args.local_feature_branch_cleanup
    if args.remote_feature_branch_cleanup is not None:
        candidate["remote_feature_branch_cleanup"] = args.remote_feature_branch_cleanup
    clear_blockers = getattr(args, "clear_blockers", False)
    if clear_blockers and args.blocker is not None:
        fail("E_EVIDENCE", "--clear-blockers cannot be combined with --blocker")
    if target == "blocked":
        if clear_blockers:
            fail("E_EVIDENCE", "--clear-blockers is not valid for blocked")
        if args.blocker is not None:
            candidate["blockers"] = args.blocker
        if args.live_resolution_condition is not None:
            candidate["live_resolution_condition"] = args.live_resolution_condition
    else:
        if args.blocker is not None or args.live_resolution_condition is not None:
            fail("E_EVIDENCE", "blockers and live resolution evidence are only valid for blocked")
        if current["blockers"] and not clear_blockers:
            fail("E_BLOCKER_CLEAR_REQUIRED", "clearing existing blockers requires --clear-blockers")
        candidate["blockers"] = []
        candidate["live_resolution_condition"] = None
    if target == "in_progress" and current["status"] == "changes_requested":
        candidate["implementation_commit_sha"] = None
    if target == "changes_requested":
        if candidate["correction_cycles_remaining"] == 0:
            fail("E_CORRECTION", "no correction cycles remain")
        _require_sha(candidate["implementation_commit_sha"], "implementation SHA")
        _require_sha(candidate["rejected_review_head_sha"], "rejected review head SHA")
        _require_sha(candidate["rejected_implementation_commit_sha"], "rejected implementation SHA")
    if target == "ready_for_review":
        _require_sha(candidate["implementation_commit_sha"], "implementation SHA")
        if current["status"] in {"changes_requested", "in_progress"} and current["rejected_review_head_sha"]:
            if not args.complete_correction:
                fail("E_CORRECTION", "correction completion requires --complete-correction")
            if candidate["correction_cycles_remaining"] == 0:
                fail("E_CORRECTION", "no correction cycles remain")
            candidate["correction_cycles_completed"] += 1
            candidate["correction_cycles_remaining"] -= 1
        elif args.complete_correction:
            fail("E_CORRECTION", "--complete-correction is only valid after changes_requested")
        candidate["reviewed_head_sha"] = None
        candidate["approval_commit_sha"] = None
    elif args.complete_correction:
        fail("E_CORRECTION", "--complete-correction is only valid for ready_for_review")
    if target != "changes_requested" and candidate["rejected_review_head_sha"] is None:
        candidate["rejected_implementation_commit_sha"] = None
    validate_state(candidate)
    validate_git_identity(root, candidate)
    candidates = _view_candidates(root, candidate, migration=False)
    _atomic_write(root / STATE_PATH, json.dumps(candidate, indent=2) + "\n")
    for path, text in candidates.items():
        _atomic_write(path, text)
    check(root)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m mmm.execution.taskctl")
    result.add_argument("--root", help="Git worktree root (defaults to the current path)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("sync")
    change = commands.add_parser("transition")
    change.add_argument("--to", required=True, choices=sorted(STATUSES))
    change.add_argument("--implementation-sha")
    change.add_argument("--rejected-review-head-sha")
    change.add_argument("--rejected-implementation-commit-sha")
    change.add_argument("--reviewed-head-sha")
    change.add_argument("--blocker", action="append")
    change.add_argument("--clear-blockers", action="store_true")
    change.add_argument("--live-resolution-condition")
    change.add_argument("--complete-correction", action="store_true")
    change.add_argument("--local-feature-branch-cleanup", choices=sorted(CLEANUP_VALUES))
    change.add_argument("--remote-feature-branch-cleanup", choices=sorted(CLEANUP_VALUES))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = discover_root(args.root)
        if args.command == "check":
            check(root)
        elif args.command == "sync":
            sync(root)
        else:
            transition(root, args)
    except TaskControlError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
