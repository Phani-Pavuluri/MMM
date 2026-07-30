# Active Task

**Status:** authorized
**Owner:** MMM repository governance
**Last updated:** 2026-07-30
**Task ID:** `MMM_REPO_NATIVE_EXECUTION_HANDOFF_ADOPTION_001`

## Identity

- **Repository:** `Phani-Pavuluri/MMM`
- **Verified code checkpoint:** `9a3aa5cb9a48c9a59d45e266685228835237f328`
- **Base branch:** `main`
- **Feature branch:** `feat/mmm-repo-native-execution-handoff-adoption-001`
- **Execution mode:** `branch_and_fast_forward`
- **Canonical MIP workflow pin:** `Phani-Pavuluri/marketing_intelligence_platform@5eebba6750a3754e4026397d6762c601b1d6a708`
- **Canonical standard:** `docs/execution/TASK_EXECUTION_STANDARD.md` in the pinned MIP commit

The task-authoring commits that add this file and `EXECUTION_STATE.json` to MMM
`main` are metadata only. Before creating the feature branch, verify the pinned
code checkpoint is an ancestor of current `main` and that every change since it
is limited to those two task-authoring files. Stop on any product-code or other
unexpected change.

## Objective

Adopt the MIP repo-native task, completion-report, review, merge, and fresh-chat
handoff workflow in MMM without creating an independent competing governance
framework. This task changes workflow metadata only.

## Prerequisites

1. Local and remote MMM `main` agree.
2. MMM checkpoint `9a3aa5cb9a48c9a59d45e266685228835237f328`
   exists and is an ancestor of current `main`.
3. MIP pin `5eebba6750a3754e4026397d6762c601b1d6a708`
   exists on MIP `main` and contains the canonical execution standard, fresh-chat
   bootstrap, stable-path model, and merged closure state.
4. `.codex/` and `docs/tasks/` remain local-only and untracked.
5. No prior MMM execution-handoff adoption is already present.

## Owned files

Create or replace only:

- `AGENTS.md`
- `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
- `docs/execution/ACTIVE_TASK.md`
- `docs/execution/LATEST_COMPLETION_REPORT.md`
- `docs/execution/EXECUTION_STATE.json`
- one focused reusable consistency test in the repository's existing governance
  or documentation test location
- one documentation index only when required for discoverability

Do not copy the MIP `TASK_EXECUTION_STANDARD.md` into MMM. Pin and reference the
canonical MIP commit instead.

## Required implementation

### `AGENTS.md`

Require every Codex session to read, in order:

1. `docs/execution/EXECUTION_STATE.json`
2. `docs/execution/ACTIVE_TASK.md`
3. `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
4. relevant MMM contracts, roadmaps, validation evidence, and the pinned MIP
   standard/program files

For `Execute the active task`, require fail-closed verification of authorization,
checkpoint ancestry, task/state agreement, exact feature branch, prerequisites,
owned files, focused/full validation, completion-report creation, state change to
`ready_for_review`, commit/push, and stop without merging.

For `Merge the approved active task`, require exact reviewed-head integrity,
`approved_for_merge`, `merge_authorized: true`, fast-forward-only merge, push,
synchronization, closure metadata, and local/remote branch cleanup. No PR is
required. Stop rather than guess.

### `REPOSITORY_CONTEXT_INDEX.md`

Create a concise navigation index, not a duplicate roadmap. Include:

- the pinned MIP execution standard and seven canonical MIP `docs/program/*`
  files;
- MMM public simulation handoff and `MMMPublicSimulationExport` evidence;
- supported-range evidence;
- `MMMCalibrationCompatibilityResult` contract, parser, fixtures, and schema
  compatibility policy;
- current MMM roadmap, validation/platform status, open investigations, and
  deferred package-side agent roadmap;
- GeoX as the experiment-readout producer and MIP as consumer/orchestrator;
- exact checkpoint verification requirements.

Add a **Fresh Chat Bootstrap** prompt that instructs a fresh ChatGPT chat to use
connected GitHub as source of truth, read the stable MMM execution files and
pinned MIP program files, verify current main and cross-repository checkpoints,
and summarize current state, active task, latest completion, blockers,
dependencies, authority boundaries, and next eligible work without modifying or
authorizing anything.

### Stable execution files

Use one replace-in-place copy of:

- `docs/execution/ACTIVE_TASK.md`
- `docs/execution/LATEST_COMPLETION_REPORT.md`
- `docs/execution/EXECUTION_STATE.json`

Git history preserves prior versions. Do not create per-task report archives.

The completed branch state must use schema
`mmm_repo_execution_state_v1`, task ID
`MMM_REPO_NATIVE_EXECUTION_HANDOFF_ADOPTION_001`, status
`ready_for_review`, `task_execution_authorized: true`,
`merge_authorized: false`, null reviewed/approval SHAs, a populated
implementation commit SHA, `capability_authorizations_changed: false`, and no
blockers.

### Completion report

Write `TASK_COMPLETION_REPORT_V1` with exact changed files, prerequisites,
deliverables, acceptance results, focused and full validation, Ruff, mypy,
`git diff --check`, Docker-backed `make validate`, GitHub-observed versus local
evidence, limitations, deferred work, merge readiness, and local-only paths.

Also report explicitly whether the task changed or authorized any of:

- model fitting or calibration behavior;
- simulation or supported-range semantics;
- optimization or candidate generation;
- recommendation authority;
- Bayesian production status;
- automatic refit or model promotion;
- public export schemas or numerical truth.

For this adoption task, every item above must remain unchanged and unauthorized.

## Focused test

Add a reusable test that derives the task ID from state rather than hardcoding
this bootstrap task forever. Verify required paths, JSON parsing, schema/status
vocabulary, task/state/report agreement, boolean fields, stable AGENTS paths,
fresh-chat bootstrap, and state-specific invariants:

- `ready_for_review` requires execution authorized, merge unauthorized,
  implementation SHA populated, and reviewed/approval SHAs null;
- `approved_for_merge` requires execution and merge authorization plus populated
  reviewed and approval SHAs;
- this adoption task specifically requires
  `capability_authorizations_changed: false`.

## Validation

Run repository-standard JSON and Markdown/path checks, the focused execution
consistency test, relevant documentation/governance tests, Ruff, mypy,
`git diff --check`, and Docker-backed `make validate`.

If Docker, dependency download, repository state, or prerequisite verification
fails, stop and preserve the branch. Do not claim success.

## Prohibited scope

Do not change MMM models, contracts, parsers, fixtures, numerical outputs,
calibration, simulation, optimization, Bayesian authority, recommendations,
production status, MIP, or GeoX. Do not add a scheduler, GitHub Action, workflow
engine, custom agent, PR template, or package-side agent.

## Commit, push, and stop

After successful validation:

- commit with message `Adopt repo-native execution handoff workflow`;
- push `feat/mmm-repo-native-execution-handoff-adoption-001`;
- do not create a PR;
- do not merge;
- do not delete the branch;
- stop for ChatGPT review.
