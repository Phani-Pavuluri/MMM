# Active Task

**Status:** proposed pending state-only authorization
**Owner:** MMM repository governance
**Last updated:** 2026-08-04
**Last verified:** 2026-08-04

## Identity

- **Task ID:** `MMM_EXECUTION_AUTHORITY_AND_OPERATIONAL_LAUNCHER_ALIGNMENT_001`
- **Repository:** `Phani-Pavuluri/MMM`
- **Pre-authoring base:** `f2e0eade0ad917c1b28ab5521e6d35a35047d988`
- **Feature branch:** `docs/mmm-execution-authority-operational-launcher-alignment-001`
- **Execution mode:** `branch_and_fast_forward`
- **Risk tier:** Tier 1 repository-execution governance with mandatory MMM Docker-backed full validation
- **Capability authorizations changed:** `false`

## Superseded prior proposal

`MMM_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_ADOPTION_001` is `superseded_without_execution`.
Its intended branch was never created; its MIP dependency and blocker are retired; and it had no implementation, review, correction, merge, PR, analytical, sibling, or capability authority. It is prior-task history only and must not be revived.

## Primary independently reviewable outcome

Adopt a simple MMM-owned execution model: synchronized `main` owns task identity and authorization provenance; the verified remote feature branch owns current lifecycle; `ACTIVE_TASK.md` owns implementation meaning; and `LATEST_COMPLETION_REPORT.md` owns execution and validation evidence. Compact operational launchers may direct Git synchronization and continuation but never copy or reinterpret task meaning.

This is governance only. It changes no analytical, contract, fixture, numerical, runtime, package, MIP, GeoX, sibling, product, or capability behavior.

## Why this task cannot be split further

Authority hierarchy, fail-closed conflicts, launcher limits, output handoff, exact-tree validation, focused semantic checks, and retained MMM merge controls are one execution surface. Prose-only or test-only work would be contradictory. Independently valid work, public contracts, migrations, integration surfaces, or authority boundaries must be separate successors rather than silent widening.

## Resolved behavior

### Authority hierarchy

Synchronized `main/docs/execution/EXECUTION_STATE.json` owns repository identity, task identity, authorization provenance, authorization head, and declared feature branch. Codex resolves that branch from synchronized `main`, fetches its exact remote branch, and verifies repository identity, task identity, declared branch name, and authorization-head ancestry.

The verified remote feature branch's `EXECUTION_STATE.json` owns `authorized`, `in_progress`, `blocked`, `changes_requested`, and `ready_for_review`; blockers, corrections, implementation evidence, and completion reporting. `ACTIVE_TASK.md` owns objective, behavior, prerequisites, owned/prohibited paths, acceptance evidence, validation, and stop conditions. `LATEST_COMPLETION_REPORT.md` is evidence only and cannot authorize execution, correction, merge, sibling, analytical, or capability work.

### Fail-closed conflicts

Codex never chooses whichever file appears newer. Disagreement involving repository, task ID, feature branch, authorization head, branch ancestry, implementation SHA, or incompatible lifecycle/authority flags stops implementation. With a safe authorized feature branch, publish Git-durable `blocked` with the exact mismatch, attempted evidence, affected validation categories, and a live resolution condition. Chat, cached prompts, and completion prose cannot resolve conflict.

### Compact operational launcher

Launchers may contain only local repository path; synchronization and required Git reads; task/branch resolution from synchronized Git; remote branch verification/resumption; continuation through implementation, validation, publication, push, and remote verification; non-terminal progress; durable terminal outcomes; prohibited Git operations; and the external exact SHA for merge. They may not define, copy, repair, override, or reinterpret task ID, branch name, non-approved SHA, objective, implementation meaning, paths, prerequisites, tests/counts, correction defects, sibling lifecycle, analytical, or capability authority.

```text
Work in <local repository path>.

Synchronize main from Git and read AGENTS.md and the repository execution
files. Resolve the authorized task, authorization provenance and exact feature
branch from synchronized main.

Fetch and verify that remote feature branch, including repository identity,
task identity, authorization ancestry and current execution state.

Execute the Git-authored active task through required validation, durable
publication, push and remote-head verification.

Do not guess through conflicting state. Publish a Git-durable blocked state
with the exact mismatch when a safe authorized branch exists.

Progress reports are non-terminal. Stop only at a remotely published
ready_for_review or genuine blocked state.

Do not create a PR, merge, squash, rebase, force-push or change sibling,
analytical or capability authority.
```

Correction uses the same launcher and reads rejected SHAs/fixes from the remote branch's Git-authored `changes_requested` state. Merge adds only `Approved exact remote head: <FULL_SHA>`.

### Output handoff

Successful or Git-durable blocked execution requires only repository, feature branch, and exact remote head SHA. Chat output is diagnostic context only when no durable remote state was published.

## Inputs, outputs, invariants, and failure semantics

- **Inputs:** synchronized MMM main, its execution files, and the verified exact remote branch.
- **Outputs:** standards/tests expressing this model and a remote `ready_for_review` receipt or durable `blocked` state.
- **Invariants:** Git authority; no stale MIP dependency/blocker; unchanged analytical, sibling, and capability authority; retained exact-tree, exact-head, fast-forward-only, cleanup, and PR #19 history.
- **Failure semantics:** conflicts, unavailable required validation, unsafe branch, or unresolvable Git evidence fail closed; new independent decisions become successors.
- **Compatibility/migration:** `not_applicable`; no public contract migration occurs.

## Acceptance evidence

`tests/test_repo_native_execution_handoff.py` separately proves: main and branch authority separation; mandatory remote identity/ancestry verification; `ACTIVE_TASK.md` implementation authority; evidence-only completion reports; fail-closed prompt-irreconcilable conflicts; launcher bounds; non-terminal progress; remote `ready_for_review`/`blocked` outcomes; repository/branch/SHA handoff; and retained full validation, exact-tree, exact-head, fast-forward, closure, cleanup, and PR #19 controls. Tests use structural current task/state checks, not a transient workstream ID.

## Owned paths

1. `AGENTS.md`
2. `docs/execution/TASK_EXECUTION_STANDARD.md`
3. `tests/test_repo_native_execution_handoff.py`
4. `docs/execution/ACTIVE_TASK.md`
5. `docs/execution/EXECUTION_STATE.json`
6. `docs/execution/LATEST_COMPLETION_REPORT.md`

## Prohibited scope

Do not modify or authorize `mmm/**`; analytical code/tests; models, diagnostics, calibration, simulation, optimization, numerical truth; contracts, adapters, parsers, fixtures, schemas, runtime, package, release, CI, or deployment; MIP/GeoX; CalibrationSignal, TrustReport, DecisionSurface, planning, recommendations, real data, or production; task manifests, prompt/Markdown generators, validation registries, repository adapters, lookup ledgers, incident generators, `taskctl`, or cross-repository orchestration. Do not create a PR, merge, squash, rebase, force-push, merge commit, or pre-merge approval commit.

## Validation

On the frozen exact task-owned tree run `python -m json.tool docs/execution/EXECUTION_STATE.json >/dev/null`, `poetry run pytest -q tests/test_repo_native_execution_handoff.py`, `poetry run ruff check tests/test_repo_native_execution_handoff.py`, `poetry run mypy tests/test_repo_native_execution_handoff.py`, `git diff --check`, and `make validate`. Also verify changed paths, task/repository/branch consistency, authorization-head ancestry, lifecycle/authority compatibility, implementation/report consistency, no stale MIP dependency/blocker, and exact local/remote feature-head equality. Repair host Poetry reasonably before calling it an environment blocker; do not start duplicate validation containers.

## Publication and stop conditions

Create one implementation commit and one final exact-tree receipt commit. The receipt records milestone, branch, implementation SHA, exact remote branch head, changed paths, behavior, every validation result and exact full-suite counts, validations not run, blockers, limitations, validation debt, repository/sibling and consumer impact, authority impact, worktree/evidence source, and no PR/merge. Publish `ready_for_review` with execution true; correction/merge/PR false; empty blockers; implementation SHA; null reviewed/approval SHAs; and unchanged analytical, sibling, and capability authority. Push, verify remote equality, and stop for external review.

## Deferred successors

- Cross-repository coordination or orchestration implementation.
- Any analytical, package, public-contract, or capability work.

**Unresolved execution-blocking design questions: none.**
