# TASK_REVIEW_DECISION

## Decision

- **Task:** `MMM_REPOSITORY_EXECUTION_PROTOCOL_ADOPTION_001`
- **Rejected exact remote head:** `ccb25680b90fa6eb4ce4dc2d6f84051797641fa6`
- **Substantive implementation:** `bde826a4b21e35c1b313db781c8d3c1d7f39d2cc`
- **Decision:** `changes_requested`
- **Correction execution authorized:** `true`
- **Merge authorized:** `false`
- **PR creation authorized:** `false`
- **Capability authority changed:** `false`

The implementation direction is acceptable and the diff is limited to the eight task-owned paths, but the published head does not satisfy the task's frozen acceptance contract and is not approved for merge.

## Findings

### 1. Current lifecycle metadata is internally inconsistent

`EXECUTION_STATE.json` at the rejected head records `status: ready_for_review` while `review_decision` remains `authorized`. `ACTIVE_TASK.md` and the completion report say `ready_for_review`. Final publication must make these current-state fields coherent and must leave correction, merge, and PR authority false.

### 2. The seven focused tests do not assert every required behavior

The required test names exist, but several only check a small subset of their named semantic contract. Missing explicit assertions include:

- split triggers, surface-appropriate resolved decisions, failure semantics, and named acceptance evidence;
- continued execution without another prompt and the external-stop exception only when no safe authorized branch exists;
- mismatch blocking and the complete durable-receipt field set;
- all MMM full-validation triggers, required-failure-to-`blocked`, and duplicate-container prevention;
- protocol/state/history reads, exact mutable-branch evidence, workstream/owner conflict stops, and the complete cross-repository impact contract;
- `pr_creation_authorized: false`, branch cleanup, and the exact PR #19 number, head, merge SHA, and absence of conforming approval.

The correction must strengthen each of the seven semantic tests so the entire named acceptance group is independently enforced.

### 3. The exact-tree receipt message is incomplete

Receipt head `ccb25680b90fa6eb4ce4dc2d6f84051797641fa6` records the task, abbreviated parent, exact-tree scope, gate pass, focused count, high-level checks, and unchanged authority. It omits required worktree state, evidence source, exact Docker test count/disposition, and warning count/disposition. The replacement receipt must include the full implementation parent and every field required by the active task.

### 4. Validation reporting lacks exact full-suite counts

The completion report says Docker-backed `make validate` passed and the non-slow suite reached 100%, but it does not record exact test counts or warning counts/disposition as required. Rerun the complete gate on the corrected exact tree and record exact counts. Focused success alone is insufficient.

### 5. Sibling evidence moved after the rejected publication

The rejected report cites authorization-time GeoX `a4bf6bfaa4311dacd3642d289dca3917543e0309`. Review-time live overlay found:

- MIP `main`: `976d3a1daeae9c52c8772e5112574f698951a57c`, task `MIP_P2_ROADMAP_AND_COORDINATION_RECONCILIATION_AFTER_GEOX_SUPERSESSION_001`, `authorized`;
- GeoX `main`: `d17bb81c9dbc67f773fd71068c26b14c92989f42`, task `GEOX_EXECUTION_BRANCH_BINDING_001`, `authorized`.

These are non-overlapping owner-repository governance tasks. Refresh the live-overlay evidence in the corrected publication without modifying either sibling.

## Required correction and validation

1. Correct lifecycle metadata and retain the rejected head in execution state.
2. Expand all seven focused tests to enforce every behavior listed in `ACTIVE_TASK.md`.
3. Refresh live MIP and GeoX evidence and confirm no ownership overlap.
4. Rerun JSON, Markdown/current-state, authoring-boundary, changed-path, and `git diff --check` verification.
5. Rerun the seven focused tests, Ruff, and mypy.
6. Rerun Docker-backed `make validate` and record exact test and warning counts/disposition.
7. Freeze the corrected tree and publish a new exact-tree receipt with the full required receipt fields.
8. Push the exact corrected branch head and stop at coherent `ready_for_review`. Do not create a PR or merge.

Any required validation failure must produce Git-durable `blocked` with exact diagnostics and a live resolution condition.

## Scope and authority

No MMM analytical, model, calibration, simulation, optimization, contract, fixture, schema, numerical-truth, runtime, sibling, or capability authority may change. Historical PR #19 remains explicitly nonconforming and unapproved. `.codex/` and `docs/tasks/` remain local-only and uncommitted.
