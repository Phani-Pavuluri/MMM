# Active Task

**Status:** proposed — dependency not yet merged
**Owner:** MMM repository governance
**Last updated:** 2026-08-03
**Last verified:** 2026-08-03

## Identity

- **Task ID:** `MMM_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_ADOPTION_001`
- **Repository:** `Phani-Pavuluri/MMM`
- **Pre-authoring base:** `ac546548784385baab67d7c935e5a4fcdfc9e1af`
- **Intended feature branch:** `docs/mmm-git-authoritative-thin-launcher-standard-adoption-001`
- **Feature branch created:** `false`
- **Execution mode after authorization:** `branch_and_fast_forward`
- **Risk tier:** Tier 1 repository-execution governance with MMM's Docker-backed full validation retained as mandatory
- **Coordination workstream:** `WS-MMM-THIN-LAUNCHER-ADOPTION-001`
- **Capability owner:** MMM repository governance
- **Capability authorizations changed:** `false`

## Current decision

This task is **proposed, not authorized**. Do not create its feature branch, run
Codex implementation, modify task-owned implementation paths, publish review
evidence, create a pull request, or merge.

The upstream MIP standard is currently available only on an unmerged feature
branch:

- MIP task: `MIP_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_001`
- MIP `main` observed at proposal: `9bed0f30879e68473a37b0e65d449ea0b6a6e3f3`
- MIP feature branch: `docs/mip-git-authoritative-thin-launcher-standard-001`
- MIP candidate review head: `e390f1b47f8a7c5dfaa7a05613c2c4de73e4a548`
- MIP candidate implementation SHA: `dde6969b1192b97aea519c9589d27186f19b6db2`
- MIP candidate state: `ready_for_review`

A feature branch never satisfies this task's dependency. The candidate SHAs are
orientation evidence only and must not be reused as an eventual merged pin.

## Dependency and live resolution condition

### Dependency ID

`DEP-MMM-THIN-LAUNCHER-MIP-STANDARD-MERGED-001`

### Current blocker

`BLOCK-MMM-THIN-LAUNCHER-UPSTREAM-NOT-MERGED-001`

### Resolution condition

Before this MMM task may be authorized, connected GitHub must prove all of the
following from the then-current MIP `main` and stable execution files:

1. `MIP_GIT_AUTHORITATIVE_THIN_LAUNCHER_STANDARD_001` is `merged` on MIP
   `main` through an externally approved exact review head.
2. MIP has one post-merge closure record with the merged implementation lineage,
   validation disposition, synchronization, branch cleanup, limitations,
   validation debt, sibling impact, consumer-verification disposition, and
   unchanged capability authority.
3. The exact merged MIP `main` SHA and exact canonical standard paths are read
   directly from GitHub.
4. The merged standard remains compatible with MMM's repository-owned rule that
   Docker-backed `make validate` is mandatory for this adoption task.
5. No live MMM task, branch, pull request, or workstream overlaps this execution
   governance surface.

After those conditions are verified, update this task on synchronized MMM
`main` with the exact merged MIP pin and any material semantic differences. Then
create one immediate state-only authorization commit, create the declared branch
from that exact authorization boundary, and only then issue the hybrid execution
launcher.

## Primary independently mergeable outcome after authorization

Adopt the exact merged MIP Git-authoritative thin-launcher standard as
MMM-owned repository execution governance.

Git remains the sole durable source for task identity, authorization, scope,
observable behavior, owned paths, implementation decisions, dependencies,
blockers, corrections, validation, authority, and stop conditions. A launcher
may carry only stable operational controls needed to make execution reliable:
repository location, synchronization and repository reads, resolution and
resumption of the Git-declared branch, continuation to durable publication,
non-terminal progress semantics, permitted terminal outcomes, prohibited
operations, and the externally approved exact SHA for merge.

This outcome changes no MMM analytical, model-fitting, calibration, simulation,
optimization, contract, adapter, fixture, numerical-truth, runtime, package,
release, sibling, product, or capability behavior.

## Why this task cannot be split further

The allowed launcher boundary, execution/correction/merge launcher patterns,
main-versus-feature-branch authority, non-terminal progress behavior, durable
terminal outcomes, exact-head merge discipline, focused semantic tests, and MMM
validation preservation form one execution contract. Updating only prose or
only tests would leave contradictory repository behavior.

## Proposed observable behavior

The exact merged MIP standard is authoritative at authorization. Subject to that
final pin, MMM must preserve the following behavior.

### 1. Git-only durable task meaning

The launcher cannot define, repair, expand, override, or reinterpret:

- task ID, lifecycle state, authorization provenance, or feature branch;
- scope, observable behavior, owned/prohibited paths, or implementation choices;
- acceptance tests, validation commands/counts, dependencies, blockers,
  correction details, rejected heads, or sibling state; or
- review, merge, cleanup, release, analytical, or capability authority.

Missing, contradictory, stale, or incomplete Git-authored instructions remain a
fail-closed blocker.

### 2. Allowed thin-launcher content

Execution and correction launchers may contain only:

- the local MMM repository path;
- synchronization and required Git-authored reads;
- resolution and resumption of the exact branch declared by synchronized main;
- continuation through implementation, required validation, exact-tree
  publication, push, and remote-head verification;
- explicit non-terminal progress semantics;
- the only permitted durable terminal outcomes;
- prohibition on PR, merge, force operations, and capability changes; and
- for merge only, the externally approved exact remote review SHA.

They must not copy task IDs, branch names, non-approved SHAs, scope, paths,
tests, counts, dependencies, correction details, implementation guidance, or
sibling lifecycle state from chat.

### 3. Canonical execution launcher pattern

```text
Work in <local MMM repository path>.

Synchronize main from Git and read AGENTS.md and the repository execution files. Resolve authorization provenance and the exact feature branch from synchronized main, then fetch and resume that remote feature branch and read its current execution files.

Execute the active task through implementation, required validation, exact-tree publication, push, and remote-head verification.

Progress updates are non-terminal. Do not stop or return control merely to report orientation or progress. Stop only when the remote feature branch durably records ready_for_review or a genuine blocked state.

Do not create a pull request, merge, or change analytical or capability authority.
```

### 4. Canonical correction launcher pattern

```text
Work in <local MMM repository path>.

Synchronize main from Git and read AGENTS.md and the repository execution files. Resolve authorization provenance and the exact feature branch from synchronized main, then fetch and resume that remote feature branch and read its current execution files.

Execute the Git-authored changes_requested correction through the complete required validation, a new exact-tree publication, push, and remote-head verification.

Progress updates are non-terminal. Do not stop or return control merely to report orientation or progress. Stop only when the remote feature branch durably records a new ready_for_review or a genuine blocked state.

Do not create a pull request, merge, or change analytical or capability authority.
```

Rejected heads and correction details come from Git, not the launcher.

### 5. Canonical merge launcher pattern

```text
Work in <local MMM repository path>.

Synchronize main from Git and read AGENTS.md and the repository execution files. Execute the active task's merge and closure workflow.

Approved exact remote head: <FULL_SHA>

Revalidate the approved head, fast-forward merge only, validate after fast-forward, push main, perform task-branch cleanup, create exactly one closure commit, and verify local and remote main equality.

Do not create a pull request, squash, rebase, force-push, or create a merge commit.
```

Only the local path and approved exact review SHA are caller-supplied values.

### 6. Main and feature-branch authority

Synchronized `main` owns repository identity, task identity, authorization
provenance, and the declared branch. The verified remote feature branch owns
current lifecycle state, blockers, correction state, implementation evidence,
and completion reporting. Cached prompt values cannot substitute for either.

### 7. Progress and terminal outcomes

Orientation and progress messages are checkpoints, not terminal outcomes. After
a safe authorized branch is verified, execution continues without another user
prompt until the remote branch durably records:

- `ready_for_review` with an exact-tree receipt; or
- a genuine Git-durable `blocked` state with exact diagnostics, attempted
  evidence, validation-category dispositions, and a live resolution condition.

### 8. Preserve MMM validation and merge rules

The thin launcher does not weaken:

- Docker-backed `make validate` during implementation, exact-head review, and
  post-fast-forward validation for this task;
- exact-tree publication receipts;
- exact-head external approval;
- fast-forward-only merge;
- one closure commit limited to the three stable execution files;
- task-branch cleanup and local/remote equality verification; or
- the prohibition on PRs, squash, rebase, merge commits, force updates,
  pre-merge approval commits, and capability changes.

## Named acceptance evidence after authorization

Update `tests/test_repo_native_execution_handoff.py` with separate assertions
that prove:

1. Git remains the sole durable task authority and launchers cannot repair or
   duplicate task meaning.
2. Execution and correction launchers contain the allowed operational controls,
   make progress non-terminal, and require remote durable terminal outcomes.
3. The merge launcher permits only repository path and approved exact SHA as
   caller-supplied values while retaining MMM validation, fast-forward, closure,
   cleanup, and prohibited-operation rules.
4. Task IDs, branch names, non-approved SHAs, scope, paths, tests/counts,
   dependencies, correction details, implementation instructions, and sibling
   state are prohibited from launchers.
5. Existing bootstrap, exact-tree receipt, resumed-branch, risk-tier,
   cross-repository, historical PR #19, validation, merge/closure, and authority
   invariants continue to pass.

Equivalent names are acceptable only when these semantic groups remain separate
and explicit.

## Intended owned paths after authorization

Implementation may modify only:

1. `AGENTS.md`
2. `docs/execution/TASK_EXECUTION_STANDARD.md`
3. `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
4. `tests/test_repo_native_execution_handoff.py`
5. `docs/execution/ACTIVE_TASK.md`
6. `docs/execution/EXECUTION_STATE.json`
7. `docs/execution/LATEST_COMPLETION_REPORT.md`

Do not modify any other path.

## Prohibited scope

Do not modify or authorize:

- `mmm/**` or any analytical, model, calibration, simulation, optimization,
  contract, adapter, parser, fixture, schema, numerical-truth, runtime, package,
  release, or CI surface;
- MIP or GeoX repositories, branches, tasks, evidence, coordination files, or
  authority;
- live integration, real data, persistence, scheduled execution, candidate
  generation, recommendations, automatic refit/promotion, Bayesian production,
  pilot, production, or package-side agents; or
- retained unrelated historical branches.

No PR, squash, rebase, merge commit, force-push, or pre-merge approval commit is
permitted.

## Proposed MMM validation gate after authorization

Run on the frozen exact task-owned tree during implementation, exact-head review,
and after fast-forward:

- parse `docs/execution/EXECUTION_STATE.json` as JSON;
- verify Markdown/current-state and task/branch consistency;
- prove the task-authoring boundary and immediate state-only authorization
  boundary;
- verify exact changed paths against the owned-path list;
- run `git diff --check`;
- run `pytest -q tests/test_repo_native_execution_handoff.py` and record the
  exact count;
- run Ruff and configured mypy for the changed test;
- run Docker-backed `make validate` and record exact passed, failed, skipped,
  deselected, warning counts, and warning disposition;
- inspect the final exact-tree receipt fields; and
- prove local/remote exact branch-head equality after push.

A required category that cannot run produces Git-durable `blocked`; it is not
silently omitted or replaced with cached evidence. Do not start duplicate
validation containers.

## Publication and reporting requirements after authorization

A successful remote `ready_for_review` head must contain one implementation SHA
and one exact-tree receipt, empty blockers, task execution true, correction,
merge, and PR authority false, null reviewed/approval SHAs, and unchanged
analytical and capability authority.

The completion report and receipt must distinguish GitHub-observed evidence from
locally reported evidence and explicitly list:

- exact validation counts and warning disposition;
- blockers, limitations, and validation debt;
- affected repositories and sibling impact;
- consumer-verification disposition;
- newly eligible and deferred work;
- authority impact;
- exact worktree state and evidence source; and
- local/remote branch-head equality.

This requirement carries forward the minor reporting debt identified in the
closure of `MMM_REPOSITORY_EXECUTION_PROTOCOL_ADOPTION_001`; it does not reopen
or amend that completed task.

## Deferred successors

- GeoX-readout normalization and certified cross-repository fixtures remain
  separately proposed and blocked on exact merged GeoX producer evidence plus
  declared MMM consumer verification.
- Any technical MMM roadmap task remains separate and requires a fresh
  evidence-based selection and authorization.
- MIP and GeoX adoption, coordination refresh, product work, and capability
  changes remain owner-repository tasks.

## Authorization boundary to use only after dependency resolution

After verifying the merged MIP standard, update this proposed task and its
proposal report on synchronized MMM `main`. The final task-authoring commit is
the future `authorization_head_sha`. The immediate next commit may change only
`docs/execution/EXECUTION_STATE.json` to set `status: authorized`, record the
exact merged MIP pin and declared branch, and enable task execution. Create the
feature branch only from that synchronized state-only authorization head.

**Unresolved execution-blocking design questions:** exact upstream merged MIP
standard pin and final merged text are pending.
