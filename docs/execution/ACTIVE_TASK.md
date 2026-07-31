# Active Task

**Status:** authorized reconciliation
**Owner:** MMM repository governance
**Last updated:** 2026-07-30
**Last verified:** 2026-07-30
**Verified against:** MMM `main` / `ad55fef6799a8bd717108781ad44fc88fa116df7`
**Update trigger:** execution-state transition, review decision, or task closure.

## Identity

- **Task ID:** `MMM_REPO_NATIVE_EXECUTION_HANDOFF_V2_RECONCILIATION_001`
- **Base branch/SHA:** `main` / `ad55fef6799a8bd717108781ad44fc88fa116df7`
- **Feature branch:** `feat/mmm-repo-native-execution-handoff-v2-reconciliation-001`
- **Execution mode:** `branch_and_fast_forward`
- **Canonical MIP V2 pin:** `Phani-Pavuluri/marketing_intelligence_platform@38f88467f55d5bc4cc64e5a58b0f08f1639a40d0`
- **Recovery target:** `MMM_REPO_NATIVE_EXECUTION_HANDOFF_ADOPTION_001`
- **External PR:** `#19`
- **External branch head:** `ea16ab7e7b1089f5de479eeffb236fad2767edf1`
- **External merge commit:** `ad55fef6799a8bd717108781ad44fc88fa116df7`
- **Capability authorizations changed:** `false`

## Why reconciliation is required

PR #19 placed the V1 workflow implementation on `main` while the committed MMM
state remained `ready_for_review`, with `merge_authorized: false`, no reviewed
head, and no approval commit. No conforming exact-head approval record is present.
The merge used a GitHub merge commit rather than the canonical V2 fast-forward
path. MMM also remains pinned to obsolete MIP commit `5eebba6` and retains the
legacy `approved_for_merge` lifecycle.

Preserve those facts. Do not retroactively describe PR #19 as an approved or
conforming merge, and do not rewrite or revert Git history merely to make it look
conforming.

## Objective

Reconcile the externally merged V1 adoption, upgrade MMM to the final MIP V2
repository-native execution workflow, and publish an auditable review head that
can later be approved and fast-forward merged. The resulting closure will make
MMM ready for normal repository-native technical tasks.

This is workflow governance only. No MMM model, contract, parser, fixture,
numerical output, or capability authority is owned or authorized.

## Owned files

Execution may modify only:

- `AGENTS.md`
- `docs/execution/ACTIVE_TASK.md`
- `docs/execution/EXECUTION_STATE.json`
- `docs/execution/LATEST_COMPLETION_REPORT.md`
- `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
- `tests/test_repo_native_execution_handoff.py`

Do not modify MMM product code, contracts, fixtures, roadmaps, validation
registries, MIP, or GeoX. If the existing focused test cannot represent the V2
workflow within these owned files, stop and report the conflict rather than
expanding scope.

## Task-authoring boundary

The pre-authoring base is `ad55fef6799a8bd717108781ad44fc88fa116df7`.
Verify `base_sha..authorization_head_sha` changes only the three stable execution
files. Because a commit cannot contain its own SHA, one final state-only commit
may be present immediately after `authorization_head_sha` solely to record that
boundary. No other path or commit is permitted between the authorization head
and synchronized `main`.

Create the feature branch from the exact synchronized post-authoring `main`, not
from stale local state or the pre-authoring base.

## Prerequisites

1. Complete the mandatory bootstrap from the pinned MIP V2 standard before task
   discovery: classify the worktree, permit untracked content only below
   `.codex/` and `docs/tasks/`, run `git fetch --prune origin`, hydrate missing
   history, switch to `main`, pull with `--ff-only`, and prove local
   `main == origin/main`.
2. Verify the canonical MIP V2 pin
   `38f88467f55d5bc4cc64e5a58b0f08f1639a40d0` exists on MIP `main`, contains
   `docs/execution/TASK_EXECUTION_STANDARD.md`, and records the closed MIP V2
   recovery.
3. Verify PR #19 lineage:
   - base `ef63068c37041bdde55373cc08ef19333aa0fb5e`;
   - original implementation `f0b0ae35619739a4ff3d95f2cf7c93bf7ec523a0`;
   - external branch head `ea16ab7e7b1089f5de479eeffb236fad2767edf1`;
   - merge commit `ad55fef6799a8bd717108781ad44fc88fa116df7`;
   - no conforming approval record may be invented.
4. Verify the external head descends from the V1 task-authoring checkpoint and
   that its changed paths are limited to the original workflow-adoption files.
5. Verify the stale remote branch
   `feat/mmm-repo-native-execution-handoff-adoption-001` still exists or record
   accurately if it was already removed.
6. Verify GeoX is not modified and remains paused pending its separate V2
   adoption task.

## Required implementation

1. Create and switch to
   `feat/mmm-repo-native-execution-handoff-v2-reconciliation-001` from exact
   synchronized post-authoring `main`.
2. Upgrade `AGENTS.md` to require the full MIP V2 bootstrap before task
   discovery: worktree classification, fetch/prune, history hydration,
   `git switch main`, `git pull --ff-only origin main`, and exact
   `main == origin/main` verification.
3. Replace the legacy merge lifecycle with MIP V2 semantics:
   - no persisted `approved_for_merge` state;
   - external user approval binds the exact remote feature-branch head;
   - no pre-merge approval metadata commit;
   - persisted `merge_authorized` remains false until closure;
   - merge uses `git merge --ff-only` with no PR, squash, rebase, merge commit,
     or force update;
   - exactly one post-merge closure commit records approval, validation,
     lineage, authority, synchronization, and cleanup.
4. Update `REPOSITORY_CONTEXT_INDEX.md` and its Fresh Chat Bootstrap to pin MIP
   V2 closure `38f88467f55d5bc4cc64e5a58b0f08f1639a40d0`, require synchronized Git before
   reading task state, and preserve MMM producer/context references.
5. Upgrade the execution state and focused test to
   `mmm_repo_execution_state_v2`. Use only V2 statuses: `idle`, `proposed`,
   `authorized`, `in_progress`, `blocked`, `ready_for_review`,
   `changes_requested`, `merged`, and `superseded`.
6. Update the focused test to enforce the canonical MIP pin, mandatory bootstrap
   commands, permitted local-only paths, exact-head external approval, Docker
   validation, no pre-merge approval commit, fast-forward merge, and exactly one
   closure commit.
7. Preserve the external PR #19 record separately from the conforming V2
   reconciliation. Do not claim the earlier merge was approved.
8. Run focused execution-handoff and relevant documentation/governance tests,
   JSON and Markdown/path checks, Ruff, mypy, `git diff --check`, and
   Docker-backed `make validate`.
9. If a prerequisite or validation fails, update only owned files to an accurate
   `blocked` state, commit and push the branch, and stop.
10. If all gates pass, publish a `ready_for_review` branch state with:
    - `task_execution_authorized: true`;
    - `merge_authorized: false`;
    - `reviewed_head_sha: null`;
    - `approval_commit_sha: null`;
    - populated `implementation_commit_sha`;
    - `capability_authorizations_changed: false`;
    - no blockers.
11. Commit and push the exact remote branch head, verify local/remote equality,
    and stop for ChatGPT review. Do not create a pull request, merge, or delete
    branches during execution.

## Completion report requirements

The report must include:

- V1 task-authoring, implementation, external-head, PR #19, and merge lineage;
- explicit absence of a conforming exact-head approval record;
- MIP V2 canonical pin and task-authoring boundary;
- exact changed paths and MMM/GeoX scope confirmation;
- focused and full validation, Docker, Ruff, mypy, and diff evidence;
- recovery implementation commit and exact published review head;
- limitations, deferred work, local-only paths, and authority impact;
- `capability_authorizations_changed: false`.

## Later approved merge and closure

Only after the user approves the exact remote V2 reconciliation head may Codex
run `Merge the approved active task`. The merge session must re-fetch and verify
the approved head, rerun required validation, fast-forward merge without a PR,
push and verify `main`, delete the V2 reconciliation branch and the stale V1
adoption branch where present, then create exactly one post-merge closure commit.

That closure becomes the canonical MMM execution-workflow pin. It must preserve
PR #19 as a separate nonconforming historical merge and must not authorize any
MMM analytical capability.

## Prohibited scope and authority

Do not change or authorize model fitting, calibration behavior, simulation or
supported-range semantics, optimization or candidate generation,
recommendations, Bayesian production, automatic refit/model promotion, public
export schemas, numerical truth, live MIP/GeoX integration, real data,
persistence, pilot, production, or package-side agents.
