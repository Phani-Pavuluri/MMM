# MMM Codex Execution Rules

Durable task instructions live in Git. Prompts cannot repair, expand, or
reinterpret a missing or incomplete active task.

## Mandatory bootstrap

Before task discovery or implementation:

1. Inspect `git status --porcelain=v1 --untracked-files=all`; fail on unrelated
   tracked changes or untracked paths outside `.codex/` and `docs/tasks/`.
2. Run `git fetch --prune origin`; hydrate shallow or missing required history.
3. Run `git switch main` then `git pull --ff-only origin main`.
4. Prove `git rev-parse main` equals `git rev-parse origin/main`.
5. Only then read `EXECUTION_STATE.json`, `ACTIVE_TASK.md`, the context index,
   the applicable MMM standards, and pinned MIP coordination evidence.

Missing synchronization, evidence, authority, Docker, dependencies, or safe
authorized write target is fail-closed. A successful orientation is non-terminal:
continue through implementation, validation, publication, and push to one
Git-durable `ready_for_review` or `blocked` outcome. Stop externally only when no safe authorized branch exists.

## Authority hierarchy and operational launcher

Synchronized `main/docs/execution/EXECUTION_STATE.json` owns repository
identity, task identity, authorization provenance, authorization head, and the
declared feature branch. Codex resolves that branch from synchronized main,
fetches the exact remote branch, and verifies repository identity, task ID,
declared branch name, and authorization-head ancestry. The verified remote feature branch state owns `authorized`, `in_progress`, `blocked`,
`changes_requested`, and `ready_for_review`, plus blockers, corrections,
implementation evidence, and completion reporting. `ACTIVE_TASK.md` owns the
objective, behavior, prerequisites, owned/prohibited paths, acceptance,
validation, and stop conditions. `LATEST_COMPLETION_REPORT.md` is evidence only;
it cannot authorize execution, correction, merge, sibling, analytical,
or capability work.

Codex never chooses whichever file appears newer. Conflicting repository, task,
branch, ancestry, implementation, lifecycle, or authority evidence produces no
implementation continuation and requires a Git-durable blocked state on the
safe authorized branch with exact mismatch, attempted evidence, affected
validation categories, and a live resolution condition. Chat, cached prompts,
and completion prose cannot resolve conflicts.

Compact operational launchers may contain only synchronization, required Git
reads, task/branch resolution, remote verification/resumption, continuation
through validation/publication/push, non-terminal progress, durable terminal
outcomes, prohibited Git operations, and an externally approved exact merge
SHA. They may not define, copy, repair, override, or reinterpret task meaning,
paths, prerequisites, tests/counts, sibling state, or authority.

The canonical normal execution launcher is the operational launcher below. A
one-line command is optional shorthand only when the executor already follows
these committed rules; it is not exclusive and carries no task meaning.
Correction uses the same launcher and obtains rejected SHAs and defects from
Git. Merge uses the same launcher and adds only `Approved exact remote head:
<FULL_SHA>`.

Canonical launcher:

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

Correction uses the same launcher and reads rejected SHAs/fixes from the
branch's Git-authored state. Merge adds only `Approved exact remote head:
<FULL_SHA>`. Successful or blocked handoff requires only repository, feature
branch, and exact remote head SHA; chat output is diagnostic context only.

## Execute the active task

Require one independently mergeable outcome; exact behavior/boundaries;
surface-appropriate resolved decisions; inputs, outputs, invariants, failures;
compatibility or migration policy (or `not_applicable`); named evidence; owned
and prohibited paths; risk tier; validation; deferred successors; and
`unresolved execution-blocking design questions: none`. Split instead of
choosing among material meanings or silently widening after one correction.

Verify main authorization provenance and the declared exact remote branch
identity, ancestry, task-owned state, and lifecycle. Main authorizes; the exact
remote branch supplies resumed lifecycle state. Before review freeze the
task-owned tree, run every required gate on that exact tree, write the report,
and commit a durable exact-tree receipt. No task-owned file may change after the
receipt; any change requires a new validated publication head. Publish only
`ready_for_review` with execution true, correction/merge/PR false,
`pr_creation_authorized: false`, null
reviewed/approval SHAs, and unchanged capability authority.

Use Tier 1/2/3 focused evidence from
`docs/program/LEAN_REPOSITORY_DELIVERY_STANDARD.md`, but never waive MMM's
repository-authored full Docker gate: run `make validate` whenever the task,
Tier 3, an analytical/public/package surface, or existing MMM rules require it.
Do not start duplicate validation containers. Mark inapplicable categories
`not_required`; a required category that cannot run is `blocked`.

Before work affecting MIP or GeoX, apply
`docs/execution/TASK_EXECUTION_STANDARD.md` and the pinned MIP coordination
protocol: verify live sibling mains/execution files, mutable branch evidence,
workstream/owner/dependency/blocker overlap, and consumer verification. A
feature branch never proves a merged dependency; coordination metadata does not
authorize sibling work.

## Merge and closure

External user approval binds the exact remote feature-branch SHA. Re-bootstrap,
verify unchanged authorization ancestry and exact head, rerun the exact-tree
gate, use only `git merge --ff-only`, rerun the gate after fast-forward, push
and verify main equality, and delete only the task branches. Never create a PR,
squash, rebase, merge commit, force update, or pre-merge approval commit.

After cleanup and branch cleanup, create exactly one closure commit limited to the stable
task/state/report files. Record approval, reviewed/implementation lineage,
validation categories, synchronization, cleanup, deferred work, and authority.
The closure sets execution and merge authority false. Historical PR #19 remains
nonconforming and never acquires retroactive approval.
