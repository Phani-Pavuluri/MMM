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
Git-durable `ready_for_review` or `blocked` outcome.

## Invocation-only contract

The normal execution/correction invocation is exactly:
`Synchronize from Git and execute the active task.`

The merge invocation is exactly:
`Synchronize from Git and execute the active task's merge and closure workflow. Approved exact remote head: <SHA>.`

Use no prompt text as durable task instruction. If Git-authored instructions
are missing or contradictory, publish `blocked` on the safe authorized branch
with the exact diagnostic, attempted evidence, validation-category status, and
live resolution condition; stop externally only when no such branch exists.

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
`ready_for_review` with execution true, correction/merge/PR false, null
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

After cleanup, create exactly one closure commit limited to the stable
task/state/report files. Record approval, reviewed/implementation lineage,
validation categories, synchronization, cleanup, deferred work, and authority.
The closure sets execution and merge authority false. Historical PR #19 remains
nonconforming and never acquires retroactive approval.
