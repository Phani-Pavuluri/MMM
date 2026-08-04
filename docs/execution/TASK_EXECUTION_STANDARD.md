# MMM Task Execution Standard

## Exact-tree publication

Before review, freeze the task-owned tree and run the active risk-tier gate on that
tree, and make the final publication commit a receipt containing task ID,
implementation parent, `exact-commit-tree` scope, gate/result, focused count,
JSON/Markdown/current-state/task-boundary/changed-path/diff outcomes,
Docker `make validate` count/disposition, Ruff/mypy outcome, worktree state,
evidence source, and unchanged authority. Any post-receipt change needs a new
validated review head.

## Live-overlay coordination

For MIP/GeoX-affecting work, read the pinned MIP coordination protocol, state,
and history; verify every sibling live `origin/main` execution file; and read
exact remote feature-branch execution files when lifecycle is mutable. A stale shared snapshot
requires a live overlay, never historical rewriting. Stop on
duplicate ownership, overlapping implementation, stale unresolved evidence, or
unclear authority. Distinguish producer completion from required consumer
verification and report affected repositories, workstream/capability owner,
dependency/blocker transitions, evidence SHA/paths, consumer verification,
eligible work, validation debt, and authority impact.

## Review and closure

Approval names the exact remote review SHA. Reviewed trees keep merge and PR
authority false. Exact-head validation repeats before and after `git merge
--ff-only`; then push, verify main equality, clean task branches, and create one
stable-file closure commit. No PR, squash, rebase, merge commit, force update,
or pre-merge approval commit is valid.
