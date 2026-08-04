# MMM Task Execution Standard

## Authority and launcher boundary

Synchronized main owns repository/task identity, authorization provenance,
authorization head, and declared feature branch. Resolve and verify the exact
remote branch, including repository identity, task identity, branch name, and
authorization-head ancestry. The remote feature branch owns mutable lifecycle
state, blockers, corrections, implementation evidence, and completion
reporting. `ACTIVE_TASK.md` is the implementation contract; the completion report is evidence only and cannot authorize work.

Conflicting identity, ancestry, implementation, lifecycle, or authority evidence
fails closed with an exact Git-durable blocked state, attempted evidence,
validation-category status, and live resolution condition. Prompts and cached
prose cannot repair the conflict. Compact launchers may carry operational Git
controls and the approved merge SHA, but may not duplicate task meaning,
paths, prerequisites, tests, counts, sibling lifecycle, or authority.

The canonical normal execution launcher is the multi-line operational launcher
recorded in `AGENTS.md`; a one-line command is optional shorthand only and is
not an exclusive format. Correction uses the same launcher and reads rejected
SHAs and defects from Git. Merge adds only an externally approved exact SHA.

Normal handoff is only repository, feature branch, and exact remote head SHA.
Progress is non-terminal; only remotely published `ready_for_review` or
`blocked` is terminal.

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
