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

The repository execution state is a single-source v3 document. Generated
execution blocks in human-facing views are synchronized by
`python -m mmm.execution.taskctl`; manual lifecycle edits are rejected after
the one-time v2 migration.

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

## Exhausted review rejection and successor rotation

An exact `ready_for_review` feature head may transition to the existing terminal
`blocked` state only when its correction budget is exhausted. The transition
must supply a valid rejected-review head and rejected-implementation SHA, the
rejected-review SHA must equal the current feature HEAD, and it must include a
non-empty blocker and live resolution condition. The transition closes task,
correction, merge, and PR authority while preserving the implementation and
correction lineage. It is not a generic escape hatch while a correction cycle
remains, and the rejected branch is not mergeable or certified by this state.

After that terminal blocked state is durably published, main may authorize a
new independent task without merging the blocked branch only through a new
authoring/authorization action. The successor must record the predecessor task
ID and exact blocked remote head, and the predecessor's blocker and live
resolution condition must require that successor. No analytical artifact from
the blocked branch may be copied into or merged with main; the blocked branch
remains historical evidence only. This rotation does not execute the successor
or change the blocked task's authority.
