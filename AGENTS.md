# MMM Codex Execution Rules

Every Codex session must read, in order:

1. `docs/execution/EXECUTION_STATE.json`
2. `docs/execution/ACTIVE_TASK.md`
3. `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
4. relevant MMM contracts, roadmaps, validation evidence, and the pinned MIP
   execution standard/program files.

Stop rather than guess if an execution file is missing, stale, contradictory,
or unauthorized. `.codex/` and `docs/tasks/` are local-only and must never be
committed.

Fresh chats must begin with the **Fresh Chat Bootstrap** in
`docs/execution/REPOSITORY_CONTEXT_INDEX.md`; prior chat summaries are not an
authoritative substitute for repository evidence.

## Execute the active task

Verify authorization; checkpoint ancestry and local/remote `main`; agreement
between task and state; the exact feature branch; prerequisites; and owned-file
scope. Run focused and full validation, write
`docs/execution/LATEST_COMPLETION_REPORT.md`, and update
`docs/execution/EXECUTION_STATE.json` to `ready_for_review` with
`merge_authorized: false`. Commit and push, then stop without merging.

## Merge the approved active task

Verify `approved_for_merge`, `merge_authorized: true`, and exact reviewed-head
integrity. Merge fast-forward only, push, verify local/remote synchronization,
record closure metadata, and clean up local and remote feature branches. No PR
is required. Stop rather than guess; execution metadata never authorizes a
product capability.
