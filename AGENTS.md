# MMM Codex Execution Rules

## Mandatory session bootstrap

Before task discovery or implementation:

1. Inspect `git status --porcelain=v1 --untracked-files=all`. Fail closed on
   unrelated tracked changes or unexpected untracked paths. During an authorized
   resumption, every tracked change must be task-owned and explained.
   Untracked content is permitted only below `.codex/` and `docs/tasks/`; never
   stage or commit it.
2. Run `git fetch --prune origin`. If the clone is shallow, run
   `git fetch --unshallow origin`; if a required ancestor is still absent,
   fetch enough additional history and verify that commit explicitly.
3. Run `git switch main` and `git pull --ff-only origin main`.
4. Verify `git rev-parse main` exactly equals `git rev-parse origin/main`.
5. Only then read, in order:
   - `docs/execution/EXECUTION_STATE.json`
   - `docs/execution/ACTIVE_TASK.md`
   - `docs/execution/REPOSITORY_CONTEXT_INDEX.md`
   - relevant MMM evidence and the pinned MIP execution standard/program files.

Stop rather than guess if synchronization, history hydration, execution files,
authorization, prerequisites, or repository state cannot be verified. Chats
and pasted summaries are never authoritative repository state.

## Execute the active task

Verify the authorized task, its task-authoring boundary, prerequisites, owned
files, and exact feature branch. Run focused and full validation, including
Docker-backed `make validate`; write `docs/execution/LATEST_COMPLETION_REPORT.md`;
update `docs/execution/EXECUTION_STATE.json` to `ready_for_review` with
`merge_authorized: false`; commit and publish the exact remote feature-branch
head; then stop without a pull request, merge, or branch deletion.

## Merge the externally approved head

External user approval must identify the exact remote feature-branch head SHA.
Re-run the mandatory bootstrap, re-fetch the feature branch, verify its head
still equals the approved SHA, verify `main` has not moved beyond the
authorization boundary, and rerun required validation. Use
`git merge --ff-only`; never create a pull request, squash, rebase, merge
commit, force update, or pre-merge approval-metadata commit.

After the fast-forwarded implementation is pushed and branch cleanup is
observed, update only the stable task/state/report files in exactly one
post-merge closure commit. Record approval provenance, reviewed head,
validation, authority impact, resulting main lineage, synchronization, and
cleanup results. Persisted `merge_authorized` remains false until that closure.
No capability authority follows from task execution metadata.
