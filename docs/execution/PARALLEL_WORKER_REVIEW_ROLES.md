# MMM Parallel Worker / Independent Review Roles

**Status:** adopted guidance for `MMM_WORKER_REVIEW_ROLE_ADOPTION_RECOVERY_001`.
This document assigns *who does what*. It grants no execution, merge, PR,
analytical, sibling, or capability authority beyond what Git-authored task state
already provides. All lifecycle authority remains exclusively in
`docs/execution/EXECUTION_STATE.json` under the task-owned controller.

## 1. Role definitions

| Role | Agent | Owns |
|------|-------|------|
| Implementation worker | Muse Spark (OpenCode session) | Implementation approach and execution **within** MMM-authorized tasks: worktree work, validation runs, evidence commits, publication commits. |
| Independent reviewer | Codex | Exact-head review of completion heads, evidence, required files, and goal alignment. Discusses findings; does **not** prescribe execution prompts or take over product implementation. |

The reviewer must not have authored the implementation under review.
One implementation writer per MMM repository/task worktree.

## 2. Pre-invocation verification (before any worker invocation)

Verify and record all of the following; a missing item blocks invocation:

1. Local/remote Git identity (`origin` URL matches `Phani-Pavuluri/MMM`).
2. Synchronized state: local `main` equals `origin/main`.
3. Exact task/branch ancestry: task ID, authorization head, declared branch.
4. Owned paths for the task and that no path outside them will be touched.
5. All required dependencies are present and their pins verified.
6. OpenCode session identity: provider, model, working directory, variant.
7. No active competing writer on the same repository/task worktree.

Other repositories work independently. Shared contracts, ownership, or
prerequisite ambiguities must be resolved before dependent implementation.
Producer completion requires merged evidence plus consumer verification where
applicable. Silence, old pins, or plan text do not prove readiness or grant
authority.

## 3. Reserved decisions (user only)

The owner retains: concepts and goals, roadmap direction, new capabilities,
material architecture and contract meaning, authority decisions,
production and data decisions, and resolution of substantive disagreements.
Agent agreement cannot expand scope.

## 4. Advisory discussion rules

- Discussions are advisory: exact-head findings with disposition.
- No automatic abort, pause, release, merge-approval, or task-approval controls.
- Corrections require remaining authorized correction budget or a new
  owner-authorized successor task.
- Reviewer discusses findings; it does not issue execution prompts and does
  not assume the implementation role.

## 5. Owner status reporting

Owner updates use plain language: completed / ongoing / planned / waiting
work, goal impact, issues, and decisions. Distinguish verified outcomes from
agent claims and plans. Never promise background reports before verification.

## 6. Non-activation (explicitly not implemented or authorized here)

- No runtime worker dispatch, automatic worker startup, or shared-service restart.
- No session binding/occupancy enforcement (a future implementation must reject
  duplicates and ambiguous restarts; deferred, not built here).
- No delegated or automatic merge/task approval; external user exact-head
  approval remains mandatory for every merge.
- No plan approval/materialization, advisory integration, or shared monitoring.
- No analytical, model, contract, sibling, or capability change of any kind.

## 7. Completion matrix

| Requirement (task contract) | Satisfied by |
|---|---|
| Muse-Spark owns implementation approach/execution | §1 worker row; AGENTS.md role note |
| Codex independently reviews heads/evidence/files/goals | §1 reviewer row; §4 discussion rules |
| Reviewer did not author implementation | §1 non-authorship rule |
| Existing lifecycle rules intact (authoring, precedence, bootstrap, ancestry, one task, one correction, validation, receipts, exact-head approval, single closure) | Unchanged AGENTS.md sections; no lifecycle edit in this task |
| Canonical launcher stays invocation-only | Launcher block untouched; role note outside it |
| One writer per worktree + pre-invocation verification | §2 checklist |
| Session observation dated, reverified before dispatch | §2 item 6; task contract session note |
| User reserved decisions; no scope expansion by agreement | §3 |
| Advisory discussion, no auto controls; corrections need budget/successor | §4 |
| Plain owner reporting; no pre-verification promises | §5 |
| Deferred items not activated | §6 |
