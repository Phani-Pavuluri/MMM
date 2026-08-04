# MMM Repository Context Index

**Status:** active navigation index
**Owner:** MMM repository governance
**Last updated:** 2026-08-03
**Canonical execution standard:**
`Phani-Pavuluri/marketing_intelligence_platform@369805d923454a51ce98845cea29bdb1ee3c3895`

## Fresh Chat Bootstrap

Use this prompt in a fresh ChatGPT chat:

> Use connected GitHub as the source of truth. First classify the MMM worktree:
> inspect status including untracked files, permit local-only content only below
> `.codex/` and `docs/tasks/`, fetch and prune `origin`, hydrate required
> history, switch to `main`, pull with `--ff-only`, and prove local `main` equals
> `origin/main`. Only then read MMM `EXECUTION_STATE.json`, `ACTIVE_TASK.md`,
> `REPOSITORY_CONTEXT_INDEX.md`, and the pinned MIP V2 execution standard and
> MIP execution/coordination standards at
> `Phani-Pavuluri/marketing_intelligence_platform@369805d923454a51ce98845cea29bdb1ee3c3895`.
> Summarize current state, active task, latest completion, blockers,
> dependencies, authority boundaries, and next eligible work. Do not modify
> files or authorize work unless explicitly requested.

## Stable execution handoff

- `docs/execution/EXECUTION_STATE.json`
- `docs/execution/ACTIVE_TASK.md`
- `docs/execution/LATEST_COMPLETION_REPORT.md`
- this index
- `docs/execution/TASK_EXECUTION_STANDARD.md`
- `docs/program/LEAN_REPOSITORY_DELIVERY_STANDARD.md`
- pinned MIP V2 standard:
  `docs/execution/TASK_EXECUTION_STANDARD.md` at the canonical commit above

## Canonical MIP protocol navigation

- `docs/execution/TASK_EXECUTION_STANDARD.md`
- `docs/program/LEAN_REPOSITORY_DELIVERY_STANDARD.md`
- `docs/program/CROSS_REPOSITORY_COORDINATION_PROTOCOL.md`
- `docs/program/CROSS_REPOSITORY_COORDINATION_STATE.json`
- `docs/program/CROSS_REPOSITORY_COORDINATION_HISTORY.md`

## MMM technical producer evidence

- Public scenario comparison: `mmm/contracts/public_simulation.py`,
  `MMMPublicSimulationExport`, and
  `tests/fixtures/mip_export/simulation_v1/`.
- Supported-range evidence: `mmm/contracts/supported_range.py` and
  `tests/contracts/test_mmm_supported_range_evidence.py`.
- Calibration compatibility: `mmm/contracts/calibration_compatibility.py`,
  `MMMCalibrationCompatibilityResult`, strict parser, fixtures at
  `tests/fixtures/mip_export/calibration_compatibility_v1/`, and the policy
  registry at
  `docs/05_validation/archives/MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001_registry.json`.
- Contract inventory and producer boundary:
  `docs/05_validation/mmm_to_mip_export_contract_inventory.md`.

## MMM program context

- Current roadmap: `docs/05_validation/platform_roadmap.md`.
- Validation/platform evidence: `docs/05_validation/validation_registry.md`.
- Open investigations: `docs/06_investigations/investigation_index.md`.
- Deferred package-side agents:
  `docs/05_validation/mmm_package_side_agents_roadmap.md`.

## Connected repositories and checkpoints

- GeoX (`Phani-Pavuluri/panel_exp`) is the experiment-readout producer.
- MIP (`Phani-Pavuluri/marketing_intelligence_platform`) is consumer and
  orchestrator.
- Verify exact checkpoints from connected GitHub repositories before dependent
  work; repository completion does not imply authority or production readiness.
- The active task state records the required MMM checkpoint and canonical MIP
  V2 commit. Never infer a replacement checkpoint from local files or chat
  text.
