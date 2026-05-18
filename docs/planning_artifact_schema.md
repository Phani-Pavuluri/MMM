# Planning artifact schema (decision bundles)

Fields below are written by **`mmm decide simulate`** / **`mmm decide optimize-budget`** into CLI JSON and the persisted **`decision_bundle`**.

## `planning_assumptions`

| Field | Type | Description |
|-------|------|-------------|
| `controls_assumption` | string | `observed` \| `overlay` \| `frozen_scenario` — how non-media columns enter μ |
| `media_assumption` | string | `constant` \| `geo_channel` \| `piecewise_path` \| `optimized` |
| `world_assumption` | string | `historical_panel` \| `explicit_scenario` |
| `planning_disclosures` | string[] | Human-readable contract reminders (also printed on CLI) |

## `scenario_lineage`

Present when a typed **`PlanningScenario`** is supplied, or (for optimize without `--scenario`) as an explicit “no overlay” stub.

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | string \| null | Scenario identifier |
| `scenario_version` | string \| null | Optional version tag |
| `scenario_hash` | string \| null | SHA-256 of canonical scenario JSON |
| `scenario_source_path` | string \| null | Path when loaded from YAML |
| `non_media_overlay_supplied` | bool | `true` when `--scenario` / scenario dict was provided |
| `non_media_overlay_applied` | bool | `true` when at least one plan overlay row was applied |
| `baseline_overlay_spec_sha256` | string \| null | Hash of baseline overlay rows |
| `plan_overlay_spec_sha256` | string \| null | Hash of plan overlay rows |
| `control_overlay_spec_sha256` | string \| null | Hash of legacy single overlay |
| `control_overlay_summary` | object | Column lists and override counts |
| `overlay_row_count_baseline` | int | Row count |
| `overlay_row_count_plan` | int | Row count |
| `spend_scenario_summary` | object | Flags / hashes for media spend blobs |
| `baseline_control_overlay_spec` | list | **Optional** — full canonical rows when `store_full_control_overlays_in_artifacts: true` |
| `plan_control_overlay_spec` | list | **Optional** — same |
| `note` | string | Optimize-without-scenario stub only |

## `control_scenario_policy`

| Field | Type | Description |
|-------|------|-------------|
| `severity` | string | `info` \| `warning` \| `block` |
| `messages` | string[] | Policy text (`block` raises before simulation) |
| `sensitive_columns_matched` | string[] | Configured / heuristic matches |
| `controls_assumption` | string | Echo of active controls mode |

## `scenario_validation_warnings`

List of strings on the simulation payload when panel/scenario validation finds non-fatal issues. Surfaced on CLI as **SCENARIO VALIDATION WARNING**.

## Prod validation rules

- **`planning_assumptions`** required on decision bundles.
- When `world_assumption=explicit_scenario`, **`scenario_lineage`** must include **`scenario_id`** and **`scenario_hash`**.
- `unsupported_questions` may include a disclaimer when controls are observed-only.

See also: [decision_runbook.md](decision_runbook.md) §2e, [config_yaml.md](config_yaml.md) (`extensions.planning_policy`).
