# Artifact schema

Reference for persisted and CLI JSON artifact shapes. Organized by artifact family.

| Family | Where produced | Primary doc |
|--------|----------------|-------------|
| **Decision bundles** | `mmm decide simulate` / `optimize-budget` | This page (planning fields below) |
| **Extension reports** | `mmm train` post-fit extensions | [../02_concepts/diagnostics.md](../02_concepts/diagnostics.md) |
| **Calibration matching** | Train + calibration config | [../02_concepts/calibration.md](../02_concepts/calibration.md) |
| **Planning scenarios** | Scenario YAML / dict | [../03_planning/planning_howto.md](../03_planning/planning_howto.md) |

**How to produce planning bundle fields:** [../03_planning/planning_howto.md](../03_planning/planning_howto.md).

Fields below are written by **`mmm decide simulate`** / **`mmm decide optimize-budget`** into CLI JSON and the persisted **`decision_bundle`**.

## Decision bundle: `planning_assumptions`

| Field | Type | Allowed values / notes |
|-------|------|------------------------|
| `controls_assumption` | string | **`observed`** \| **`overlay`** \| **`frozen_scenario`** |
| `media_assumption` | string | **`constant`** \| **`geo_channel`** \| **`piecewise_path`** \| **`optimized`** |
| `world_assumption` | string | **`historical_panel`** \| **`explicit_scenario`** \| **`multi_world`** (reserved; **rejected in prod** today) |
| `seasonality_assumption` | string | Free text path label (default `observed_panel`) |
| `promo_assumption` | string | Free text (default `observed_panel_unless_overlay`) |
| `macro_assumption` | string | Free text |
| `pricing_assumption` | string | Free text |
| `planning_disclosures` | string[] | Human-readable reminders (CLI stderr) |
| `controls_disclosure` | string | Short controls summary |

### Meaning of enum values

| `controls_assumption` | Meaning |
|-----------------------|---------|
| `observed` | Non-media columns use historical panel values per row |
| `overlay` | Sparse `control_overlay_*` overrides on matched `(geo, week)` rows |
| `frozen_scenario` | Fixed non-media world (typical on optimize with `--scenario`) |

| `media_assumption` | Meaning |
|--------------------|---------|
| `constant` | National constant `candidate_spend` |
| `geo_channel` | Per-geo channel spend dict |
| `piecewise_path` | Piecewise spend path on the panel |
| `optimized` | Media chosen by SLSQP (`decide optimize-budget` only) |

| `world_assumption` | Meaning |
|--------------------|---------|
| `historical_panel` | No typed explicit scenario world (controls still may use overlays) |
| `explicit_scenario` | `PlanningScenario` / lineage with `scenario_id` |
| `multi_world` | Reserved; not implemented — **prod bundles fail validation** |

## Prod semantic validation rules

Enforced by `validate_planning_assumptions_semantics` on **`artifact_tier=decision`** prod CLI bundles (fail closed → `SemanticContractError`).

## Training extension vs CLI decision bundle

| Artifact | Typical `artifact_tier` | Use |
|----------|-------------------------|-----|
| `extension_report` (train output) | research / diagnostic | Governance review, diagnostics, promotion inputs |
| Nested `extension_report.decision_bundle` | **research** | Not a substitute for `mmm decide` outputs |
| `mmm decide simulate\|optimize-budget --out` JSON | **decision** | Production budgeting and simulation contracts |

After training, `extension_report.artifact_tier_disclosure` repeats this split. **Do not** feed the nested research bundle into prod decide paths as if it were CLI decision-grade.

| Rule | Condition |
|------|-----------|
| Structure | `planning_assumptions` must be a dict with all three enum fields |
| Enum values | Each field must match an allowed literal exactly (typos fail) |
| `explicit_scenario` | Requires `scenario_lineage.scenario_id` + `scenario_lineage.scenario_hash` |
| `frozen_scenario` | Requires scenario lineage identity **or** `non_media_overlay_applied` / overlay SHA evidence |
| `overlay` | Requires overlay column summary or overlay SHA in `scenario_lineage` when lineage present |
| `optimized` media | Bundle must be optimize context (`optimizer_success` or optimize `simulation_contract.source`) |
| Optimize bundle | `media_assumption` must be `optimized` |
| `multi_world` | Rejected in prod (`multi_world_not_implemented_for_decision_bundles`) |

## Decision bundle: `scenario_lineage`

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

## Decision bundle: `data_fingerprint` / `panel_fingerprint`

On training and `mmm decide` paths, `data_fingerprint` and `panel_fingerprint` are identical dicts attached to decision bundles.

### Fingerprint v2 (current)

| Field | Description |
|-------|-------------|
| `fingerprint_version` | `fingerprint_v2` |
| `sha256_combined` | Canonical combined hash (panel key columns incl. controls, schema, model form, transforms, config schema version, `data_version_id`, resolved seeds) |
| `sha256_panel_keycols_sorted_csv` | Legacy panel hash (geo, week, target, channels, controls) |
| `sha256_schema_json` | Legacy schema hash |
| `n_rows` | Panel row count |
| `fingerprint_details.included_fields` | Auditable list of hashed inputs |
| `fingerprint_details.omitted_fields` | Non-hashed volatile metadata (`run_id`, timestamps, generated IDs) |

### Legacy bundles (pre-v2)

Older artifacts may omit `sha256_combined` and `fingerprint_version`. They remain valid for lineage when `sha256_panel_keycols_sorted_csv` (and usually `sha256_schema_json`) are present. Compare fingerprints in this order:

1. If both sides have `sha256_combined`, require equality on that field.
2. Else require equality on `sha256_panel_keycols_sorted_csv` (and schema hash if recorded).

New runs should always emit v2 fields. See [../documentation_truth_audit.md](../documentation_truth_audit.md#fingerprint-v2-migration-legacy-compatible).

## Decision bundle: `control_scenario_policy`

| Field | Type | Description |
|-------|------|-------------|
| `severity` | string | `info` \| `warning` \| `block` |
| `messages` | string[] | Policy text (`block` raises before simulation) |
| `sensitive_columns_matched` | string[] | Configured / heuristic matches |
| `controls_assumption` | string | Echo of active controls mode |

## Decision bundle: `scenario_validation_warnings`

List of strings on the simulation payload when panel/scenario validation finds non-fatal issues. Surfaced on CLI as **SCENARIO VALIDATION WARNING**.

See also: [../03_planning/decision_runbook.md](../03_planning/decision_runbook.md) §2e, [../03_planning/planning_execution.md](../03_planning/planning_execution.md), [../01_getting_started/config_yaml.md](../01_getting_started/config_yaml.md).
