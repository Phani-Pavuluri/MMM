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

## Extension report: calibration / replay disclosure

Written to `extension_report.calibration_summary` (and mirrored on Ridge `ridge_fit_summary` / BO `best_detail` when replay calibration is active). **Diagnostic only** — does not change the optimization objective.

### Per-unit replay lift (`implied_lift_from_counterfactual` / unit meta)

| Field | Type | Description |
|-------|------|-------------|
| `replay_transform_mode` | string | Canonical: `full_panel_transform_estimand_mask` |
| `replay_uses_full_panel_transform` | bool | `true` when transforms run on the full sorted panel |
| `lift_evaluated_on_estimand_mask_only` | bool | `true` — lift aggregated only on experiment estimand rows |
| `replay_estimand` | object | Serialized geo/time window + aggregation spec |

### BO trial / extension aggregate (`build_replay_calibration_metadata`)

| Field | Type | Description |
|-------|------|-------------|
| `calibration_refit_mode` | string | `full_panel_same_hyperparameters`, `fold_aligned_cv`, or `holdout_diagnostic_only` |
| `replay_refit_mode` | string | Config mirror: `full_panel_refit` \| `fold_aligned` \| `holdout_only_diagnostic` |
| `replay_uses_full_panel_refit` | bool | Train replay uses full-panel refit coef |
| `fold_replay_losses` | float[] | Per-CV-fold replay losses when `replay_refit_mode=fold_aligned` |
| `fold_replay_units_used` | int | Units scored in fold-aligned path |
| `fold_replay_units_skipped` | int | Units skipped (estimand / validation overlap) |
| `replay_train_loss` | float | Replay loss at full-panel refit |
| `replay_holdout_loss` | float \| null | Replay loss at last CV-fold coef (if available) |
| `replay_holdout_available` | bool | `false` when CV holdout replay was not computed |
| `replay_generalization_gap` | float \| null | `replay_holdout_loss − replay_train_loss` |
| `replay_generalization_gap_severity` | string | `none` \| `moderate` \| `severe` |
| `replay_overfit_warning` | string | Advisory text when gap is moderate/severe |
| `replay_training_units` | int | Unit count in train replay |
| `replay_holdout_units` | int | Same count when holdout available; `0` otherwise |
| `legacy_replay_warnings` | string[] | Deprecation / skip warnings (e.g. `legacy_replay_deprecated_use_evidence_registry`) |
| `legacy_replay_upgrade_warnings` | string[] | Window-slice → full-panel upgrade notices |

Replay gap fields measure **train vs CV-fold replay consistency**, not causal incrementality. See [../02_concepts/calibration.md](../02_concepts/calibration.md).

## Decision bundle: promotion lineage (optional)

When a promotion record is supplied to prod decide paths, `decision_bundle` may include:

| Field | Type | Description |
|-------|------|-------------|
| `promoted_model_id` | string | Model id from promotion record |
| `promotion_id` | string | Immutable promotion record id |
| `promotion_registry_ref` | string | Path to append-only JSONL registry |
| `promotion_fingerprint_match` | bool | Data/config fingerprint match at decide time |
| `promotion_expiration_date` | string \| null | ISO date when set |
| `rollback_lineage` | object | `rollback_of`, `parent_promotion_id` when rolling back |

See [promotion_workflow.md](promotion_workflow.md).

See also: [../03_planning/decision_runbook.md](../03_planning/decision_runbook.md) §2e, [../03_planning/planning_execution.md](../03_planning/planning_execution.md), [../01_getting_started/config_yaml.md](../01_getting_started/config_yaml.md).
