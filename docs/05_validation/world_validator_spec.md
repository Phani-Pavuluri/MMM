# World validator specification (Phase 1B)

**Status:** L1–L3 implemented in `mmm/validation/synthetic/validator.py`. Level 4 and JSON Schema deferred.  
**Inputs:** [world_schema.md](world_schema.md), [world_bundle_schema.md](world_bundle_schema.md), [validation_registry.md](validation_registry.md).

---

## 1. Purpose

Define **four validation levels** that any future validator (CLI, CI step, or pre-cert hook) must implement. Levels are cumulative: Level N assumes Level N−1 passed.

**Out of scope:** threshold values (`TBD_v1`), DGP math, certification execution, ReliabilityScorecard logic, negative-world catalog implementation (DR-03).

---

## 2. Validator outputs

| Output | Description |
|--------|-------------|
| `validation_level` | Highest level attempted |
| `passed` | boolean |
| `hard_failures` | list of invariant / rule ids |
| `warnings` | list of warning ids |
| `world_id` | From truth |
| `version_triple` | contract, generator, materialization |

Certification runners must refuse to start if Level 1 or Level 2 hard-fail.

---

## 3. Level 1 — Structural validation

**Goal:** Files exist; JSON shape and types are valid; schema versions recognized.

### 3.1 Checks

| Check ID | Rule |
|----------|------|
| L1-001 | Bundle directory exists for `world_id` |
| L1-002 | `world_truth.json` exists and parses as JSON object |
| L1-003 | All top-level sections in [world_schema.md](world_schema.md) §4 present |
| L1-004 | `metadata.world_contract_version` === `groundtruth_world_v1` |
| L1-005 | Required metadata fields present with correct JSON types |
| L1-006 | `metadata.json` exists with required fields per [world_bundle_schema.md](world_bundle_schema.md) |
| L1-007 | `checksums.json` exists for certification-grade bundles |
| L1-008 | `checksum_version` === `checksums_v1` |
| L1-009 | Required bundle files exist per catalog `family` (panel unless exempt) |
| L1-010 | `catalog_version` === `world_catalog_v1` when validating index entry |
| L1-011 | No unknown top-level keys in `world_truth.json` (strict mode) or warn in permissive mode |

### 3.2 Failure severity

All L1 failures are **hard failures**.

---

## 4. Level 2 — Semantic validation

**Goal:** Declared values are internally consistent and physically plausible for the v1 prod modeling path (no generative simulation).

### 4.1 Checks

| Check ID | Rule |
|----------|------|
| L2-001 | `time_truth.end_date` >= `start_date`; `n_periods` consistent |
| L2-002 | `train_window` indices within `[0, n_periods)`; `eval_window` disjoint or ordered as declared |
| L2-003 | `geo_truth.weights`: non-negative; sum = 1 ± 1e-9 |
| L2-004 | `geo_truth.n_geos` === length of `geos`; unique geos |
| L2-005 | `media_truth.channels` non-empty; unique |
| L2-006 | `coefficient_truth.true_beta_by_channel` keys exactly match `channels` |
| L2-007 | All `true_beta` and `intercept` finite (not NaN/Inf) |
| L2-008 | `transform_truth.adstock_decay_by_channel` values in (0, 1) |
| L2-009 | `transform_truth.hill_half_max_by_channel` > 0; `hill_slope_by_channel` > 0 |
| L2-010 | `baseline_spend_by_channel` and scenario spends >= 0 |
| L2-011 | `outcome_truth.model_form` === `semi_log` for v1 prod-cert worlds |
| L2-012 | `transform_truth` families geometric + hill for v1 prod-cert worlds |
| L2-013 | Experiment unit date ranges within `time_truth` calendar span |
| L2-014 | `experiment_truth.units[].channel` ∈ `channels` |
| L2-015 | `experiment_truth.units[].geos` ⊆ `geo_truth.geos` |
| L2-016 | `decision_truth.budget_constraints[].total_budget` > 0 |
| L2-017 | `artifact_truth.expected_gates[].expected` ∈ {`pass`,`fail`,`warn`} |

### 4.2 Failure severity

L2-001–L2-017 hard failures unless catalog marks world as `draft` (future optional flag — **deferred**).

---

## 5. Level 3 — Cross-object validation

**Goal:** References across truth domains and bundle files resolve correctly.

### 5.1 Checks

| Check ID | Rule |
|----------|------|
| L3-001 | `metadata.json` identity fields match `world_truth.metadata` |
| L3-002 | `decision_truth.scenarios[].candidate_spend` keys ⊆ `channels` |
| L3-003 | `decision_truth.budget_constraints[].true_optimal_spend` keys = `channels` |
| L3-004 | Optimizer budget: sum of optimum spend within declared tolerance of `total_budget` (**tolerance TBD_v1**) |
| L3-005 | Every `experiment_truth.units[].unit_id` appears in materialized `replay_units.json` when file required |
| L3-006 | Replay unit geos and channels match truth unit specs |
| L3-007 | `decision_truth.json` (if present) contains no `true_beta` or lift duplicates |
| L3-008 | `panel.parquet` columns include `geo_truth.geo_column_name`, week column, `target_column`, all `channels` |
| L3-009 | `drift_truth.changepoints[].period_index` < `n_periods` |
| L3-010 | `artifact_truth.expected_certification_levels` keys reference known cert types |
| L3-011 | `governance_truth.model_release_state` is known enum if present |
| L3-012 | Checksums match on-disk file bytes (L1-007 prerequisite) |
| L3-replay-* | Replay units ↔ `experiment_truth` (see §5.1) |

### 5.1 Replay unit checks (Phase 2B, implemented)

| Check ID | Rule |
|----------|------|
| L3-replay-missing-file | `experiment_truth.units` non-empty ⇒ `replay_units.json` exists |
| L3-replay-unexpected-file | No replay file when no experiment units |
| L3-replay-unknown-unit | Each `unit_id` in JSON exists in truth |
| L3-replay-world_id | `world_id` matches truth metadata |
| L3-replay-channel | `channel` / `treated_channel_names` ⊆ `media_truth.channels` |
| L3-replay-geo | `geo_ids` ⊆ `geo_truth.geos` |
| L3-replay-window | `time_window` within `time_truth` calendar span |
| L3-replay-lift-scale | `lift_scale` ∈ supported KPI-level scales |
| L3-replay-estimand | `estimand` non-empty |
| L3-replay-transform-mode | `replay_transform_mode` = `full_panel_transform_estimand_mask` |
| L3-replay-lift-mismatch | `lift` matches truth `lift_definition.value` |

### 5.2 Failure severity

L3-012 hard failure for certification-grade bundles. Others hard unless noted **TBD_v1** tolerance.

---

## 6. Level 4 — Certification compatibility validation

**Goal:** World declares sufficient truth for the certifications and `validation_id` rows it claims to support.

### 6.1 Checks

| Check ID | Rule |
|----------|------|
| L4-001 | Catalog `supported_certifications` ⊆ known cert module names |
| L4-002 | For each `validation_id` in catalog `expected_capabilities`, required truth sections populated |
| L4-003 | `VAL-001` requires `coefficient_truth` |
| L4-004 | `VAL-002` requires `transform_truth.adstock_decay_by_channel` |
| L4-005 | `VAL-003` requires Hill maps |
| L4-006 | `VAL-004` requires `decision_truth.scenarios` with `true_delta_mu` |
| L4-007 | `VAL-005` requires `decision_truth.budget_constraints` with optimum |
| L4-008 | `VAL-006` requires `experiment_truth.units` and `replay_units.json` |
| L4-009 | `VAL-009`/`VAL-013` require `artifact_truth` |
| L4-010 | Negative worlds: `expected_failures` non-empty; `negative_world` true |
| L4-011 | Unsupported combinations fail (e.g. `optimizer` cert without `decision_truth.budget_constraints`) |
| L4-012 | `unsupported_uses` includes `causal_incrementality_claims` for every catalog entry |
| L4-013 | Prod-cert worlds must not declare `outcome_truth.model_form` = `log_log` |

### 6.2 Failure severity

L4 failures are **hard** when running a certification that requested the incompatible capability.

---

## 7. Explicit invariants

### 7.1 Hard failures (must never pass validation)

| Invariant ID | Statement |
|--------------|-----------|
| INV-001 | `geo_truth.weights` sum to 1 (± 1e-9) |
| INV-002 | `world_id` in truth, metadata, and directory name are identical and immutable for published bundle |
| INV-003 | `decision_truth` numeric truth (`true_delta_mu`, optimum spends, regret) is immutable after publish |
| INV-004 | Checksums are reproducible: same truth + generator + materialization + seed ⇒ same required sha256 fields |
| INV-005 | `true_delta_mu` is defined only in `world_truth.json`, never inferred from `panel.parquet` or fit artifacts |
| INV-006 | No certification-grade bundle with hand-edited derived files (checksum mismatch) |
| INV-007 | Catalog entry must not list `validation_id` not satisfiable by truth sections (Level 4) |
| INV-008 | `world_truth.json` is the sole authoritative source of β, lift, and expected gates |

### 7.2 Warnings (validation passes with warnings recorded)

| Warning ID | Condition |
|------------|-----------|
| WARN-001 | `spend_process_spec.correlation_level` === `severe` |
| WARN-002 | Any `true_beta` magnitude above **TBD_v1** bound |
| WARN-003 | `n_periods` * `n_geos` below **TBD_v1** sample size floor |
| WARN-004 | `observation_noise_level` === `high` |
| WARN-005 | Hill half-max near zero relative to spend scale |
| WARN-006 | Adstock decay > **TBD_v1** (near-instant decay) |
| WARN-007 | More than **TBD_v1** fraction of experiment units marked `stale` |
| WARN-008 | `eval_window` shorter than **TBD_v1** periods |

Warnings do not block Level 1–3 unless escalated by future policy (deferred).

---

## 8. Validator invocation matrix (informative)

| Consumer | Minimum level |
|----------|----------------|
| Materializer post-hook | L1 + L2 |
| CI bundle publish | L1 + L2 + L3 |
| SyntheticCertification | L1–L4 for declared caps |
| ReliabilityScorecard batch | L1–L3 (L4 per cert) |

---

## 9. Explicitly deferred

| Topic | DR / phase |
|-------|------------|
| Numeric pass thresholds | `TBD_v1`; DR-04 |
| DGP equations and generator internals | Phase 2B+ |
| Negative-world template catalog | DR-03 |
| ReliabilityScorecard gating | DR-06 |
| Certification execution logic | Phase 4 |
| JSON Schema code generation | Optional after Phase 1B review |
| Runtime vs CI ownership split | DR-05 |

---

## 10. Related documents

| Doc | Role |
|-----|------|
| [world_schema.md](world_schema.md) | Field definitions validated at L1–L2 |
| [world_bundle_schema.md](world_bundle_schema.md) | `metadata.json`, `checksums.json` |
| [validation_registry.md](validation_registry.md) | `VAL-*` ↔ Level 4 mapping |
| [truth_versioning.md](truth_versioning.md) | When validation rules version |
