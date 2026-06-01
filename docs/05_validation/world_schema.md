# GroundTruthWorld schema specification (`world_truth.json`)

**Schema ID:** `groundtruth_world_v1`  
**Status:** Frozen (Phase 1B) — specification only; no JSON Schema file, no Python types, no generators.  
**File:** `validation/worlds/<world_id>/world_truth.json` per [world_materialization.md](world_materialization.md).

**Related:** [world_bundle_schema.md](world_bundle_schema.md) · [world_validator_spec.md](world_validator_spec.md) · [groundtruth_contract.md](groundtruth_contract.md) (Phase 0 conceptual contract)

---

## 1. Purpose

This document is the **normative field-level schema** for `world_truth.json`. It freezes names, types, requiredness, constraints, producers, and consumers so Phase 2+ cannot introduce parallel truth shapes.

**Out of scope here:** DGP equations, generation algorithms, pass thresholds, certification runner logic.

---

## 2. Document rules

| Rule | Requirement |
|------|-------------|
| Root object | Single JSON object with exactly the top-level sections defined below (required sections may be empty objects only where explicitly allowed). |
| Immutability | After bundle publication, `world_truth.json` bytes are immutable for a given `world_id` + `world_version`. |
| No derived truth | Values in `decision_truth` are authored from the generative story, not copied from fitted models or materialized panels. |
| Channel keys | Media and coefficient maps use the same channel identifier strings as `media_truth.channels`. |
| Contract version | `metadata.world_contract_version` must equal `groundtruth_world_v1` for this schema revision. |

---

## 3. Phase 0 → Phase 1B section mapping

| Phase 1B section | Phase 0 [groundtruth_contract.md](groundtruth_contract.md) |
|------------------|-------------------------------------------------------------|
| `metadata` | `metadata` |
| `time_truth` | `time_structure` |
| `geo_truth` | `geo_structure` |
| `media_truth` | `media_truth` (spend/process only) |
| `outcome_truth` | (split out from implicit target) |
| `transform_truth` | adstock/saturation fields formerly under `media_truth` |
| `coefficient_truth` | coefficients formerly under `media_truth` |
| `experiment_truth` | `experiment_truth` |
| `decision_truth` | `decision_truth` |
| `drift_truth` | `shift_truth` |
| `artifact_truth` | `artifact_truth` |
| `governance_truth` | (new explicit section) |

---

## 4. Top-level sections (required presence)

| Section | Required on all worlds | May be empty object |
|---------|------------------------|---------------------|
| `metadata` | **yes** | no |
| `time_truth` | **yes** | no |
| `geo_truth` | **yes** | no |
| `media_truth` | **yes** | no |
| `outcome_truth` | **yes** | no |
| `transform_truth` | **yes** | no |
| `coefficient_truth` | **yes** | no |
| `experiment_truth` | **yes** | yes — `{}` if no experiments |
| `decision_truth` | **yes** | yes — `{}` only for worlds that declare no decision scenarios in catalog |
| `drift_truth` | **yes** | yes — `{}` if no drift |
| `artifact_truth` | **yes** | no |
| `governance_truth` | **yes** | yes — `{}` uses platform defaults |

Family-specific required subfields are enforced in [world_validator_spec.md](world_validator_spec.md) Level 4.

---

## 5. Field reference

Notation: **R** = required, **O** = optional. Types use JSON primitives unless noted.

---

### 5.1 `metadata`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `world_id` | string | R | Stable catalog identifier | Catalog authoring | All certs; catalog; CI | Non-empty; matches bundle directory name; immutable after publish | `"WORLD-001-baseline"` |
| `world_version` | string | R | Instance version (semver) | Catalog authoring | Lineage; catalog | Semver `MAJOR.MINOR.PATCH` | `"1.0.0"` |
| `world_contract_version` | string | R | This schema revision | Schema release | Validator L1 | Must be `groundtruth_world_v1` | `"groundtruth_world_v1"` |
| `world_generator_version` | string | R | Truth authoring logic version | Generator (future) | Repro; scorecard | Non-empty | `"TBD_v1"` |
| `materialization_version` | string | R | Derived artifact renderer version | Materializer (future) | CI cache; checksums | Non-empty | `"TBD_v1"` |
| `generation_seed` | integer | R | RNG seed for stochastic components | Generator / ScenarioBuilder | Repro cert (`VAL-010`) | `>= 0` | `4242` |
| `scenario_tags` | array[string] | R | Lattice tags (`key:value`) | ScenarioBuilder | Catalog; scorecard strata | Unique strings; `^[a-z_]+:[a-z0-9_]+$` | `["signal:medium","noise:low"]` |
| `creation_timestamp` | string | R | ISO 8601 UTC issuance time | Authoring pipeline | Audit | Valid ISO 8601 ending in `Z` | `"2026-05-22T12:00:00Z"` |
| `archetype_id` | string | O | DGP archetype template id | Generator | Catalog family mapping | Known archetype enum (doc only) | `"baseline_world"` |
| `negative_world` | boolean | O | Shorthand: world tests expected failures | Catalog author | Level 4 cert compatibility | Default `false` | `false` |
| `description` | string | O | One-line human summary | Catalog author | Docs | Max 500 chars | `"Baseline coef recovery smoke"` |

---

### 5.2 `time_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `date_frequency` | string | R | Calendar period | Archetype | Panel; CV; replay windows | Enum: `weekly` (v1 prod) | `"weekly"` |
| `start_date` | string | R | First period (inclusive) | Archetype | Panel; replay | ISO date `YYYY-MM-DD` | `"2020-01-06"` |
| `end_date` | string | R | Last period (inclusive) | Archetype | Panel | `end_date >= start_date` | `"2021-12-27"` |
| `n_periods` | integer | R | Period count | Archetype | Validator L2 | `>= 4`; consistent with date range | `104` |
| `train_window` | object | R | Fit/replay-train span | ScenarioBuilder | Train fixtures | See subfields | `{"start_period_index":0,"end_period_index":79}` |
| `train_window.start_period_index` | integer | R | Zero-based start | ScenarioBuilder | Train | `>= 0` | `0` |
| `train_window.end_period_index` | integer | R | Inclusive end index | ScenarioBuilder | Train | `< n_periods` | `79` |
| `eval_window` | object | O | Holdout span | ScenarioBuilder | Replay holdout | Same subfields as train | `{"start_period_index":80,"end_period_index":103}` |
| `seasonality_declared` | boolean | O | Whether seasonal component exists in generative story | Archetype | Diagnostics only | — | `false` |

---

### 5.3 `geo_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `geos` | array[string] | R | Geo identifiers | Archetype | Panel; replay masks; fingerprint | Non-empty; unique | `["G0","G1","G2","G3"]` |
| `n_geos` | integer | R | Count of geos | Archetype | Validator L2 | Equals `len(geos)` | `4` |
| `weights` | object | R | Non-negative geo weights | Archetype | Δμ aggregation; L2 checks | Keys ⊆ `geos`; sum = 1 ± 1e-9 | `{"G0":0.25,"G1":0.25,"G2":0.25,"G3":0.25}` |
| `hierarchy` | object | O | Parent → children map | `geo_world` | Hierarchy research | Acyclic; keys in `geos` | `{"US":["G0","G1"]}` |
| `geo_column_name` | string | O | Panel column name for geo | Materializer | Train config template | Default `"geo_id"` | `"geo_id"` |

---

### 5.4 `media_truth`

Media **identifiers and spend process** only — not coefficients (see `coefficient_truth`).

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `channels` | array[string] | R | Media channel names | Archetype | All media maps; panel columns | Non-empty; unique | `["search","social"]` |
| `spend_process_spec` | object | R | Declarative spend generation spec (no formulas) | ScenarioBuilder | Collinearity warnings; materializer | See subfields | `{"kind":"constant","level":10.0}` |
| `spend_process_spec.kind` | string | R | Process class name | ScenarioBuilder | Generator (future) | Enum doc: `constant`, `trend`, `collinear_block`, `ar1` | `"constant"` |
| `spend_process_spec.correlation_level` | string | O | Tag for correlation axis | ScenarioBuilder | Warnings | `low` \| `medium` \| `severe` | `"low"` |
| `baseline_spend_by_channel` | object | O | Reference spend levels for scenarios | Archetype | BAU scenarios | Keys ⊆ `channels`; values `>= 0` | `{"search":10.0,"social":10.0}` |

---

### 5.5 `outcome_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `target_column` | string | R | KPI column name in panel | Archetype | Panel; train config | Non-empty | `"revenue"` |
| `target_scale` | string | R | Declared KPI scale for interpretation | Archetype | Semi_log path validation | Enum: `positive_level` | `"positive_level"` |
| `model_form` | string | R | Declared modeling form for world | Archetype | Prod path validation | v1 cert: `semi_log` | `"semi_log"` |
| `base_level_mean` | number | O | Declared typical KPI level (story) | Archetype | Smoke panels | `> 0` if set | `100.0` |
| `observation_noise_level` | string | O | Noise regime tag | ScenarioBuilder | Warnings | `low` \| `medium` \| `high` | `"low"` |

---

### 5.6 `transform_truth`

Canonical prod transform stack declaration (v1).

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `adstock_family` | string | R | Adstock type | Archetype | Transform policy; VAL-002 | v1 prod cert: `geometric` | `"geometric"` |
| `saturation_family` | string | R | Saturation type | Archetype | VAL-003 | v1 prod cert: `hill` | `"hill"` |
| `adstock_decay_by_channel` | object | R | True decay per channel | `adstock_world` | Adstock recovery | Keys = `channels`; each in `(0,1)` | `{"search":0.5,"social":0.5}` |
| `hill_half_max_by_channel` | object | R | Hill half-max | `saturation_world` | Hill recovery | Keys = `channels`; `> 0` | `{"search":10.0,"social":10.0}` |
| `hill_slope_by_channel` | object | R | Hill slope | `saturation_world` | Hill recovery | Keys = `channels`; `> 0` | `{"search":2.0,"social":2.0}` |
| `global_adstock_decay` | number | O | Single decay if shared | Archetype | Simpler worlds | In `(0,1)` | `0.55` |

---

### 5.7 `coefficient_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `intercept` | number | R | True intercept on modeling scale | Archetype | VAL-001; simulate | Finite | `4.605` |
| `true_beta_by_channel` | object | R | True media coefficients (modeling scale) | Generator | VAL-001; VAL-004; VAL-005 | Keys = `channels`; finite | `{"search":0.42,"social":0.10}` |
| `controls` | object | O | Control name → coefficient | Archetype | Control overlay tests | Finite values | `{}` |
| `interactions` | array[object] | O | Declared interaction terms | ScenarioBuilder | Identifiability | References valid channels | `[]` |

**Example field (normative pattern):**

| | |
|--|--|
| **Field** | `coefficient_truth.true_beta_by_channel.search` |
| **Type** | number (value in map) |
| **Required** | yes when `search` ∈ `media_truth.channels` |
| **Meaning** | Ground-truth coefficient used to define media contribution in the generative story |
| **Producer** | generator |
| **Consumers** | coefficient certification (`VAL-001`); optimizer certification (`VAL-005`); decision certification (`VAL-004`) |
| **Constraints** | finite; non-null |
| **Example** | `0.42` |

---

### 5.8 `experiment_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `units` | array[object] | O | Replay unit definitions | `experiment_world` | VAL-006; replay materializer | Empty array if no experiments | `[]` |
| `units[].unit_id` | string | R per unit | Stable unit key | Generator | Replay JSON | Unique within world | `"u1"` |
| `units[].channel` | string | R per unit | Treated channel | Generator | Replay | ∈ `channels` | `"search"` |
| `units[].geos` | array[string] | R per unit | Treated geos | Generator | Replay | Subset of `geo_truth.geos` | `["G0"]` |
| `units[].week_start` | string | R per unit | Inclusive start date | Generator | Replay | Within `time_truth` range | `"2020-03-02"` |
| `units[].week_end` | string | R per unit | Inclusive end date | Generator | Replay | `>= week_start` | `"2020-05-25"` |
| `units[].lift_definition` | object | R per unit | Declared lift | Generator | Calibration | See subfields | `{"scale":"mean_kpi_level_delta","value":0.02}` |
| `units[].lift_definition.scale` | string | R | Lift units | Generator | Replay ETL | MMM-supported scale names | `"mean_kpi_level_delta"` |
| `units[].lift_definition.value` | number | R | True lift | Generator | VAL-006 | Finite | `0.02` |
| `units[].uncertainty` | object | O | Observation noise on lift | ScenarioBuilder | VAL-007 | `se >= 0` | `{"se":0.08}` |
| `units[].freshness` | object | O | Staleness | ScenarioBuilder | Calibration freshness | `age_days >= 0` | `{"age_days":30,"stale":false}` |
| `units[].estimand` | string | O | Declared estimand label | Generator | Replay | Non-empty if set | `"geo_time_ATT"` |

---

### 5.9 `decision_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `scenarios` | array[object] | O | Named spend scenarios for Δμ | Archetype | VAL-004 | Unique `scenario_id` | See example object |
| `scenarios[].scenario_id` | string | R | Scenario key | Archetype | Simulate | Non-empty | `"bau_vs_bump_search"` |
| `scenarios[].candidate_spend_by_channel` | object | R | Spend plan | Archetype | Simulate | Keys ⊆ `channels`; `>= 0` | `{"search":20.0,"social":10.0}` |
| `scenarios[].baseline_spend_by_channel` | object | O | BAU comparison | Archetype | Simulate | Same | `{"search":10.0,"social":10.0}` |
| `scenarios[].true_delta_mu` | number | R | True full-panel Δμ | Truth authoring | VAL-004 | Finite; **not** sourced from panel | `1.25` |
| `budget_constraints` | array[object] | O | Optimizer constraint sets | `optimizer_world` | VAL-005 | Unique `constraint_set_id` | `[]` |
| `budget_constraints[].constraint_set_id` | string | R | Constraint key | Archetype | Optimizer cert | — | `"default_budget_100"` |
| `budget_constraints[].total_budget` | number | R | Budget scalar | Archetype | Optimizer | `> 0` | `100.0` |
| `budget_constraints[].true_optimal_spend_by_channel` | object | R | True optimum allocation | Truth authoring | VAL-005 | Keys = `channels`; sum ≈ budget | `{"search":70.0,"social":30.0}` |
| `budget_constraints[].true_regret_by_allocator` | object | O | Regret for named allocators | Truth authoring | Reliability program | Non-negative | `{"simulation_optimizer":0.05}` |
| `response_surface_ref` | string | O | Opaque handle to stored grid spec | `optimizer_world` | Optimizer cert mode | Must not embed fitted coefs | `"grid_v1_two_channel"` |

**Invariant:** `true_delta_mu` values are **authored** before materialization and validated independently of `panel.parquet` (see [world_validator_spec.md](world_validator_spec.md)).

---

### 5.10 `drift_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `changepoints` | array[object] | O | Structural breaks | `drift_world` | VAL-012 | Sorted by period index | `[]` |
| `changepoints[].period_index` | integer | R | Break location | Generator | Drift detection | `< n_periods` | `40` |
| `changepoints[].affected_domains` | array[string] | R | Domains that change | Generator | Validator L3 | Subset of domain enum | `["coefficient_truth"]` |
| `coefficient_drift` | array[object] | O | β drift schedule | `drift_world` | VAL-012; calibration readiness | — | `[]` |
| `policy_changes` | array[object] | O | Transform/governance policy shifts | ScenarioBuilder | Governance cert | — | `[]` |
| `privacy_shifts` | array[object] | O | Panel perturbations | ScenarioBuilder | VAL-009 fingerprint tests | — | `[]` |

---

### 5.11 `artifact_truth`

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `expected_gates` | array[object] | R | Expected gate outcomes | Catalog / negative worlds | VAL-013 | Valid `gate_id` registry (doc) | `[{"gate_id":"production_readiness_approved","expected":"pass"}]` |
| `expected_gates[].gate_id` | string | R | Gate identifier | Author | Governance cert | Non-empty | `"production_readiness_approved"` |
| `expected_gates[].expected` | string | R | `pass` \| `fail` \| `warn` | Author | Governance cert | Enum | `"pass"` |
| `expected_failures` | array[object] | O | Expected hard failures | Negative worlds | VAL-009 | — | `[]` |
| `expected_failures[].surface` | string | R | API surface | Author | CI negative tests | e.g. `decide_simulate` | `"decide_simulate"` |
| `expected_failures[].error_class` | string | R | Exception class name | Author | CI | e.g. `PolicyError` | `"PolicyError"` |
| `expected_warnings` | array[object] | O | Expected warnings | Author | Readiness / decide JSON | — | `[]` |
| `expected_certification_levels` | object | O | Expected cert report levels | Author | VAL-014 | Keys = cert type names | `{"synthetic":"exact","optimizer":"directional_fallback"}` |

---

### 5.12 `governance_truth`

Expected governance **posture** for extension report / decide tests (not live config).

| Field | Type | R/O | Meaning | Producer | Consumers | Constraints | Example |
|-------|------|-----|---------|----------|-----------|-------------|---------|
| `approved_for_optimization` | boolean | O | Expected governance flag | Author | Optimization gates | — | `true` |
| `model_release_state` | string | O | Expected `model_release.state` | Author | Decide prod gate | e.g. `planning_allowed` | `"planning_allowed"` |
| `require_production_certification` | boolean | O | Config expectation for strict gate tests | Author | VAL-013 | — | `false` |
| `require_promoted_model` | boolean | O | Promotion workflow tests | Author | VAL-011 | — | `false` |
| `replay_calibration_active` | boolean | O | Whether replay evidence expected | Author | VAL-006 | — | `true` |

---

## 6. `decision_truth.json` bundle file (derived, not duplicate truth)

Materialized `decision_truth.json` in the bundle contains **only**:

- `world_id`, `world_version`, version triple
- `scenario_index`: list of `{scenario_id, ref: "decision_truth.scenarios[n]"}`  
- Optional precomputed tables for tooling

It must **not** repeat `true_beta_by_channel` or lift values. See [world_bundle_schema.md](world_bundle_schema.md).

---

## 7. Family minimum subfields (informative)

| World family | Extra required sections / fields |
|--------------|----------------------------------|
| `baseline` | `coefficient_truth`, `decision_truth.scenarios` (≥1) |
| `adstock` | `transform_truth.adstock_decay_by_channel` |
| `saturation` | `transform_truth` Hill maps |
| `experiment` | `experiment_truth.units` (≥1) |
| `optimizer` | `decision_truth.budget_constraints` (≥1) |
| `drift` | `drift_truth` non-empty |
| `negative-artifact` | `artifact_truth.expected_failures` (≥1) |

Enforced in validator Level 4.

---

## 8. Deferred (Phase 1B)

- JSON Schema `.json` files
- Python / Pydantic types
- Threshold numeric values
- DGP equations
- Negative-world implementation detail (DR-03)
- ReliabilityScorecard schema

---

## 9. Change control

Breaking field renames or semantic changes require `world_contract_version` major bump per [truth_versioning.md](truth_versioning.md) and ADR update.
