# World catalog specification (Phase 1A)

**Status:** Partial index committed — `validation/worlds/world_catalog.index.json` includes **WORLD-BAYES-*** (Bayes-H2b); full catalog TBD.  
**Bundles:** Layout per [world_materialization.md](world_materialization.md) (Option C).

**Related:** [world_bundle_schema.md](world_bundle_schema.md) (catalog index schema) · [groundtruth_contract.md](groundtruth_contract.md) · [validation_registry.md](validation_registry.md) · [truth_versioning.md](truth_versioning.md)

---

## 1. Purpose

The **world catalog** is the index of certification-grade synthetic worlds. It answers:

- Which `world_id` exists and what it is for
- Which validation capabilities and certifications it supports
- Which version triple applies
- Whether it is a negative (expected failure) world

The catalog **does not duplicate truth**. Authoritative coefficients, lift, and gates live in `worlds/<world_id>/world_truth.json`.

---

## 2. Catalog index file (future)

**Path (planned):** `validation/world_catalog.index.json`

**Entries:** Array of catalog records (schema below). Each record points to `validation/worlds/<world_id>/` bundle root.

**Rules:**

- One catalog row per certifiable `world_id` + instance `world_version`.
- Updating truth semantics requires new instance version or new `world_id` — never silent overwrite.
- `expected_capabilities` lists `validation_id` values from [validation_registry.md](validation_registry.md).

---

## 3. Catalog record schema (required fields)

| Field | Type | Description |
|-------|------|-------------|
| `world_id` | string | Stable identifier; kebab-case suffix for family |
| `world_family` | string | Archetype / family (`baseline`, `adstock`, `saturation`, `experiment`, `optimizer`, `drift`, `collinearity`, `negative-artifact`) |
| `world_description` | string | One-paragraph human summary |
| `world_version` | string | Instance version for this catalog row (e.g. `1.0.0`) |
| `world_contract_version` | string | e.g. `groundtruth_world_v1` |
| `world_generator_version` | string | e.g. `generator_baseline_v1.0.0` (TBD at implementation) |
| `materialization_version` | string | e.g. `materialize_v1.0.0` (TBD at implementation) |
| `scenario_tags` | list[string] | Lattice coordinates; see [roadmap ScenarioBuilder](synthetic_validation_roadmap.md) |
| `expected_capabilities` | list[string] | `validation_id` values exercised (e.g. `VAL-001`) |
| `expected_failures` | list[string] | For negative worlds: `gate_id` or `validation_id` expected to fail; empty for positive worlds |
| `difficulty` | enum | `smoke` \| `standard` \| `stress` — CI tiering, not truth |
| `intended_certifications` | list[string] | Planned cert modules: `SyntheticCertification`, `ReplayCertification`, etc. |
| `unsupported_uses` | list[string] | Explicit non-claims (e.g. `causal_incrementality`, `bayesian_prod_decide`) |
| `bundle_path` | string | Relative path to bundle root |
| `manifest_hash` | string | Optional precomputed checksums manifest hash (populated after materialize) |

### Optional fields (minor catalog extensions)

| Field | Description |
|-------|-------------|
| `generation_seed` | Default seed for reproducibility docs |
| `negative_world` | boolean — shorthand for failure-test worlds |
| `supersedes` | prior `world_id` if replaced |
| `deprecated` | boolean + reason |

---

## 4. Field semantics

### `world_family`

Maps to DGP archetypes in the roadmap. One family may have many `world_id` instances (e.g. smoke vs stress).

### `expected_capabilities`

Must be subset of [validation_registry.md](validation_registry.md). Certifications must not assert capabilities not listed here.

### `expected_failures`

For **positive** worlds: usually `[]`. For **negative** worlds (e.g. `WORLD-008-negative-artifact`): list expected `artifact_truth` gate failures or `PolicyError` surfaces — aligns with `artifact_truth.expected_failures` in truth file, not duplicated prose.

### `difficulty`

| Value | Intended use |
|-------|----------------|
| `smoke` | PR CI; fast materialize; minimal geos/weeks |
| `standard` | Nightly |
| `stress` | Reliability program / pre-release deep tier |

### `intended_certifications`

Names only (no implementation). Examples: `SyntheticCertification`, `DecisionCertification`, `ReplayCertification`, `OptimizerCertification`, `GovernanceCertification`, `ArtifactCertification`.

### `unsupported_uses`

Mandatory disclaimer fields preventing misuse. Every row should include at minimum:

- `causal_incrementality_claims`
- `production_generalization_to_real_markets`

---

## 5. Illustrative catalog entries (examples only)

**Not committed data** — structure reference for Phase 1B/2.

### WORLD-001-baseline

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-001-baseline` |
| `world_family` | `baseline` |
| `world_description` | Low-interaction semi_log surface; coef recovery and Δμ smoke. |
| `world_version` | `1.0.0` |
| `world_contract_version` | `groundtruth_world_v1` |
| `world_generator_version` | `TBD_v1` |
| `materialization_version` | `TBD_v1` |
| `scenario_tags` | `signal:medium`, `noise:low`, `correlation:low` |
| `expected_capabilities` | `VAL-001`, `VAL-004`, `VAL-008`, `VAL-014` |
| `expected_failures` | `[]` |
| `difficulty` | `smoke` |
| `intended_certifications` | `SyntheticCertification`, `DecisionCertification` |
| `unsupported_uses` | `causal_incrementality_claims`, `bayesian_prod_decide` |

### WORLD-002-adstock

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-002-adstock` |
| `world_family` | `adstock` |
| `world_description` | Known geometric decay; impulse carryover and VAL-002. |
| `expected_capabilities` | `VAL-002`, `VAL-004`, `VAL-014` |
| `difficulty` | `smoke` |

### WORLD-003-saturation

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-003-saturation` |
| `world_family` | `saturation` |
| `world_description` | Hill parameters with monotone design-matrix path. |
| `expected_capabilities` | `VAL-003`, `VAL-005`, `VAL-014` |

### WORLD-004-replay

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-004-replay` |
| `world_family` | `experiment` |
| `world_description` | Replay units with declared lift and medium experiment quality. |
| `expected_capabilities` | `VAL-006`, `VAL-007`, `VAL-013` |
| `intended_certifications` | `ReplayCertification`, `GovernanceCertification` |
| `difficulty` | `standard` |

### WORLD-005-optimizer

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-005-optimizer` |
| `world_family` | `optimizer` |
| `world_description` | Two-channel surface with grid-computable optimum and regret. |
| `expected_capabilities` | `VAL-005`, `VAL-004`, `VAL-014` |
| `intended_certifications` | `OptimizerCertification`, `DecisionCertification` |

### WORLD-006-drift

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-006-drift` |
| `world_family` | `drift` |
| `world_description` | Coefficient drift and changepoints for readiness and VAL-012. |
| `expected_capabilities` | `VAL-012`, `VAL-007`, `VAL-013` |
| `difficulty` | `standard` |

### WORLD-007-collinearity

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-007-collinearity` |
| `world_family` | `collinearity` |
| `world_description` | Severe cross-channel correlation; identifiability diagnostics, not prod approval. |
| `expected_capabilities` | `VAL-001`, `VAL-013` |
| `scenario_tags` | `correlation:severe` |
| `unsupported_uses` | `approved_for_prod_without_review` |

### Bayes-H2b hierarchy evidence worlds (Track 4)

**Specification:** [BAYES_H2B_VALIDATION_WORLDS_001.md](../BAYES_H2B_VALIDATION_WORLDS_001.md) — seven worlds, `world_family: bayes-hierarchy-evidence`, VAL-BAYES-001–008. **Not yet materialized.**

| `world_id` | Purpose |
|------------|---------|
| `WORLD-BAYES-GEOX-LOCAL` | Local GeoX; gated upward propagation |
| `WORLD-BAYES-CLS-NATIONAL` | National CLS; borrowed-strength children |
| `WORLD-BAYES-CONFLICT` | No silent average |
| `WORLD-BAYES-STALE` | Freshness visibility |
| `WORLD-BAYES-MISSING-SE` | Missing SE not decision-grade |
| `WORLD-BAYES-SPARSE-GEO` | Sparse geo claim levels |
| `WORLD-BAYES-ESTIMAND-EXCLUDE` | Estimand mismatch excluded |

---

### WORLD-008-negative-artifact

| Field | Example value |
|-------|----------------|
| `world_id` | `WORLD-008-negative-artifact` |
| `world_family` | `negative-artifact` |
| `world_description` | Expect fingerprint mismatch and blocked readiness / decide warnings. |
| `expected_capabilities` | `VAL-009`, `VAL-013`, `VAL-008` |
| `expected_failures` | `fingerprint_match`, `production_readiness_approved` |
| `negative_world` | true |
| `intended_certifications` | `ArtifactCertification`, `GovernanceCertification` |
| `difficulty` | `smoke` |

---

## 6. v1 CHECK_REGISTRY migration (informative)

| Legacy check | Target `world_id` (illustrative) |
|--------------|----------------------------------|
| `semi_log_delta_mu_exact` | `WORLD-001-baseline` |
| `geometric_adstock_carryover` | `WORLD-002-adstock` |
| `hill_saturation_analytic` | `WORLD-003-saturation` |
| `two_channel_optimizer_direction` | `WORLD-005-optimizer` |
| `transform_policy_consistency` | `WORLD-008-negative-artifact` (or dedicated negative row) |

Exact mapping finalized in Phase 2 when bundles exist.

---

## 7. Change control

| Action | Requirement |
|--------|-------------|
| Add `world_id` | Catalog PR + bundle PR (or bundle generation job) |
| Bump instance `world_version` | New materialization; update `manifest_hash` |
| Bump contract major | ADR + registry review |

---

## 8. Deferred

- Committed `world_catalog.index.json`
- Catalog validator
- Auto-discovery of bundles from filesystem
- ScenarioBuilder-generated catalog rows at scale
