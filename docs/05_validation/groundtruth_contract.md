# GroundTruthWorld contract (frozen)

**Contract ID:** `groundtruth_world_v1`  
**Status:** Frozen (Phase 0 conceptual contract). **Normative field schema:** [world_schema.md](world_schema.md) (Phase 1B).  
**Authority:** This document is the **only** permitted definition of synthetic ground truth for the MMM validation framework.

**Related:** [world_schema.md](world_schema.md) (Phase 1B normative field schema) · [validation_registry.md](validation_registry.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) · [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md)

---

## 1. Purpose

`GroundTruthWorld` is a **declarative truth system** for production MMM platform validation. It is not a dataset, not a panel CSV, and not an implicit generative story embedded in test code.

**Materialized artifacts** (panels, replay unit JSON, scenario YAML, trained run directories) are **views** derived from a world. If a test asserts a coefficient, Δμ, gate outcome, or regret value, that value must be traceable to a field in the world document referenced by `world_id`.

---

## 2. Contract rules (normative)

| Rule | Requirement |
|------|-------------|
| **Single source of truth** | One `world_id` → one authoritative world document per `world_version`. |
| **No duplicated truth** | Tests, certifications, and generators must not redefine β, lift, or expected gates outside the world document. |
| **Versioned evolution** | Breaking semantic changes increment `world_contract_version` (major); see [truth_versioning.md](truth_versioning.md). |
| **Reproducibility** | `generation_seed` + version triple + archetype parameters must fully determine materializations. |
| **Non-causal scope** | Worlds prove engineering robustness under declared truth; they do not certify causal incrementality on real markets. |

---

## 3. Canonical object shape

```text
GroundTruthWorld
├── metadata
├── time_structure
├── geo_structure
├── media_truth
├── experiment_truth
├── decision_truth
├── shift_truth
└── artifact_truth
```

---

## 4. Field reference

For every field: **purpose**, **producer** (who or what writes the field at world creation time), **downstream consumers** (who reads it for validation).

### 4.1 Metadata

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `world_id` | string | Stable identifier for registry, CI logs, and certification reports | Scenario composition or archetype catalog | All certifications; validation registry; ReliabilityScorecard |
| `world_version` | string | **Instance** version for this world (catalog row) | Catalog authoring | Certification lineage |
| `world_contract_version` | string | Schema/truth semantics (e.g. `groundtruth_world_v1`) | Phase 0 contract; bumped only via ADR | Validators; registry truth_object paths |
| `world_generator_version` | string | DGP/truth authoring logic version | Generator (Phase 2+) | Reproducibility; scorecard filters |
| `materialization_version` | string | Derived artifact rendering version | Materializer (Phase 2+) | CI cache; checksum regeneration |
| `generation_seed` | integer | Reproducible materialization of panels and stochastic components | ScenarioBuilder or archetype factory | Reproducibility certification; reliability program |
| `scenario_tags` | list[string] | Human- and machine-readable lattice coordinates (e.g. `signal:high`, `drift:on`) | ScenarioBuilder | Scorecard stratification; nightly sampling |
| `creation_timestamp` | ISO 8601 UTC | Audit trail for world document issuance | World authoring pipeline | Operator logs; regression forensics |
| `archetype_id` | string (optional) | Which DGP archetype template was used (`baseline_world`, etc.) | DGP library (Phase 2+) | Documentation; dependency mapping |
| `bundle_ref` | string (optional) | Relative path to world bundle root — **not** truth | Catalog index | CI resolves `worlds/<world_id>/`; see [world_materialization.md](world_materialization.md) |

### 4.2 Time structure

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `date_frequency` | enum | Calendar granularity (`weekly` for v1 prod path) | Archetype + ScenarioBuilder | Panel builder; CV split axis; replay unit windows |
| `start_date` | date | First period anchor | Archetype | Replay `week_start` / `week_end`; fingerprint stability |
| `end_date` | date | Last period anchor | Archetype | Horizon for adstock burn-in; eval windows |
| `n_periods` | integer | Count of periods (redundant with start/end but required for validation) | Archetype | Adstock state initialization; cert timeouts |
| `train_window` | period range | Rows used for fit / replay train | ScenarioBuilder | Train config; fold-aligned replay |
| `eval_window` | period range | Holdout or post-train evaluation | ScenarioBuilder | Replay holdout; generalization gap metrics |
| `seasonality_truth` | object (optional) | Declared seasonal component if generated | Archetype (optional) | Diagnostic decomposition only — **not** decision truth |

### 4.3 Geo structure

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `geos` | list[geo_id] | Stable geo keys | Archetype | Panel `geo_column`; replay geo masks; fingerprints |
| `hierarchy` | tree or null | Parent/child geo relationships for partial pooling scenarios | `geo_world` archetype | Hierarchy extensions (research); identifiability diagnostics |
| `weights` | map[geo_id → float] | Population or spend weights for aggregation semantics | Archetype | Full-panel Δμ aggregation checks; weighted replay (if enabled) |

### 4.4 Media truth

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `coefficients` | map[channel → float] on modeling scale | True linear predictors for semi_log path | Archetype / ScenarioBuilder | Coefficient recovery; calibration targets; identifiability |
| `adstock_parameters` | map[channel → decay] or global | True geometric adstock decay | `adstock_world` | Adstock recovery; design matrix carryover certs |
| `saturation_parameters` | map[channel → half_max, slope] | True Hill parameters | `saturation_world` | Hill recovery; monotone feature certs |
| `interactions` | list or map (optional) | Cross-channel interaction terms | ScenarioBuilder (correlation axis) | Separability / identifiability scenarios |
| `controls` | map[control → coefficient] (optional) | Non-media predictors | Archetype | Control overlay and planning assumption tests |
| `spend_process_spec` | declarative spec | How media time series is generated (IID, AR, collinear block) | ScenarioBuilder | Correlation / signal axes — not recovered directly |

**v1 prod modeling constraint:** Worlds targeting production certification must declare `adstock_parameters` and `saturation_parameters` consistent with **geometric adstock + Hill saturation** unless `artifact_truth` explicitly expects transform policy failure.

### 4.5 Experiment truth

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `treatment_effects` | list[unit specs] | Geo-time treatment increments (ATT or declared estimand) | `experiment_world` | Evidence registry; replay ETL |
| `lift_definitions` | map[unit_id → lift spec] | Declared lift scale (`mean_kpi_level_delta`, etc.) | `experiment_world` | Replay unit JSON; calibration objective |
| `uncertainty` | per-unit variance or SE | Observation noise on experimental lift | ScenarioBuilder (`experiment_quality` axis) | Calibration robustness; weighted replay |
| `freshness` | per-unit age or stale flag | Staleness for calibration readiness | ScenarioBuilder | Calibration freshness gates; GovernanceCertification |

### 4.6 Decision truth

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `true_delta_mu` | map[scenario_id → float] | Counterfactual Δμ on full panel for reference spend scenarios | Analytic engine on true surface (Phase 2+) | Δμ recovery; DecisionCertification; simulate path |
| `true_optimal_budget` | map[constraint_set_id → allocation] | Constrained budget optimum under true surface | Grid / analytic optimizer on truth | OptimizerCertification; optimizer recovery registry |
| `true_regret` | map[allocator_id → float] | Suboptimality vs true optimum | Derived from optimum and candidate allocation | Reliability program; decision quality metrics |
| `true_response_surface` | declarative handle or sampled grid | Reference μ(spend) for certification and debugging | Archetype (`optimizer_world`, `saturation_world`) | Optimizer certification modes; directional vs analytic tolerance |

**Note:** `true_response_surface` is a **reference**, not a substitute for running `mmm decide` — certifications compare platform outputs to truth, not to a second implementation hidden in tests.

### 4.7 Shift truth

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `changepoints` | list[{period, affected_fields}] | Structural breaks in spend or generative params | `drift_world` | Drift detection; readiness warnings |
| `coefficient_drift` | schedule or piecewise β(t) | Smooth or step drift in media coefficients | `drift_world` | Calibration drift; coefficient readiness |
| `policy_changes` | list[{period, policy_id}] | Transform stack or budget rule changes | ScenarioBuilder | `transform_policy` gates; GovernanceCertification |
| `privacy_shifts` | list[{period, perturbation_spec}] | Aggregation noise, rounding, or column suppression | ScenarioBuilder (`privacy_loss` axis) | Fingerprint mismatch; panel QA |

### 4.8 Artifact truth

| Field | Type (logical) | Purpose | Producer | Downstream consumers |
|-------|----------------|---------|----------|----------------------|
| `expected_gates` | list[{gate_id, expected: pass\|fail\|warn}] | Governance and safety gates | Scenario author / negative-world catalog | GovernanceCertification; prod workflow tests |
| `expected_failures` | list[{surface, error_class, message_substr?}] | Expected `PolicyError` or blocked commands | Negative-world catalog | ArtifactCertification; CI negative tests |
| `expected_warnings` | list[{warning_id, severity}] | Non-fatal warnings (readiness, fingerprint, directional optimizer) | Scenario author | Production readiness; decide payload checks |
| `expected_certification_levels` | map[cert_type → level] | e.g. synthetic `exact`, optimizer `analytic_tolerance` | Archetype + scenario_tags | Certification behavior registry row |

---

## 5. Materialization (Phase 1A architecture)

**Normative layout:** [world_materialization.md](world_materialization.md) — Option C world bundle directory.

| Bundle file | Relationship to world |
|-------------|------------------------|
| `world_truth.json` | Authoritative truth (this contract) |
| `panel.parquet` | Derived from time/geo/media/shift truth |
| `replay_units.json` | Derived from `experiment_truth` |
| `decision_truth.json` | Scenario references — must not duplicate β or lift |
| `checksums.json` | Integrity manifest for certification-grade bundles |

**Versioning:** [truth_versioning.md](truth_versioning.md) — contract, generator, and materialization versions are independent.

**Catalog:** [world_catalog.md](world_catalog.md) — index by `world_id`; no truth duplication in catalog.

**Forbidden:** Treating materialized panel statistics as ground truth; hand-editing derived files (forfeits certification grade).

### Phase 3A — Deterministic archetype generators

**Implementation:** `mmm/validation/synthetic/generators.py` (`GENERATOR_VERSION = archetype_gen_v1.0.0`).

| Responsibility | Owner |
|----------------|--------|
| Authoring `world_truth.json` from `(seed, world_id, archetype)` | **Generator** |
| `panel.parquet`, `replay_units.json`, `checksums.json`, bundle `metadata.json` | **Materializer only** |

**Generators (truth only):**

- `generate_baseline_world_truth(seed, world_id)` — `baseline_world`; no `experiment_truth.units`
- `generate_replay_world_truth(seed, world_id)` — `experiment_world`; one replay-compatible unit per smoke template
- `write_world_truth(bundle_dir, truth)` — writes **only** `world_truth.json`

**Determinism contract:** For fixed `generation_seed`, `world_id`, and `world_generator_version`, the canonical JSON object is identical across runs. `creation_timestamp` is derived from seed (not wall clock). Controlled variation across seeds includes geo count, horizon, coefficients, spend levels, and experiment windows — not unbounded random DGP families.

**Explicit limitations (Phase 3A):**

- These are **deterministic archetype templates**, not stochastic DGP families, not **ScenarioBuilder**, and not equation-backed panel simulation.
- Generators must not create panels, replay JSON, checksums, or certification outputs.
- Full generative media/KPI dynamics remain Phase 2 DGP library + Phase 3B composition.

**Smoke worlds:** `WORLD-003-generated-baseline` (seed 3003), `WORLD-004-generated-replay` (seed 3004). See [world_materialization.md](world_materialization.md) §9.

---

## 6. Compatibility with v1 package artifacts

Existing v1 certifications remain valid as **named smoke worlds** until migrated:

| v1 artifact | Migration target |
|-------------|------------------|
| `mmm/governance/synthetic_certification.CHECK_REGISTRY` | One `world_id` per check; no inline DGP in tests |
| `optimizer_certification` scenarios A/B/repeatability | `optimizer_world` + `decision_truth` |
| `test_synthetic_dgp_recovery` | Registry rows `VAL-*` with explicit `truth_object` paths |

Migration is tracked in Phase 1; Phase 0 only freezes this contract.

---

## 7. Change control

| Change type | Process |
|-------------|---------|
| Add optional field | Minor `world_version` doc bump; backward compatible |
| Rename or retype field | New `world_version`; ADR required |
| New truth domain | ADR + registry update + roadmap amendment |

**Frozen as of Phase 0:** Field names and top-level domains in this document. Implementation may not introduce parallel truth structs (e.g. `DGPConfig`, `TestFixtureTruth`) without ADR superseding this contract.
