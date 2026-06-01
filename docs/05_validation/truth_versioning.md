# Truth versioning policy (Phase 1A)

**Status:** Frozen — documentation only.  
**ADR:** DR-02 accepted — three independent version dimensions ([synthetic_architecture_decisions.md](synthetic_architecture_decisions.md)).

**Related:** [groundtruth_contract.md](groundtruth_contract.md) · [world_materialization.md](world_materialization.md) · [world_catalog.md](world_catalog.md)

---

## 1. Purpose

Synthetic validation must remain **comparable across time**. A single `world_version` string is insufficient: schema, generator logic, and materialization pipelines change on different cadences.

This policy defines **three independent versions** recorded in every world bundle `metadata.json` and every certification report.

---

## 2. Version dimensions

### 2.1 `world_contract_version`

| | |
|--|--|
| **What** | Version of the [GroundTruthWorld](groundtruth_contract.md) schema and truth semantics |
| **Example** | `groundtruth_world_v1` |
| **Changes when** | Field meaning changes, required domains added/removed, dot-paths in validation registry change |
| **Does not change when** | Generator bugfix that preserves semantics; Parquet column order; doc typo |

**Consumers:** Contract validators; registry `truth_object` resolution; ADR review for breaking changes.

### 2.2 `world_generator_version`

| | |
|--|--|
| **What** | Version of logic that **creates or populates** `world_truth.json` (DGP archetype + ScenarioBuilder) |
| **Example** | `generator_baseline_v1.2.0` |
| **Changes when** | Archetype math changes, sampling policy changes, default scenario axis changes |
| **Does not change when** | Materialization-only changes; contract doc clarifications |

**Consumers:** Reproducibility audits; ReliabilityScorecard stratification (“same contract, new generator”).

### 2.3 `materialization_version`

| | |
|--|--|
| **What** | Version of logic that **renders derived artifacts** from immutable truth |
| **Example** | `materialize_v1.0.1` |
| **Changes when** | Panel column naming, replay JSON shape, fingerprint computation on bundle, Parquet partition strategy |
| **Does not change when** | Generator truth changes (bump `world_generator_version` instead) |

**Consumers:** CI cache keys; checksum regeneration; train/decide fixture templates.

---

## 3. Relationship to catalog `world_version`

| Field | Meaning |
|-------|---------|
| `world_version` (catalog) | **Instance** version for a specific `world_id` — monotonic per world (e.g. `WORLD-001-baseline@3`) |
| `world_contract_version` | **Schema** compatibility |
| `world_generator_version` | **Truth authoring** compatibility |
| `materialization_version` | **Derived artifact** compatibility |

**Rule:** Certification reports must record all four: `world_id`, `world_version`, and the version triple.

---

## 4. Semantic versioning bump rules

Applies independently to `world_contract_version`, `world_generator_version`, and `materialization_version` where each uses `MAJOR.MINOR.PATCH` or tagged equivalents (`groundtruth_world_v2` = contract major).

### 4.1 PATCH

**Increment when:**

- Documentation clarifications with no semantic change
- Metadata additions that do not affect validation logic (e.g. `creation_timestamp` format)
- Non-semantic field ordering in JSON
- Checksum algorithm unchanged; file order in manifest sorted differently

**Downstream:** No recalibration of `TBD_v1` thresholds required.

### 4.2 MINOR

**Increment when:**

- Optional truth section added (backward compatible for readers)
- Optional derived artifact added (e.g. `experiments.parquet` on existing bundle)
- New catalog metadata fields
- New optional `scenario_tags`
- Generator adds optional randomness axis with default preserving prior worlds

**Downstream:** Existing certifications remain valid for unchanged worlds; new worlds may use new fields.

### 4.3 MAJOR

**Increment when (non-exhaustive):**

- Meaning of `decision_truth.true_delta_mu` changes (e.g. aggregation semantics)
- Decision truth or regret definition changes
- Existing **required** truth field renamed, retyped, or removed
- Materialized panel semantics change (row grain, target transform, spend units)
- `validation_registry.md` `truth_object` path breaking change
- Replay unit estimand mapping changes for same `unit_id`

**Downstream:**

- New `world_contract_version` or major generator/materialization bump
- ReliabilityScorecards **not comparable** to prior major without relabeling
- Registry thresholds (`TBD_v1` calibration) must be revisited

---

## 5. Compatibility matrix (informative)

| Change | Contract | Generator | Materialization |
|--------|----------|-----------|-----------------|
| Hill formula in archetype fixed (bug) | — | PATCH or MINOR | — |
| Add `privacy_shifts` optional domain | MINOR | — | — |
| Δμ aggregated mean → sum | MAJOR | MAJOR | possibly MAJOR |
| Parquet: week column renamed | — | — | MAJOR |
| New `WORLD-009` catalog entry | — | new instance `world_version` | — |

---

## 6. ReliabilityScorecard comparability

Scorecards must declare:

```text
scorecard_schema_version
world_contract_version_filter
world_generator_version_filter (optional)
materialization_version_filter (optional)
manifest_hash_baseline
```

**Comparing scorecards** requires matching contract major and documented generator/materialization filters. Cross-major comparisons are **invalid** for regression gates until DR-06 is resolved.

---

## 7. Migration from Phase 0 `world_version`

Phase 0 [groundtruth_contract.md](groundtruth_contract.md) used a single `world_version` field. Phase 1A splits semantics:

| Phase 0 | Phase 1A |
|---------|----------|
| `world_version` in truth metadata | Split into version triple + catalog `world_version` instance |
| `schema_version` alias | Renamed conceptually to `world_contract_version` |

Phase 1B JSON schema will use the three names explicitly; no Python implementation in 1A.

---

## 8. DR-02 decision summary

**Accepted:** Three independent versions with PATCH/MINOR/MAJOR rules above.

**Rejected:**

- Single global `world_version` only — conflates schema and rendering
- Git commit SHA as version — not semantic for consumers
- Package `__version__` only — does not track materialization

**Consequences:**

- Every bundle `metadata.json` includes the triple.
- Materializer and generator bumps are independent ADR notes.
- [validation_registry.md](validation_registry.md) threshold docs must cite contract major when set.

---

## 9. Open (not DR-02)

| ID | Topic |
|----|--------|
| DR-03 | Negative world catalog representation |
| DR-04 | Threshold ownership for `TBD_v1` |
| DR-05 | Runtime vs CI certification split |
| DR-06 | ReliabilityScorecard release gating |
