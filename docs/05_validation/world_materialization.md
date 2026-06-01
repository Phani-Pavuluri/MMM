# World materialization architecture (Phase 1A)

**Status:** Phase 2A smoke materializer, Phase 3A generators, Phase 3B ScenarioBuilder, Phase **4B-1** rich DGP materializer — see `materializer.py`, `generators.py`, `dgp_materializer.py`.  
**ADR:** DR-01 accepted — **Option C: world bundle directory** (see [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md)).

**Related:** [groundtruth_contract.md](groundtruth_contract.md) · [truth_versioning.md](truth_versioning.md) · [world_catalog.md](world_catalog.md)

---

## 1. Purpose

This document defines **how** frozen `GroundTruthWorld` truth is stored on disk, how derived artifacts are laid out, and how certifications and CI will consume bundles—without implementing Python code.

Goals:

- One immutable truth document per world
- Reproducible derived artifacts with checksums
- Alignment with MMM artifact governance (fingerprints, extension reports, decide bundles)
- No certification-grade hand-edited panels

---

## 2. Materialization options evaluated (DR-01)

### Option A — Single world file

**Layout:** `worlds/<world_id>/world.json` containing truth + embedded or base64 panel payloads, or truth only with inline arrays.

| Dimension | Assessment |
|-----------|------------|
| **Advantages** | Single artifact to copy; simple mental model; easy `git diff` on truth |
| **Disadvantages** | Large panels bloat JSON; mixes immutable truth with derived data; poor binary diff; violates separation of concerns |
| **Artifact size** | Poor — multi-MB JSON is slow in CI and review |
| **Lineage** | Weak — unclear which bytes are truth vs derived |
| **Reproducibility** | Ambiguous if panel embedded without separate hash |
| **Train→decide** | Awkward — decide needs panel path + extension report, not one monolith |
| **CI suitability** | Poor for caching large blobs; LFS temptation |
| **Human review** | Good for truth-only small worlds; unusable at scale |

### Option B — Split truth and derived artifacts (flat files)

**Layout:**

```text
worlds/<world_id>/
  world_truth.json
  panel.parquet
  experiment.parquet
  replay.parquet
  metadata.json
```

| Dimension | Assessment |
|-----------|------------|
| **Advantages** | Clear truth vs derived split; columnar panels efficient |
| **Disadvantages** | No standard manifest; `metadata.json` vs `world_truth.json` overlap; decision truth location unclear |
| **Artifact size** | Good for panels |
| **Lineage** | Moderate — needs explicit checksum file |
| **Reproducibility** | Good if versions recorded in metadata |
| **Train→decide** | Good — panel path explicit |
| **CI suitability** | Good with content hashes |
| **Human review** | Truth JSON reviewable; panel via notebook |

### Option C — World bundle directory (selected)

**Layout:**

```text
worlds/<world_id>/
  world_truth.json          # immutable GroundTruthWorld (all truth domains)
  panel.parquet               # derived panel
  experiments.parquet         # optional experiment-level table
  replay_units.json           # derived replay fixture (MMM contract)
  decision_truth.json         # derived or copied decision reference scenarios (no duplicate β)
  metadata.json               # catalog + version triple + materialization provenance
  checksums.json              # sha256 per file in bundle
```

| Dimension | Assessment |
|-----------|------------|
| **Advantages** | Matches artifact-store patterns; checksum manifest; room for decide fixtures; cert-grade boundary clear |
| **Disadvantages** | More files to manage; requires bundle validator (Phase 1B+) |
| **Artifact size** | Excellent — Parquet for panel; JSON for contracts |
| **Lineage** | Strong — `metadata.json` + `checksums.json` + version triple |
| **Reproducibility** | Strong — re-materialize bundle from truth + versions + seed |
| **Train→decide** | Strong — train config can point at `panel.parquet`; decide uses extension report from train fixture subdir |
| **CI suitability** | Strong — cache bundle by content hash; skip re-materialize on hit |
| **Human review** | Truth and catalog in JSON; binaries via checksum only |

### DR-01 decision: Option C

**Chosen:** World bundle directory.

**Rationale:**

1. Separates **immutable truth** (`world_truth.json`) from **derived materializations** (Parquet, replay JSON).
2. Supports **large panels** without polluting truth documents.
3. Supports **lineage and checksum gates** consistent with production artifact governance.
4. Allows **optional derived files** per world family (e.g. optimizer worlds need `decision_truth.json`; baseline worlds may omit `replay_units.json`).
5. Enables future **train/decide fixture subdirectories** under the same bundle without merging truth systems.

---

## 3. Canonical bundle layout (normative)

### 3.1 Required files (all certification-grade bundles)

| File | Role |
|------|------|
| `world_truth.json` | Full `GroundTruthWorld` per [groundtruth_contract.md](groundtruth_contract.md) |
| `metadata.json` | Catalog fields, version triple, `generation_seed`, materialization timestamp |
| `checksums.json` | `sha256` per bundle file (including self-reference rules below) |

### 3.2 Derived files (family-dependent)

| File | When required |
|------|----------------|
| `panel.parquet` | Any world used for train, replay, or full-panel decide |
| `replay_units.json` | `experiment_world`, replay registry rows (`VAL-006`) |
| `decision_truth.json` | Optimizer / Δμ worlds — scenario keyed references into `world_truth.decision_truth` |
| `experiments.parquet` | Optional tabular view of `experiment_truth` for tooling |

### 3.3 Optional fixture subtrees (Phase 1B+ spec)

```text
worlds/<world_id>/fixtures/
  train_config.yaml.template    # parameterizes paths to bundle panel/replay
  decide_config.yaml.template
  extension_report.fragment.json  # inject-only fields only — not truth
```

Templates reference bundle paths; they **must not** embed coefficients or lift values.

### 3.4 `checksums.json` rules

- One entry per file in the bundle except `checksums.json` itself.
- Algorithm: `sha256` hex digest of file bytes.
- `checksums.json` includes a digest of the sorted list of `{path, sha256}` pairs (manifest hash) for single-shot integrity checks.
- CI compares manifest hash to cached value; mismatch triggers re-materialize or fail.

### 3.5 Immutability and certification grade

| Rule | Requirement |
|------|-------------|
| **Immutable truth** | `world_truth.json` is write-once per `world_id` + version triple; edits require new catalog entry |
| **Reproducible derive** | Same truth + `world_generator_version` + `materialization_version` + `generation_seed` → same checksums |
| **No truth in derived files** | Derived files contain panel columns, replay units, and **references** (`world_id`, scenario ids) — not authoritative β or lift |
| **No hand-written panels** | Certification paths load panel only from bundle; pytest fixtures that inline DataFrames are non-cert unless migrated |
| **Manual edit forfeits grade** | Any change to derived file without re-materialize invalidates `checksums.json` and cert eligibility |
| **Cert result lineage** | Every certification report must cite `world_id`, `world_contract_version`, `world_generator_version`, `materialization_version`, and bundle manifest hash |

---

## 4. Materialization flow (future)

```text
GroundTruthWorld (authoring → world_truth.json)
        │
        ▼
   materializer  (world_generator_version + materialization_version + seed)
        │
        ├──► panel.parquet
        ├──► replay_units.json
        ├──► decision_truth.json  (scenario tables / pointers only)
        ├──► experiments.parquet  (optional)
        │
        ▼
   checksums.json + metadata.json
        │
        ▼
   certification inputs  (train / decide / replay / governance)
```

**Materializer responsibilities (documentation only):**

1. Read `world_truth.json` and validate against contract (Phase 1B JSON schema).
2. Emit derived artifacts per world family profile in [world_catalog.md](world_catalog.md).
3. Write `metadata.json` and `checksums.json`.
4. Never write estimated parameters back into `world_truth.json`.

**Certification responsibilities:**

1. Load bundle by `world_id`.
2. Verify checksums before run.
3. Run platform paths (train, decide, replay) using fixtures.
4. Compare outputs to truth via [validation_registry.md](validation_registry.md).
5. Emit report with version triple + manifest hash.

---

## 5. CI and train→decide compatibility (planned)

How bundles will support existing MMM workflows **without** new truth definitions:

| Workflow | Bundle usage (planned) |
|----------|------------------------|
| **Training fixture** | `fixtures/train_config.yaml.template` sets `data.path` → `panel.parquet`; replay path → `replay_units.json`; `data_version_id` derived from `world_id` |
| **Decide fixture** | Train output or `extension_report.fragment.json` + `decide_config` pointing at same panel fingerprint |
| **Replay unit generation** | Materializer builds `replay_units.json` from `experiment_truth` — same shape as `write_calibration_units_to_json` |
| **Promotion workflow tests** | Bundle `artifact_truth` declares expected promotion pass/fail; promotion JSON fixture under `fixtures/` |
| **Fingerprint mismatch tests** | Pair of bundles or metadata flag `fingerprint_mutator` spec — second materialization with perturbed panel for negative test |
| **Production readiness tests** | Train-on-bundle → extension report → readiness rollup vs `artifact_truth.expected_certification_levels` |

**Train→decide fingerprint alignment:**

- `metadata.json` records `expected_panel_fingerprint` after materialization (v1 fingerprint v2 `sha256_combined` when available).
- Decide tests load extension report from train fixture; mismatch worlds deliberately set divergent fingerprint in derived metadata only with truth documenting expected severe warning / failure.

**CI caching strategy (planned):**

- Cache key: `hash(world_truth.json) + world_generator_version + materialization_version + generation_seed`.
- On cache hit, verify `checksums.json` only.
- On miss, run materializer (Phase 2+) or pull pre-built bundle from artifact store.

---

## 6. Repository layout (recommended)

```text
# Truth bundles (certification-grade, versioned)
validation/worlds/<world_id>/...

# Catalog index (Phase 1B)
validation/world_catalog.index.json   # pointers only — no duplicate truth

# Not in repo at large scale (optional)
# CI artifact store: s3://.../mmm-validation-worlds/<manifest_hash>/
```

Large bundles may be **gitignored** with manifest committed; CI fetches by manifest hash. Policy decision deferred to Phase 2 (artifact hosting).

---

## 7. Phase 2A implementation (smoke)

| Component | Location |
|-----------|----------|
| Materializer | `mmm/validation/synthetic/materializer.py` (`materialize_v1.0.0`) |
| Validator L1–L3 | `mmm/validation/synthetic/validator.py` |
| Smoke world | `validation/worlds/WORLD-001-baseline/world_truth.json` |
| Tests | `tests/test_synthetic_world_materializer.py` |

```python
from pathlib import Path
from mmm.validation.synthetic import materialize_world, validate_bundle

bundle = Path("validation/worlds/WORLD-001-baseline")
materialize_world(bundle, overwrite=True)
assert validate_bundle(bundle, max_level=3).passed
```

Re-running `materialize_world` with the same `world_truth.json` yields **identical** `checksums.json` when truth `creation_timestamp` is fixed.

---

## 8. Phase 2B — Replay unit materialization (smoke)

| Component | Location |
|-----------|----------|
| Replay rendering | `mmm/validation/synthetic/replay_units.py` |
| Smoke world | `validation/worlds/WORLD-002-replay/` |
| Tests | `tests/test_synthetic_world_replay_materializer.py` |

### Replay materialization flow

```text
world_truth.json
  experiment_truth.units[]
        │
        ▼
  build_replay_units_payload()  ──► replay_units.json
        │                              (derived; not truth)
        ▼
  load_calibration_units_from_json()  (loader compatibility)
```

When `experiment_truth.units` is non-empty, the materializer writes `replay_units.json` with:

| Field | Role |
|-------|------|
| `unit_id`, `experiment_id`, `world_id` | Identity and traceability |
| `channel`, `treated_channel_names` | Media alignment |
| `geo_scope`, `geo_ids`, `time_window` | Scope for validator L3 |
| `lift`, `standard_error` | Trace aliases for `observed_lift` / `lift_se` |
| `lift_scale`, `estimand` | Economics contract |
| `replay_transform_mode` | Top-level echo; canonical value in `replay_estimand` |
| `replay_estimand` | Full spec for `ReplayEstimandSpec.from_dict` |

Default `replay_transform_mode`: `full_panel_transform_estimand_mask` (matches prod replay path).

**Loader expectation:** `mmm.calibration.units_io.load_calibration_units_from_json` must load the file without error; `ReplayEstimandSpec.from_dict` must accept each `replay_estimand` block.

**Not truth:** `replay_units.json` is a derived artifact. Authoritative lift and windows remain in `world_truth.json` only.

---

## 9. Phase 3A — Deterministic archetype generators (truth only)

| Component | Location |
|-----------|----------|
| Generators | `mmm/validation/synthetic/generators.py` |
| Baseline smoke | `validation/worlds/WORLD-003-generated-baseline/` |
| Replay smoke | `validation/worlds/WORLD-004-generated-replay/` |
| Tests | `tests/test_synthetic_world_generators.py` |

### Generator / materializer separation

```text
generate_baseline_world_truth(seed, world_id)
generate_replay_world_truth(seed, world_id)
        │
        ▼
  write_world_truth()  ──► world_truth.json   (authoritative; only file generators write)
        │
        ▼
  materialize_world()  ──► panel.parquet, replay_units.json, metadata.json, checksums.json, …
```

| Rule | Requirement |
|------|-------------|
| Generators emit truth only | No `panel.parquet`, no `replay_units.json`, no checksum side effects |
| Same `seed` + `world_id` + `world_generator_version` | Identical `world_truth.json` bytes (including deterministic `creation_timestamp` from seed) |
| Materializer unchanged | All derived artifacts still flow through `materialize_world` |
| Not DGP families | No stochastic panel equations in generators — constant panel materialization until Phase 2 DGP |
| Not ScenarioBuilder | Fixed archetype templates (`baseline_world`, `experiment_world`), not lattice composition |
| Not Monte Carlo | No random world families, batch sampling, or certification runners |

API:

- `generate_baseline_world_truth(seed, world_id)` — two-channel baseline with decision scenarios, empty `experiment_truth.units`
- `generate_replay_world_truth(seed, world_id)` — single-channel experiment world with one replay unit
- `write_world_truth(bundle_dir, truth)` — writes `world_truth.json` only; directory name must match `metadata.world_id`

Committed smoke seeds: `3003` (baseline), `3004` (replay).

---

## 10. Phase 3B — ScenarioBuilder (truth composition)

| Component | Location |
|-----------|----------|
| ScenarioBuilder | `mmm/validation/synthetic/scenario_builder.py` |
| Composition core | `compose_archetype_truth()` in `generators.py` |
| Spec doc | [scenario_builder.md](scenario_builder.md) |
| Smoke worlds | `WORLD-005-scenario-low-noise`, `WORLD-006-scenario-high-collinearity`, `WORLD-007-scenario-replay-drift` |

Flow: `ScenarioSpec` → `build_world_truth()` → `write_scenario_world()` → `materialize_world()`.

ScenarioBuilder emits **world_truth only**. Drift and collinearity severe cases are represented in `drift_truth` and `artifact_truth.expected_warnings` for MVP even when the materialized panel remains constant-level.

---

## 11. Phase 4B-1 — Rich DGP materialization (exact-recovery worlds)

| Component | Location |
|-----------|----------|
| DGP materializer | `mmm/validation/synthetic/dgp_materializer.py` |
| Spec | [dgp_materialization.md](dgp_materialization.md) |
| Canonical world | `validation/worlds/WORLD-008-exact-recovery/` |

Flow: `world_truth.json` → `materialize_dgp_world()` → `panel.parquet` + `dgp_diagnostics.parquet` (derived, not truth).

KPI is generated from `coefficient_truth` + `transform_truth` using production geometric adstock and Hill formulas. `observation_noise_std: 0` for exact recovery. Does **not** modify `world_truth.json`.

Validator L3 accepts `materialization_version` ∈ `{materialize_v1.0.0, dgp_materialize_v1.0.0}`.

---

## 12. Deferred (explicit)

- JSON Schema artifact files (optional)
- Train/decide recovery certification (Phase 4B-2)
- Validator Level 4 (certification compatibility)
- Monte Carlo / reliability batch runners
- `ReliabilityScorecard` implementation
- External benchmark ingestion

See [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) for phase order.

---

## 12. Related ADR

| ADR | Decision |
|-----|----------|
| DR-01 | Option C — world bundle directory ([synthetic_architecture_decisions.md](synthetic_architecture_decisions.md)) |
| DR-02 | Three-version policy ([truth_versioning.md](truth_versioning.md)) |
