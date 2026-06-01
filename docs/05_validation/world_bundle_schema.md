# World bundle schema specification (Phase 1B)

**Status:** Frozen — documentation only; no JSON Schema files or validators.  
**Layout:** Option C bundle per [world_materialization.md](world_materialization.md).  
**Truth file:** [world_schema.md](world_schema.md) (`world_truth.json`).

---

## 1. Bundle directory layout

```text
validation/worlds/<world_id>/
  world_truth.json       # required — see world_schema.md
  metadata.json          # required
  checksums.json         # required after materialization
  panel.parquet          # required for most families
  replay_units.json      # required when experiment_truth.units non-empty
  decision_truth.json    # optional index — no duplicate coefficients
  experiments.parquet    # optional tabular experiment view
  fixtures/              # optional — train/decide templates (Phase 2A+)
```

---

## 2. `metadata.json`

Bundle-level provenance and catalog linkage. **Not a source of generative truth** — duplicates only identity and version fields from `world_truth.json.metadata` for fast CI reads.

### 2.1 Required fields

| Field | Type | Meaning | Constraints | Example |
|-------|------|---------|-------------|---------|
| `world_id` | string | Must match directory name and `world_truth.json` | Non-empty; immutable | `"WORLD-001-baseline"` |
| `world_version` | string | Instance version | Semver; equals truth metadata | `"1.0.0"` |
| `world_contract_version` | string | Schema id | `groundtruth_world_v1` | `"groundtruth_world_v1"` |
| `world_generator_version` | string | Generator build | Non-empty | `"TBD_v1"` |
| `materialization_version` | string | Materializer build | Non-empty | `"TBD_v1"` |
| `seed` | integer | Materialization RNG seed | Same as `world_truth.metadata.generation_seed` unless documented override | `4242` |
| `creation_timestamp` | string | Bundle materialization time (UTC) | ISO 8601 `Z` | `"2026-05-22T14:30:00Z"` |
| `scenario_tags` | array[string] | Copy of truth tags | Same set as truth | `["signal:medium"]` |
| `checksum_version` | string | Checksum manifest format version | Enum: `checksums_v1` | `"checksums_v1"` |

### 2.2 Optional fields

| Field | Type | Meaning | Constraints |
|-------|------|---------|-------------|
| `bundle_path` | string | Relative path from repo root | Default `validation/worlds/<world_id>` |
| `manifest_hash` | string | sha256 of canonical `checksums.json` body (excluding self-describe) | 64 hex chars |
| `expected_panel_fingerprint` | object | Post-materialize fingerprint snapshot | Keys per MMM fingerprint v2; for train→decide tests |
| `materialized_files` | array[string] | Files present in bundle | Subset of layout §1 |
| `catalog_ref` | string | Pointer to `world_catalog.index.json` entry | Same `world_id` |

### 2.3 Consistency rules (validator Level 3)

- `metadata.world_id` === `world_truth.metadata.world_id`
- Version triple matches truth metadata exactly
- `seed` === `world_truth.metadata.generation_seed` when both present

---

## 3. `checksums.json`

Integrity manifest for **certification-grade** bundles. Required after any materialization run.

### 3.1 Required fields (`checksums_v1`)

| Field | Type | Meaning | Constraints |
|-------|------|---------|-------------|
| `checksum_version` | string | Manifest format | Must be `checksums_v1` |
| `world_truth_sha256` | string | Digest of `world_truth.json` bytes | 64 lowercase hex |
| `panel_sha256` | string | Digest of `panel.parquet` | 64 hex; required if panel present |
| `experiment_sha256` | string | Digest of `experiments.parquet` **or** `replay_units.json` when experiments exist | 64 hex; use `replay_units.json` if no parquet |
| `replay_sha256` | string | Digest of `replay_units.json` | 64 hex; `null` only if `experiment_truth.units` empty |

**Clarification for `experiment_sha256`:**

| Bundle contents | `experiment_sha256` covers |
|-----------------|----------------------------|
| `experiments.parquet` present | `experiments.parquet` |
| Only `replay_units.json` | Same bytes as `replay_sha256` |
| No experiments | Omitted or JSON `null` with Level 2 waiver in catalog |

### 3.2 Optional fields (future — document only, not required in v1)

| Field | Type | When used |
|-------|------|-----------|
| `decision_truth_sha256` | string | When `decision_truth.json` present |
| `metadata_sha256` | string | Separate metadata integrity |
| `fixtures_sha256` | string | Hash of `fixtures/` tarball |
| `train_config_template_sha256` | string | Fixture template integrity |
| `manifest_hash` | string | Hash of sorted `{path, sha256}` list (see below) |

### 3.3 Manifest hash (optional enhancement)

When `manifest_hash` is present:

1. Build list of `{ "path": "<relative>", "sha256": "<hex>" }` for every file except `checksums.json`, sorted by `path`.
2. Canonical JSON serialize (sorted keys, no whitespace).
3. `manifest_hash = sha256(canonical_bytes)`.

Enables single-shot CI verification.

### 3.4 Reproducibility rule

Re-materializing with the same `world_truth.json`, `world_generator_version`, `materialization_version`, and `seed` must yield **identical** required checksum fields. Mismatch is a **hard failure** (see invariants).

---

## 4. `world_catalog.index.json` (catalog index)

**Path (planned):** `validation/world_catalog.index.json`  
**Normative detail:** [world_catalog.md](world_catalog.md) (Phase 1A prose) + this section (Phase 1B schema).

### 4.1 Root shape

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `catalog_version` | string | yes | Index schema version `world_catalog_v1` |
| `worlds` | array[object] | yes | Catalog entries |

### 4.2 Entry required fields

| Field | Type | Meaning | Constraints | Example |
|-------|------|---------|-------------|---------|
| `world_id` | string | Stable id | Unique in index | `"WORLD-001-baseline"` |
| `family` | string | Archetype family | Enum per [world_catalog.md](world_catalog.md) | `"baseline"` |
| `difficulty` | string | CI tier | `smoke` \| `standard` \| `stress` | `"smoke"` |
| `supported_certifications` | array[string] | Cert modules supported | Subset of cert names | `["SyntheticCertification"]` |
| `expected_failures` | array[string] | Expected failure tokens | Empty for positive worlds | `[]` |
| `tags` | array[string] | Scenario tags | Same vocabulary as `scenario_tags` | `["signal:medium"]` |
| `world_version` | string | Instance version | Semver | `"1.0.0"` |

### 4.3 Entry optional fields (recommended)

| Field | Meaning |
|-------|---------|
| `bundle_path` | Relative bundle root |
| `world_contract_version` | Schema version |
| `expected_capabilities` | `validation_id` list (`VAL-*`) |
| `negative_world` | boolean |
| `unsupported_uses` | disclaimer strings |
| `manifest_hash` | From materialization |

### 4.4 Illustrative index fragment (not committed)

```json
{
  "catalog_version": "world_catalog_v1",
  "worlds": [
    {
      "world_id": "WORLD-001-baseline",
      "family": "baseline",
      "difficulty": "smoke",
      "supported_certifications": ["SyntheticCertification", "DecisionCertification"],
      "expected_failures": [],
      "tags": ["signal:medium", "noise:low"],
      "world_version": "1.0.0"
    }
  ]
}
```

**No production catalog file is created in Phase 1B.**

---

## 5. `replay_units.json` (derived)

Written when `experiment_truth.units` is non-empty. JSON **list** of unit objects compatible with `load_calibration_units_from_json`.

Required loader fields (minimum):

- `unit_id`, `treated_channel_names`, `observed_lift`, `lift_se`, `target_kpi`, `geo_ids`
- `estimand`, `lift_scale`, `replay_estimand` (includes `replay_transform_mode`)

Traceability fields (validator L3; ignored by loader if unknown):

- `world_id`, `experiment_id`, `channel`, `geo_scope`, `time_window`, `lift`, `standard_error`, `replay_transform_mode`

Authoritative experiment definitions remain in `world_truth.experiment_truth.units`.

---

## 6. Cross-file reference rules

| Rule | Description |
|------|-------------|
| Truth authority | On conflict, `world_truth.json` wins over `metadata.json` |
| No coef in panel metadata | `panel.parquet` footer/metadata must not embed `true_beta` |
| Replay provenance | Each replay unit `unit_id` must match `experiment_truth.units[].unit_id` |
| Decision index | `decision_truth.json` references scenario ids present in truth |

---

## 6. Deferred

- JSON Schema artifacts
- Validator implementation
- Git LFS / artifact store policy
- Committed bundles and catalog index data
