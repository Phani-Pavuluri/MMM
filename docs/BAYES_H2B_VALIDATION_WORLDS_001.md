# Bayes-H2b Validation Worlds — Hierarchical Evidence Propagation and Trust Semantics

**Document ID:** `BAYES_H2B_VALIDATION_WORLDS_001`  
**Version:** `1.0.0`  
**Status:** **Accepted** (specification for Track 2 materialization — does **not** authorize Bayes-H3, PyMC, or Bayesian model code)  
**Date:** 2026-05-29  
**Authority:** [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) (`bayes_h2b_hierarchical_experiment_prior_scope_rules_v1`, **Accepted**)  
**Related:** [world_catalog.md](05_validation/world_catalog.md) · [groundtruth_contract.md](05_validation/groundtruth_contract.md) · [bayes_h2_calibration_signal_mapping_adr.md](05_validation/bayes_h2_calibration_signal_mapping_adr.md) · [trust_report_semantics.md](05_validation/trust_report_semantics.md)

---

## 1. Status

| Field | Value |
|-------|--------|
| **Purpose** | Convert Bayes-H2b ADR rules into **concrete validation-world specifications** for Track 2 reliability tooling |
| **Deliverable type** | Catalog + truth-design specification (docs-only at this step) |
| **Planned bundle root** | `validation/worlds/<world_id>/` per [world_materialization.md](05_validation/world_materialization.md) |
| **Catalog index** | `validation/worlds/world_catalog.index.json` (seven WORLD-BAYES-* rows committed) |
| **Blocks** | Bayes-H2d (architecture), Bayes-H3 (PyMC) until worlds materialized **and** [runner contract](BAYES_H2B_VALIDATION_RUNNER_002.md) CI smoke passes |
| **Does not block** | Ridge production path, existing WORLD-001–012 reliability program |

---

## 2. Context

Bayes-H1–H2b ADRs froze platform ABI: **CalibrationSignal** ingress, **DecisionSurface** / full-panel Δμ decisioning, **TrustReport** for evidence quality, **Release Gates** unchanged.

The seven **`WORLD-BAYES-*`** worlds prove **contract behavior**:

- Correct signal **routing** and **propagation** (up / down / lateral)  
- **Inclusion / exclusion** and **fail-closed** conflict handling  
- **TrustReport visibility** (no silent drops)  
- **Release-gate / readiness** implications  

They do **not** prove Bayesian inference quality, coefficient recovery, or posterior calibration. Those belong to Bayes-H4 **after** Bayes-H3 — and only on separate worlds.

**Decision recovery ≠ attribution recovery** (Reliability Program 5C–5D): these worlds may include `decision_truth` for optional Δμ smoke, but **pass/fail** is on evidence routing and TrustReport — not REC-4B2-001 coef recovery.

---

## 3. Non-goals

| Non-goal | Notes |
|----------|--------|
| Implement PyMC, priors, likelihoods, samplers, model classes | Bayes-H3+ |
| Prove “recover true β” or pooling τ | Bayes-H4 / separate geo worlds |
| Authorize Bayes-H2c hierarchical model math spec | **Blocked** until this catalog is materialized + runner contract `002` |
| Authorize Bayes-H3 | Blocked until H2c accepted |
| Change DecisionSurface, Estimand, CalibrationSignal base contracts | Extensions listed in §9 only |
| Replace existing WORLD-008–012 recovery certification | Parallel Track 4 stratum |

---

## 4. Shared validation assumptions

### 4.1 What is under test

The **signal registry + hierarchy propagation adapter** (future) or **standalone contract validator** (interim) must:

1. Ingest `experiment_truth` → **CalibrationSignal** fixtures (JSON).  
2. Apply Bayes-H2 mechanism rules (R1–R10).  
3. Apply Bayes-H2b propagation, conflict, claim-level rules.  
4. Emit a **TrustReport-shaped** `hierarchy_evidence` object (no fit required).  

### 4.2 What is not under test

- Posterior sampling, PPC, coverage  
- Ridge vs Bayes fit comparison  
- Optimizer allocation quality (unless optional VAL-004 smoke tagged `optional`)  

### 4.3 Pass/fail unit

| Result | Meaning |
|--------|---------|
| **PASS** | All **mandatory assertions** (MA-*) for the world hold on validator output |
| **FAIL** | Any mandatory assertion fails |
| **WARN** | Optional assertions (OA-*) fail — does not fail world |

### 4.4 Validation IDs (canonical: runner contract)

**Registry numbering:** [BAYES_H2B_VALIDATION_RUNNER_002.md §9](BAYES_H2B_VALIDATION_RUNNER_002.md) defines **`VAL-BAYES-001`–`012`** and **`VAL-BAYES-H2B-SMOKE`**. This table is a summary; runner doc supersedes on conflict.

| `validation_id` | Description | Worlds |
|-----------------|-------------|--------|
| `VAL-BAYES-001` | CalibrationSignal-only ingress | All |
| `VAL-BAYES-002` | Scope mapping | All |
| `VAL-BAYES-003` | Influence class | All |
| `VAL-BAYES-004` | Propagation eligibility | All |
| `VAL-BAYES-005` | Inclusion/exclusion | All |
| `VAL-BAYES-006` | Conflict surfacing | CONFLICT, SPARSE-GEO |
| `VAL-BAYES-007` | TrustReport visibility | All |
| `VAL-BAYES-008` | Release-gate implications | All |
| `VAL-BAYES-009` | No silent averaging | CONFLICT |
| `VAL-BAYES-010` | No model-fit dependency | All |
| `VAL-BAYES-011` | No posterior dependency | All |
| `VAL-BAYES-012` | No alternate decision surface | All |
| `VAL-BAYES-H2B-SMOKE` | Seven-world loadable + deterministic pass | Suite |

**Certification module (planned):** `BayesHierarchyEvidenceCertification` — runs VAL-BAYES-* without training.

### 4.5 Estimator stub policy

Until Bayes-H3 exists, certification uses **`hierarchy_evidence_validator`** per [BAYES_H2B_VALIDATION_RUNNER_002.md](BAYES_H2B_VALIDATION_RUNNER_002.md) that:

- Reads `world_truth.json` + `calibration_signals.json`  
- Writes `hierarchy_evidence_report.json`  
- Does **not** call PyMC or alter coefficients  

Ridge train on the same panel is **optional** for artifact fingerprint smoke only.

---

## 5. Shared synthetic hierarchy

### 5.1 Scope graph (all Bayes worlds unless noted)

```text
national
├── region-west          region-east
│   ├── state-CA         state-TX
│   │   ├── dma-sf       dma-la       dma-houston
│   │   └── ...
```

| Node type | IDs (canonical set) | Notes |
|-----------|---------------------|--------|
| `national` | `US` | Singleton |
| `region` | `west`, `east` | 2 regions |
| `state` | `CA`, `TX` | 2 states per region |
| `dma` | `sf`, `la`, `houston`, `dallas`, `phoenix`, `denver` | 6 DMAs — 2 sparse (low spend weight) |
| `channel` | `search`, `social` | Cross-cut |

**Sparse DMAs (default):** `phoenix`, `denver` — `spend_weight` &lt; 5% national each.

### 5.2 Panel geometry (materialization defaults)

| Field | Value |
|-------|--------|
| `date_frequency` | `weekly` |
| `n_periods` | 104 |
| `n_geos` | 6 DMA rows in panel |
| `train_window` | weeks 1–78 |
| `eval_window` | weeks 79–104 |

### 5.3 Hierarchy truth (for diagnostics only)

`geo_structure.hierarchy` in `world_truth.json` encodes parent/child edges matching §5.1. **Not** used as pass/fail on coef recovery.

Optional `media_truth.hyperparameters` (future): `mu_search`, `tau_search` for documentation — **not** asserted in VAL-BAYES-*.

---

## 6. Shared CalibrationSignal contract assumptions

### 6.1 Fixture file

Each bundle includes:

`validation/worlds/<world_id>/calibration_signals.json`

Array of **CalibrationSignal** records traceable to `experiment_truth.treatment_effects`.

### 6.2 Required fields (all signals)

| Field | Type | Notes |
|-------|------|--------|
| `signal_id` | string | Stable within world |
| `source` | enum | `geox` \| `cls` \| `ab_test` \| `replay` \| `holdout` |
| `scope_type` | enum | §5.1 + `channel` \| `segment` |
| `scope_id` | string | |
| `parent_scope_ids` | string[] | |
| `child_scope_ids` | string[] | When aggregating |
| `geo_ids` | string[] | Resolved DMA list |
| `channel_ids` | string[] | |
| `estimand` | string | Registry id |
| `lift_scale` | string | e.g. `mean_kpi_level_delta` |
| `observed_lift` | float | |
| `lift_se` | float \| null | |
| `time_window` | object | `{start, end}` |
| `freshness` | object | `{age_days, tier: fresh\|stale\|expired}` |
| `design_quality_tier` | `A` \| `B` \| `C` | |
| `coverage_ratio` | float | 0–1 |
| `experiment_id` | string | Traceability |

### 6.3 Truth linkage

Each signal must reference `experiment_truth.treatment_effects[].unit_id` via `truth_ref` field (extension for worlds).

---

## 7. World specifications

### 7.1 WORLD-BAYES-GEOX-LOCAL

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-GEOX-LOCAL` |
| `world_family` | `bayes-hierarchy-evidence` |
| `world_description` | Single-DMA GeoX; local influence only; upward gated to summary. |
| `world_version` | `1.0.0` |
| `scenario_tags` | `bayes:geox-local`, `hierarchy:dma`, `signal:fresh` |
| `expected_capabilities` | `VAL-BAYES-001`, `VAL-BAYES-002`, `VAL-BAYES-007`, `VAL-BAYES-008` |
| `expected_failures` | `[]` |
| `difficulty` | `smoke` |
| `intended_certifications` | `BayesHierarchyEvidenceCertification` |
| `unsupported_uses` | `bayesian_prod_decide`, `coef_recovery_certification`, `causal_incrementality_claims` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-GEOX-LOCAL/` |

#### 1. Purpose

Validate **local GeoX** stays at DMA \(\beta_{g,c}\); **no** national \(\mu_c\) point mass; upward propagation is **summary/diagnostic** unless coverage gates pass.

#### 2. Synthetic setup

- **Treatment:** DMA `sf`, channel `search`, GeoX lift **+6%** (`observed_lift=0.06`), `lift_se=0.015`, fresh, design **A**  
- **Panel:** Full 6-DMA panel; spend concentrated so `sf` is informative  
- **experiment_truth:** One `treatment_effects` unit `geox-sf-search` aligned with signal  

#### 3. Hierarchy geometry

- Native scope: `dma` / `sf`  
- Parents: `state-CA` → `region-west` → `national`  
- `coverage_ratio` for state rollup: 0.33 (1 of 3 CA DMAs treated) — **below** state gate 0.60  

#### 4. CalibrationSignal inputs

| `signal_id` | `source` | `scope_type` | `scope_id` | `geo_ids` | `lift` | `lift_se` |
|-------------|----------|--------------|------------|-----------|--------|-----------|
| `SIG-GEOX-SF` | `geox` | `dma` | `sf` | `[sf]` | 0.06 | 0.015 |

#### 5. Estimand alignment requirements

- `estimand`: `ATT_dma_week_search` (in model allowlist fixture)  
- `lift_scale`: `mean_kpi_level_delta` — **PASS** inclusion gate  

#### 6. Expected propagation behavior

| Edge | Expected |
|------|----------|
| DMA `sf` | `local_likelihood_style` on `sf`×`search` |
| DMA → state | **Blocked** — `coverage_ratio` 0.33 &lt; 0.60 → `parent_summary_of_child_evidence` only |
| DMA → national | **Blocked** — no national mechanism |
| Lateral | `lateral_borrowing: none` or `pooled_only` only via unrelated pooling path |

#### 7. Expected inclusion/exclusion behavior

| Signal | Mechanism | Influence |
|--------|-----------|-----------|
| `SIG-GEOX-SF` | `likelihood_term` | Included, decision-grade |

#### 8. Expected TrustReport fields

- `included_signals`: `[SIG-GEOX-SF]`  
- `direct_signal_ids`: `[SIG-GEOX-SF]`  
- `propagation_path`: `[{edge: sf→CA, gate: coverage_fail, influence: none}]`  
- `claim_level` at `sf`: `directly_observed_experimental_evidence`  
- `claim_level` at `national`: **absent** or `parent_summary_of_child_evidence` only in `hierarchy_warnings`  
- `lateral_borrowing`: `none` or `pooled_only`  

#### 9. Expected release-gate effect

- `prod_decisioning_allowed`: false (research world)  
- `decision_safe`: may be true if only routing tested on stub  
- **No** `optimization_blocked` solely from this world unless paired with severe trust modifier test  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | National `included_signals` contains `SIG-GEOX-SF` with `national_calibration_evidence` |
| F2 | `claim_level` at national = `directly_observed_experimental_evidence` |
| F3 | State-level causal claim without gate pass |
| F4 | Missing `propagation_path` |

#### 11. Anti-patterns

- Local GeoX → national \(\mu_c\) point mass  
- Single DMA → “national search ROI” narrative without gate  

#### 12. Acceptance criteria (mandatory assertions)

| ID | Assertion |
|----|-----------|
| MA-GEOX-01 | `SIG-GEOX-SF` ∈ `included_signals` with `local_likelihood_style` |
| MA-GEOX-02 | No parent edge with `influence: national_calibration` |
| MA-GEOX-03 | `propagation_path` documents `coverage_fail` on sf→CA |
| MA-GEOX-04 | `claim_level(sf,search)` = `directly_observed_experimental_evidence` |
| MA-GEOX-05 | National scope has no `directly_observed_experimental_evidence` for this signal |

---

### 7.2 WORLD-BAYES-CLS-NATIONAL

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-CLS-NATIONAL` |
| `world_family` | `bayes-hierarchy-evidence` |
| `world_description` | National CLS on channel hyper; children get borrowed-strength caveat. |
| `expected_capabilities` | `VAL-BAYES-002`, `VAL-BAYES-007`, `VAL-BAYES-008` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-CLS-NATIONAL/` |

#### 1. Purpose

National CLS influences **\(\mu_c\)** / national calibration only; DMAs get **borrowed strength**, not direct CLS causal claims.

#### 2. Synthetic setup

- **CLS:** national scope, channel `search`, lift **+3%**, `lift_se=0.01`, fresh, design **A**  
- No local GeoX  

#### 3. Hierarchy geometry

- Native: `national` / `US`  
- Children: all 6 DMAs via `child_scope_ids`  

#### 4. CalibrationSignal inputs

| `signal_id` | `source` | `scope_type` | `scope_id` | `channel_ids` | `lift` | `lift_se` |
|-------------|----------|--------------|------------|---------------|--------|-----------|
| `SIG-CLS-NAT` | `cls` | `national` | `US` | `[search]` | 0.03 | 0.01 |

#### 5. Estimand alignment requirements

- `estimand`: `national_incremental_lift_search` — allowlisted  

#### 6. Expected propagation behavior

| Edge | Expected |
|------|----------|
| National → \(\mu_c\) | `national_calibration_evidence` or `hyper_prior_style` |
| National → each DMA | Downward **shrinkage center only**; `borrowed_strength_from_parent: true` |

#### 7. Expected inclusion/exclusion behavior

| Signal | Included | Notes |
|--------|----------|-------|
| `SIG-CLS-NAT` | Yes | National target only |

#### 8. Expected TrustReport fields

- `child_signal_ids` / `borrowed_strength_sources`: all DMAs × search  
- `claim_level` DMA: `model_estimate_with_borrowed_strength`  
- `unsupported_claims` includes `"DMA causal lift from national CLS"`  
- `local_vs_national_alignment`: computed (no local contradicting signal)  

#### 9. Expected release-gate effect

- Mislabeled DMA direct-experiment claim → `attribution_safe: false` if asserted in artifact narrative flags  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | Any DMA `claim_level` = `directly_observed_experimental_evidence` for CLS |
| F2 | Missing `borrowed_strength_from_parent` on child rows |
| F3 | `local_likelihood_style` on DMA from CLS signal |

#### 11. Anti-patterns

- National CLS → DMA causal claim  
- National A/B → state ROI as experiment result  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-CLS-01 | `SIG-CLS-NAT` targets national scope only |
| MA-CLS-02 | All DMAs have `borrowed_strength_from_parent: true` |
| MA-CLS-03 | DMA `claim_level` = `model_estimate_with_borrowed_strength` |
| MA-CLS-04 | `unsupported_claims` contains DMA causal CLS string |
| MA-CLS-05 | No `local_likelihood_style` on DMA from `SIG-CLS-NAT` |

---

### 7.3 WORLD-BAYES-CONFLICT

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-CONFLICT` |
| `world_family` | `bayes-hierarchy-evidence` |
| `scenario_tags` | `bayes:conflict`, `signal:multi-source` |
| `expected_capabilities` | `VAL-BAYES-003`, `VAL-BAYES-007` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-CONFLICT/` |

#### 1. Purpose

Conflicting **GeoX local (+8%)** vs **CLS national (+2%)** on same channel — **no silent average**; conflict surfaced.

#### 2. Synthetic setup

- DMA `houston` GeoX: +8% search  
- National CLS: +2% search  
- Overlapping channel, compatible estimands at respective scopes  

#### 3. Hierarchy geometry

- GeoX native: `dma` / `houston`  
- CLS native: `national`  
- Implied tension when inferring single national lift from local only  

#### 4. CalibrationSignal inputs

| `signal_id` | `source` | Scope | `lift` | `lift_se` |
|-------------|----------|-------|--------|-----------|
| `SIG-GEOX-HOU` | `geox` | dma/houston | 0.08 | 0.02 |
| `SIG-CLS-NAT` | `cls` | national/US | 0.02 | 0.008 |

#### 5. Estimand alignment requirements

- Both allowlisted at native scopes  

#### 6. Expected propagation behavior

- GeoX: local on `houston` only  
- CLS: national on \(\mu_c\)  
- **Conflict group** when `local_vs_national_alignment` divergence &gt; policy tolerance → `scope_conflicts` populated  
- **No** blended lift target (e.g. 5%)  

#### 7. Expected inclusion/exclusion behavior

| Mode | Expected |
|------|----------|
| Default | Both included at native scopes; conflict warning |
| `conflict_fail_closed: true` fixture variant | Downweight group or exclude conflicting national/local pair per ADR §9.4 |

#### 8. Expected TrustReport fields

- `conflicting_signals`: `[SIG-GEOX-HOU, SIG-CLS-NAT]` (or group id)  
- `sensitivity_required: true`  
- `scope_conflicts` non-empty with metric  
- `signal_weight_summary`: no single merged weight implying average  

#### 9. Expected release-gate effect

- `release_gate_recommendation`: `warn` or `block` when fail-closed fixture enabled  
- `optimization_blocked` may be true under severe conflict modifier  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | Implied posterior target = average(0.08, 0.02) in registry trace |
| F2 | Empty `conflicting_signals` |
| F3 | `sensitivity_required: false` |

#### 11. Anti-patterns

- Silent local/parent averaging  
- Conflict precision-merge without warning  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-CNF-01 | `conflicting_signals` non-empty |
| MA-CNF-02 | `sensitivity_required` = true |
| MA-CNF-03 | No `merged_lift` / `averaged_lift` field in trace |
| MA-CNF-04 | GeoX remains `local_likelihood_style` on houston |
| MA-CNF-05 | CLS remains `national_calibration_evidence` on national |

---

### 7.4 WORLD-BAYES-STALE

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-STALE` |
| `expected_capabilities` | `VAL-BAYES-004`, `VAL-BAYES-007` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-STALE/` |

#### 1. Purpose

**Fresh** local GeoX vs **stale** national CLS — stale downweighted/excluded; visible in TrustReport.

#### 2. Synthetic setup

- Fresh GeoX on `la`: +5%, age 30 days  
- Stale CLS national: +3%, age 200 days (past `calibration_max_age_days` default 180)  

#### 3. Hierarchy geometry

- Standard §5.1 graph  

#### 4. CalibrationSignal inputs

| `signal_id` | `freshness.tier` | `lift` |
|-------------|------------------|--------|
| `SIG-GEOX-LA` | `fresh` | 0.05 |
| `SIG-CLS-STALE` | `stale` | 0.03 |

#### 5. Estimand alignment requirements

- Both aligned at native scopes  

#### 6. Expected propagation behavior

- Fresh GeoX: full local influence  
- Stale CLS: precision × **0.25** or **excluded** per Bayes-H2 §11  
- Stale parent **does not** refresh child freshness  

#### 7. Expected inclusion/exclusion behavior

| Signal | Expected |
|--------|----------|
| `SIG-GEOX-LA` | Included, decision-grade |
| `SIG-CLS-STALE` | Included with downweight **or** `excluded` — must be explicit |

#### 8. Expected TrustReport fields

- `stale_signals`: contains `SIG-CLS-STALE`  
- `stale_hierarchy_signals`: same  
- `freshness_diagnostics`: documents age and tier  
- `signal_weight_summary`: CLS tier `low` or `excluded`  

#### 9. Expected release-gate effect

- Stale-heavy profile → trust modifier `caution` or `degraded` (Phase 5E alignment)  
- May contribute to `optimization_blocked` when governance strict  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | Stale CLS at full precision |
| F2 | `stale_signals` empty |
| F3 | Child treated as fresh because parent was stale |

#### 11. Anti-patterns

- Stale parent silently influencing children at full weight  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-STL-01 | `SIG-CLS-STALE` ∈ `stale_signals` |
| MA-STL-02 | CLS weight tier ≠ high (unless excluded entirely) |
| MA-STL-03 | `SIG-GEOX-LA` remains fresh tier in `included_signals` |
| MA-STL-04 | `freshness_diagnostics` present |

---

### 7.5 WORLD-BAYES-MISSING-SE

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-MISSING-SE` |
| `expected_capabilities` | `VAL-BAYES-005`, `VAL-BAYES-007` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-MISSING-SE/` |

#### 1. Purpose

**Missing SE** cannot become **decision-grade**; visible exclusion or diagnostic tier.

#### 2. Synthetic setup

- GeoX on `dallas`: point lift +7%, `lift_se: null`  
- Optional second signal: CI only (`ci_low`, `ci_high`, `ci_level`) — conversion policy tested  

#### 3. Hierarchy geometry

- DMA native  

#### 4. CalibrationSignal inputs

| `signal_id` | `lift_se` | Notes |
|-------------|-----------|-------|
| `SIG-GEOX-NOSE` | null | Post-data GeoX — default exclude decision-grade |
| `SIG-GEOX-CI` | null | `ci_90: [0.04, 0.10]` — diagnostic unless conversion declared |

#### 5. Estimand alignment requirements

- Aligned estimand — isolation tests **uncertainty** policy  

#### 6. Expected propagation behavior

- `SIG-GEOX-NOSE`: **excluded** decision-grade or `trust_report_only`  
- No high \(\omega\) from point estimate alone  

#### 7. Expected inclusion/exclusion behavior

| Signal | Expected |
|--------|----------|
| `SIG-GEOX-NOSE` | `excluded` or diagnostic-only influence class |
| `SIG-GEOX-CI` | diagnostic unless `ci_to_se_policy` declared in fixture |

#### 8. Expected TrustReport fields

- `missing_se_signals`: `[SIG-GEOX-NOSE]`  
- `excluded_signals` or `signal_weight_summary.tier` = `excluded` / `diagnostic`  
- Logged conversion for CI signal if applicable  

#### 9. Expected release-gate effect

- `decision_safe` stricter when missing-SE signals would have been sole evidence  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | `decision_grade: true` for `SIG-GEOX-NOSE` |
| F2 | `missing_se_signals` empty |
| F3 | Implied \(\omega\) &gt; cap from point estimate only |

#### 11. Anti-patterns

- Missing-SE as high-precision calibration  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-MSE-01 | `SIG-GEOX-NOSE` ∉ decision-grade included set |
| MA-MSE-02 | `missing_se_signals` contains `SIG-GEOX-NOSE` |
| MA-MSE-03 | No precision &gt; `governance_max_omega` from point-only |
| MA-MSE-04 | TrustReport documents exclusion reason `missing_se` |

---

### 7.6 WORLD-BAYES-SPARSE-GEO

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-SPARSE-GEO` |
| `expected_capabilities` | `VAL-BAYES-002`, `VAL-BAYES-003`, `VAL-BAYES-008` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-SPARSE-GEO/` |

#### 1. Purpose

**Sparse DMAs** borrow via hierarchy; **local GeoX** on one dense DMA; correct **claim levels**.

#### 2. Synthetic setup

- Sparse: `phoenix`, `denver` — low spend weights  
- Local GeoX on `sf` only (+6%, with SE)  
- National CLS absent (pooling-only borrow)  

#### 3. Hierarchy geometry

- Sparse nodes under `west` / `state` assignments per §5.1  

#### 4. CalibrationSignal inputs

| `signal_id` | `geo_ids` | Notes |
|-------------|-----------|-------|
| `SIG-GEOX-SF` | `[sf]` | Only local signal |

#### 5. Estimand alignment requirements

- Standard allowlist  

#### 6. Expected propagation behavior

| Geo | Expected |
|-----|----------|
| `sf` | `directly_observed_experimental_evidence` |
| `phoenix`, `denver` | `model_estimate_with_borrowed_strength` via pooling — **no** local experiment |
| Lateral | `pooled_only` — sf does not calibrate phoenix directly |

#### 7. Expected inclusion/exclusion behavior

- `SIG-GEOX-SF` included locally only  

#### 8. Expected TrustReport fields

- `pooling_diagnostics`: sparse flags on phoenix, denver  
- `borrowed_strength_sources`: links to hyper/parent pooling path  
- `lateral_borrowing`: `pooled_only`  
- `claim_level` sparse geos = `model_estimate_with_borrowed_strength`  

#### 9. Expected release-gate effect

- Wrong claim level → `attribution_safe: false`  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | Phoenix `claim_level` = `directly_observed_experimental_evidence` |
| F2 | `local_likelihood_style` on phoenix from sf signal |
| F3 | `lateral_borrowing: unsupported` with direct cross-DMA calibration |

#### 11. Anti-patterns

- Sibling DMA direct calibration  
- Sparse geo labeled as experiment result without local signal  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-SPR-01 | `sf` claim = `directly_observed_experimental_evidence` |
| MA-SPR-02 | `phoenix`, `denver` claim = `model_estimate_with_borrowed_strength` |
| MA-SPR-03 | `lateral_borrowing` ≠ unsupported cross-calibration |
| MA-SPR-04 | `pooling_diagnostics` lists sparse geos |
| MA-SPR-05 | No mechanism edge sf→phoenix |

---

### 7.7 WORLD-BAYES-ESTIMAND-EXCLUDE

**Catalog row (planned)**

| Field | Value |
|-------|--------|
| `world_id` | `WORLD-BAYES-ESTIMAND-EXCLUDE` |
| `expected_capabilities` | `VAL-BAYES-006`, `VAL-BAYES-007` |
| `bundle_path` | `validation/worlds/WORLD-BAYES-ESTIMAND-EXCLUDE/` |

#### 1. Purpose

**Product-level A/B** estimand on geo MMM panel **without** scope bridge — **excluded** from influence; TrustReport-only.

#### 2. Synthetic setup

- A/B lift on `segment` / `product_sku_test` — +10%  
- Geo panel has no bridge  

#### 3. Hierarchy geometry

- Segment node **orthogonal** to geo graph  

#### 4. CalibrationSignal inputs

| `signal_id` | `source` | `scope_type` | `estimand` | `scope_bridge_id` |
|-------------|----------|--------------|------------|-------------------|
| `SIG-AB-PROD` | `ab_test` | `segment` | `product_conversion_lift` | null |

#### 5. Estimand alignment requirements

- `product_conversion_lift` **not** in geo MMM allowlist fixture → **fail** inclusion  

#### 6. Expected propagation behavior

- **Zero** hierarchy propagation edges  
- Influence class: `excluded` or `trust_report_only`  

#### 7. Expected inclusion/exclusion behavior

| Signal | Expected |
|--------|----------|
| `SIG-AB-PROD` | **Not** in `included_signals` for fit influence |

#### 8. Expected TrustReport fields

- `estimand_excluded_signals`: `[SIG-AB-PROD]`  
- `excluded_hierarchy_signals`: same with `exclusion_reason: estimand_mismatch`  
- `excluded_signals` per Bayes-H2  

#### 9. Expected release-gate effect

- Exclusion visible — must not silently drop from audit trail  

#### 10. Failure modes

| ID | Failure |
|----|---------|
| F1 | `SIG-AB-PROD` in `included_signals` with non-zero weight |
| F2 | Any `propagation_path` edge for this signal |
| F3 | Missing `estimand_excluded_signals` entry |

#### 11. Anti-patterns

- A/B product lift moving geo \(\beta_{g,c}\) without bridge  
- Accepting evidence without estimand check  

#### 12. Acceptance criteria

| ID | Assertion |
|----|-----------|
| MA-EST-01 | `SIG-AB-PROD` ∉ decision-grade `included_signals` |
| MA-EST-02 | `estimand_excluded_signals` contains `SIG-AB-PROD` |
| MA-EST-03 | `propagation_path` empty for this signal |
| MA-EST-04 | `exclusion_reason` = `estimand_mismatch` |

---

## 8. Cross-world invariants

| ID | Invariant | Applies to |
|----|-----------|------------|
| X-01 | All signals ingested only via CalibrationSignal fixtures | All |
| X-02 | No experiment-specific API in validation path | All |
| X-03 | Mandatory assertions pass without PyMC | All |
| X-04 | `hierarchy_evidence` block present in validator output | All |
| X-05 | No silent averaging across conflicting signals | CONFLICT (+ global) |
| X-06 | Missing SE never decision-grade | MISSING-SE (+ global) |
| X-07 | Stale signals in `stale_signals` when tier stale/expired | STALE (+ global) |
| X-08 | National evidence does not create DMA direct experiment claims | CLS-NATIONAL, CONFLICT |
| X-09 | Local GeoX does not set national point mass | GEOX-LOCAL, CONFLICT |
| X-10 | Estimand mismatch → excluded with audit trail | ESTIMAND-EXCLUDE |
| X-11 | DecisionSurface / Δμ not used as pass/fail for these certs | All |
| X-12 | `prod_decisioning_allowed: false` on research artifacts | All |

---

## 9. TrustReport required schema additions or confirmations

### 9.1 Confirmed (no base semantic change)

Bayes-H2 fields remain: `included_signals`, `excluded_signals`, `stale_signals`, `conflicting_signals`, `signal_weight_summary`, `sensitivity_required`, `posterior_evidence_alignment`.

### 9.2 Required extension: `hierarchy_evidence` (Bayes-H2b)

Validator output must include object matching [Bayes-H2b ADR §13](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md):

`hierarchy_scope_map`, `propagated_evidence`, `pooling_diagnostics`, `hierarchy_diagnostics`, `sensitivity_diagnostics`, `freshness_diagnostics`, `local_vs_national_alignment`, `direct_signal_ids`, `parent_signal_ids`, `child_signal_ids`, `sibling_signal_ids`, `borrowed_strength_sources`, `claim_level` (per scope/channel), `unsupported_claims`, `lateral_borrowing`, `propagation_path`, `included_hierarchy_signals`, `excluded_hierarchy_signals`, `stale_hierarchy_signals`, `conflicting_hierarchy_signals`, `missing_scope_metadata_signals`, `estimand_excluded_signals`, `missing_se_signals`.

**Schema ADR:** deferred to `BAYES_H2B_VALIDATION_RUNNER_002` — this document **confirms** required keys for world pass/fail.

---

## 10. Release gate implications

| Scenario | Expected gate behavior |
|----------|------------------------|
| All worlds PASS on contract validator | Does **not** imply `approved_for_prod` for Bayesian path |
| CONFLICT fail-closed fixture | `release_gate_recommendation: block` or `warn` in scorecard interpretation |
| STALE-heavy | Trust modifier downgrade; may set `optimization_blocked` per [trust_report_semantics.md](05_validation/trust_report_semantics.md) |
| MISSING-SE / ESTIMAND-EXCLUDE | Stricter `decision_safe` when evidence base incomplete |
| Research artifacts | Always `prod_decisioning_allowed: false` until Bayes-H5 |

**Unchanged:** Ridge promotion, `PolicyError` on prod decide, VAL-008 structural gates.

---

## 11. Implementation-readiness checklist

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Accept `BAYES_H2B_VALIDATION_WORLDS_001` | Architecture | ✅ This document |
| 2 | Add 7 rows to `validation/world_catalog.index.json` | Track 2 | ✅ Done |
| 3 | Author Bayes-H2b fixture JSON per bundle (`calibration_signals`, `hierarchy_evidence_fixture`, …) | Track 2 | ✅ Done |
| 4 | Materialize panel CSV / `train_config.yaml` (Ridge smoke optional) | Track 2 | Pending |
| 5 | Register VAL-BAYES-001–008 in `validation_registry.md` | Track 2 | Pending |
| 6 | Accept [BAYES_H2B_VALIDATION_RUNNER_002](BAYES_H2B_VALIDATION_RUNNER_002.md) | Architecture | ✅ |
| 7 | Implement `hierarchy_evidence_validator` stub | Track 2 | Pending |
| 8 | Wire `VAL-BAYES-H2B-SMOKE` in CI | Track 2 | Pending |
| 9 | Bayes-H2d hierarchical model spec ADR | Track 4 | **Blocked** until row 8 passes |
| 10 | Bayes-H3 PyMC research | Track 4 | **Blocked** until H2d + Bayes-H4 worlds |

---

## 12. Open questions

| ID | Question | Resolution phase |
|----|----------|------------------|
| OQ-W01 | Exact `conflict_tolerance` default in validator vs ADR 15% | Runner 002 |
| OQ-W02 | `conflict_fail_closed` as separate world variant vs tag on CONFLICT | Materialize v1.1 |
| OQ-W03 | Whether optional Ridge train on Bayes worlds shares WORLD-008 config | Materialize |
| OQ-W04 | Catalog `world_family` enum registration: `bayes-hierarchy-evidence` | world_catalog ADR patch |

---

## 13. Final recommendation

1. **Accept** this document as the Track 2 catalog specification for Bayes-H2b.  
2. **Next:** `hierarchy_evidence_validator` stub + `VAL-BAYES-H2B-SMOKE` (fixtures under `validation/worlds/WORLD-BAYES-*/`).  
3. **Keep Bayes-H2d blocked** until CI smoke passes.  
4. **Sequence:** (1) worlds catalog → (2) runner contract → (3) fixtures ✅ → (4) validator + CI → (5) Bayes-H2d → (6) Bayes-H3.

**Bayes-H3 is not authorized by this document.** This document only unblocks **validation-world implementation** and contract certification scaffolding.

---

## Appendix A — Planned catalog index rows (JSON-ready)

```json
[
  {
    "world_id": "WORLD-BAYES-GEOX-LOCAL",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "Local GeoX DMA scope; upward propagation gated; no national point mass.",
    "world_version": "1.0.0",
    "world_contract_version": "groundtruth_world_v1",
    "scenario_tags": ["bayes:geox-local", "hierarchy:dma"],
    "expected_capabilities": ["VAL-BAYES-001", "VAL-BAYES-002", "VAL-BAYES-007", "VAL-BAYES-008"],
    "expected_failures": [],
    "difficulty": "smoke",
    "intended_certifications": ["BayesHierarchyEvidenceCertification"],
    "unsupported_uses": ["bayesian_prod_decide", "coef_recovery_certification"],
    "bundle_path": "validation/worlds/WORLD-BAYES-GEOX-LOCAL/"
  },
  {
    "world_id": "WORLD-BAYES-CLS-NATIONAL",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "National CLS; child DMAs borrowed-strength only.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-002", "VAL-BAYES-007", "VAL-BAYES-008"],
    "difficulty": "smoke",
    "bundle_path": "validation/worlds/WORLD-BAYES-CLS-NATIONAL/"
  },
  {
    "world_id": "WORLD-BAYES-CONFLICT",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "GeoX local vs CLS national conflict; no silent average.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-003", "VAL-BAYES-007"],
    "difficulty": "standard",
    "bundle_path": "validation/worlds/WORLD-BAYES-CONFLICT/"
  },
  {
    "world_id": "WORLD-BAYES-STALE",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "Fresh GeoX vs stale CLS; freshness visible.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-004", "VAL-BAYES-007"],
    "difficulty": "smoke",
    "bundle_path": "validation/worlds/WORLD-BAYES-STALE/"
  },
  {
    "world_id": "WORLD-BAYES-MISSING-SE",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "Missing SE cannot be decision-grade.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-005", "VAL-BAYES-007"],
    "difficulty": "smoke",
    "bundle_path": "validation/worlds/WORLD-BAYES-MISSING-SE/"
  },
  {
    "world_id": "WORLD-BAYES-SPARSE-GEO",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "Sparse DMA pooling and claim levels.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-002", "VAL-BAYES-003", "VAL-BAYES-008"],
    "difficulty": "standard",
    "bundle_path": "validation/worlds/WORLD-BAYES-SPARSE-GEO/"
  },
  {
    "world_id": "WORLD-BAYES-ESTIMAND-EXCLUDE",
    "world_family": "bayes-hierarchy-evidence",
    "world_description": "Misaligned estimand excluded; TrustReport audit only.",
    "world_version": "1.0.0",
    "expected_capabilities": ["VAL-BAYES-006", "VAL-BAYES-007"],
    "difficulty": "smoke",
    "bundle_path": "validation/worlds/WORLD-BAYES-ESTIMAND-EXCLUDE/"
  }
]
```

---

## References

- [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)  
- [bayes_h2_calibration_signal_mapping_adr.md](05_validation/bayes_h2_calibration_signal_mapping_adr.md)  
- [bayes_h1_decision_surface_preservation_adr.md](05_validation/bayes_h1_decision_surface_preservation_adr.md)  
- [world_catalog.md](05_validation/world_catalog.md)  
- [groundtruth_contract.md](05_validation/groundtruth_contract.md)
