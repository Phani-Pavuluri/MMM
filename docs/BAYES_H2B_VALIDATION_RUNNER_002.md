# Bayes-H2b Validation Runner — Contract, Fixtures, and Pass/Fail Semantics

**Document ID:** `BAYES_H2B_VALIDATION_RUNNER_002`  
**Version:** `1.0.0`  
**Status:** **Accepted** (validator contract + fixtures outline — does **not** authorize implementation code in this deliverable, Bayes-H2d, or Bayes-H3)  
**Date:** 2026-05-29  
**Prerequisites:** [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) · [BAYES_H2B_VALIDATION_WORLDS_001.md](BAYES_H2B_VALIDATION_WORLDS_001.md) (both **Accepted**)  
**Planned implementation module:** `mmm.validation.synthetic.hierarchy_evidence_validator` (stub only after this contract)  
**Related:** [world_catalog.md](05_validation/world_catalog.md) · [validation_registry.md](05_validation/validation_registry.md) · [trust_report_semantics.md](05_validation/trust_report_semantics.md)

---

## 1. Status

| Field | Value |
|-------|--------|
| **Purpose** | Define the **no-fit** `hierarchy_evidence_validator` contract, per-world fixture shape, assertion groups `VAL-BAYES-001`–`012`, pass/fail semantics, and CI smoke target `VAL-BAYES-H2B-SMOKE` |
| **Deliverable type** | Docs-only contract (this document) |
| **Unblocks** | Fixture JSON authoring, validator **stub** implementation, `world_catalog.index.json` rows, CI smoke wiring |
| **Does not unblock** | Bayes-H2d hierarchical model spec ADR, Bayes-H3 (PyMC), posterior or coef certification |
| **Blocks Bayes-H2d** | Until `VAL-BAYES-H2B-SMOKE` passes on materialized bundles |

---

## 2. Context

Track 2 has:

1. **Bayes-H2b ADR** — propagation, conflicts, claim levels, TrustReport hierarchy fields.  
2. **[BAYES_H2B_VALIDATION_WORLDS_001.md](BAYES_H2B_VALIDATION_WORLDS_001.md)** — seven `WORLD-BAYES-*` specifications with MA-* mandatory assertions.

This document defines **how** a future runner loads fixtures, executes deterministic rules (Bayes-H2 R1–R10 + Bayes-H2b §5–§13), emits a report, and maps results to **VAL-BAYES-*** without fitting any model.

The validator is the **reliability scaffolding gate** before Bayes-H2d (architecture) or Bayes-H3 (samplers).

---

## 3. Non-goals

| Non-goal | Notes |
|----------|--------|
| Implement PyMC, priors, likelihoods, samplers, model classes | Bayes-H3+ |
| Estimate posteriors, β, transforms, τ | Bayes-H4 / separate worlds |
| Validate coefficient or transform recovery | REC-4B2-* out of scope |
| Call Ridge train as pass/fail requirement | Optional fingerprint smoke only |
| Create Bayesian-only or experiment-specific APIs | Platform ABI frozen |
| Use optimizer, posterior draws, or coef tables as inputs | Bayes-H1 |
| Authorize Bayes-H2d or Bayes-H3 | Explicitly blocked |
| Change DecisionSurface, Estimand, CalibrationSignal, Release Gate base semantics | Extensions only |

---

## 4. Validator purpose

### 4.1 Component name

**`hierarchy_evidence_validator`**

### 4.2 Responsibility

Deterministically compute, from fixtures only:

| Check domain | Source rules |
|--------------|--------------|
| CalibrationSignal-only ingress | Bayes-H2 §1 |
| Scope → hierarchy target mapping | Bayes-H2b §6, WORLDS_001 §6 |
| Evidence influence class | Bayes-H2 §7, Bayes-H2b §7 |
| Propagation eligibility (up / down / lateral) | Bayes-H2b §8 |
| Inclusion / exclusion | Bayes-H2 R1–R10, Bayes-H2b §9–§12 |
| Conflict detection | Bayes-H2b §9 |
| Stale evidence | Bayes-H2 §11, WORLDS STALE |
| Missing SE | Bayes-H2 §10, WORLDS MISSING-SE |
| Estimand alignment | Bayes-H2 §12, WORLDS ESTIMAND-EXCLUDE |
| TrustReport obligations | Bayes-H2b §13, WORLDS_001 §9 |
| Release-gate **implications** (readiness flags on report) | WORLDS_001 §10, §15 |

### 4.3 Explicit non-responsibility

- No MCMC, no likelihood evaluation, no prior sampling  
- No Δμ numeric recovery as pass/fail (optional `decision_truth` ignored by default)  
- No comparison to `media_truth.coefficients`  

---

## 5. Validator inputs

### 5.1 Invocation

```text
hierarchy_evidence_validator.validate(
  bundle_path: Path,
  fixture_path: Path | None = None,  # default: bundle_path / "hierarchy_evidence_fixture.json"
  policy: ValidatorPolicy | None = None,
) -> HierarchyEvidenceReport
```

### 5.2 Required bundle files

| File | Required | Role |
|------|----------|------|
| `hierarchy_evidence_fixture.json` | **Yes** | Expected routing/propagation/TR (this contract) |
| `calibration_signals.json` | **Yes** | CalibrationSignal array (ingress) |
| `hierarchy_spec.json` | **Yes** (or embedded in fixture) | Scope graph |
| `world_truth.json` | Recommended | `experiment_truth` linkage, geo panel |
| `estimand_allowlist.json` | **Yes** | Model allowlist for estimand gate |
| `train_config.yaml` | Optional | Ridge fingerprint smoke only |

### 5.3 `hierarchy_evidence_fixture.json` (normative shape)

One file per world — **expected outcomes** for diff against validator output.

```json
{
  "fixture_version": "bayes_h2b_fixture_v1",
  "world_id": "WORLD-BAYES-GEOX-LOCAL",
  "hierarchy_spec_ref": "hierarchy_spec.json",
  "calibration_signals_ref": "calibration_signals.json",
  "policy_overrides": {
    "conflict_tolerance": 0.15,
    "conflict_fail_closed": false,
    "coverage_gate_state": 0.60,
    "coverage_gate_region": 0.60,
    "coverage_gate_national": 0.80,
    "stale_precision_multiplier": 0.25,
    "governance_max_omega": 100.0
  },
  "expected_routing": [],
  "expected_influence_class": [],
  "expected_propagation": [],
  "expected_inclusion_exclusion": [],
  "expected_conflicts": [],
  "expected_trust_report_fields": {},
  "expected_release_gate_effect": {},
  "expected_failures": [],
  "mandatory_assertions": ["MA-GEOX-01", "MA-GEOX-02"]
}
```

### 5.4 `hierarchy_spec.json`

```json
{
  "scope_graph_version": "bayes_h2b_scope_graph_v1",
  "nodes": [
    {"scope_type": "national", "scope_id": "US", "parent_scope_ids": [], "child_scope_ids": ["west", "east"]}
  ],
  "edges": [
    {"parent": "US", "child": "west", "edge_type": "geo_hierarchy"}
  ],
  "sparse_geo_ids": ["phoenix", "denver"],
  "panel_geo_ids": ["sf", "la", "houston", "dallas", "phoenix", "denver"]
}
```

Shared graph: [BAYES_H2B_VALIDATION_WORLDS_001.md §5](BAYES_H2B_VALIDATION_WORLDS_001.md).

### 5.5 `calibration_signals.json`

Per [BAYES_H2B_VALIDATION_WORLDS_001.md §6](BAYES_H2B_VALIDATION_WORLDS_001.md). Validator **rejects** bundle if any evidence object lacks `signal_id` or `scope_type`.

### 5.6 `estimand_allowlist.json`

```json
{
  "allowlist_version": "bayes_h2b_estimand_v1",
  "estimand_ids": [
    "ATT_dma_week_search",
    "national_incremental_lift_search"
  ],
  "lift_scales": ["mean_kpi_level_delta"],
  "segment_bridge_required": true
}
```

World **ESTIMAND-EXCLUDE** uses fixture allowlist **without** `product_conversion_lift`.

### 5.7 Forbidden inputs

| Input | Reason |
|-------|--------|
| Posterior samples / draws | VAL-BAYES-011 |
| Fitted coef vectors as evidence targets | VAL-BAYES-010 |
| `BayesianExperimentAPI` payloads | VAL-BAYES-001 |
| Optimizer allocation vectors | VAL-BAYES-012 |
| Non–CalibrationSignal experiment JSON | VAL-BAYES-001 |

---

## 6. Validator outputs

### 6.1 Primary artifact

**Path:** `validation/reports/<world_id>/hierarchy_evidence_report.json`

### 6.2 Normative report shape

```json
{
  "report_version": "hierarchy_evidence_report_v1",
  "world_id": "WORLD-BAYES-GEOX-LOCAL",
  "validator_version": "hierarchy_evidence_validator_v0.0.0",
  "status": "pass",
  "assertion_results": [],
  "routing_results": [],
  "propagation_results": [],
  "inclusion_exclusion_results": [],
  "conflict_results": [],
  "trust_report_results": {},
  "release_gate_results": {},
  "failure_reasons": [],
  "warnings": [],
  "hierarchy_evidence": {}
}
```

### 6.3 Field semantics

| Field | Type | Description |
|-------|------|-------------|
| `status` | `pass` \| `fail` \| `blocked` | See §10 |
| `assertion_results` | array | One row per MA-* / VAL-BAYES assertion |
| `routing_results` | array | Per-signal mechanism + rule_id (R1–R10) |
| `propagation_results` | array | Per-edge gate outcome |
| `inclusion_exclusion_results` | array | Per-signal influence + decision_grade flag |
| `conflict_results` | array | Conflict groups + metrics |
| `trust_report_results` | object | Subset of `hierarchy_evidence` + presence checks |
| `release_gate_results` | object | Readiness implications only — not prod approval |
| `failure_reasons` | string[] | Machine-readable on fail/blocked |
| `warnings` | string[] | Non-fatal (OA-* optional assertions) |
| `hierarchy_evidence` | object | Full TrustReport-shaped block per Bayes-H2b §13 |

### 6.4 `assertion_results[]` row

```json
{
  "assertion_id": "MA-GEOX-01",
  "validation_id": "VAL-BAYES-005",
  "outcome": "pass",
  "message": "",
  "observed": {},
  "expected": {}
}
```

### 6.5 Determinism requirement

Same bundle + fixture + policy → **byte-identical** `hierarchy_evidence` (excluding timestamps). Required for `VAL-BAYES-H2B-SMOKE`.

---

## 7. Fixture structure

### 7.1 Per-world bundle layout

```text
validation/worlds/<world_id>/
├── hierarchy_evidence_fixture.json   # expected_* (normative for test)
├── hierarchy_spec.json
├── calibration_signals.json
├── estimand_allowlist.json
├── world_truth.json                  # optional sections for materialize
├── experiment_truth.json             # optional alias section
└── train_config.yaml                 # optional
```

### 7.2 Sub-structure reference tables

#### `expected_routing[]`

```json
{
  "signal_id": "SIG-GEOX-SF",
  "mechanism": "likelihood_term",
  "rule_id": "R6",
  "ingress": "calibration_signal"
}
```

#### `expected_influence_class[]`

```json
{
  "signal_id": "SIG-GEOX-SF",
  "influence_class": "local_likelihood_style",
  "target": {"scope_type": "dma", "scope_id": "sf", "channel_ids": ["search"]}
}
```

#### `expected_propagation[]`

```json
{
  "edge": "sf→CA",
  "direction": "up",
  "gate": "coverage_ratio",
  "gate_value": 0.33,
  "threshold": 0.60,
  "outcome": "blocked",
  "allowed_influence": "parent_summary_of_child_evidence"
}
```

#### `expected_inclusion_exclusion[]`

```json
{
  "signal_id": "SIG-GEOX-SF",
  "included": true,
  "decision_grade": true,
  "exclusion_reason": null
}
```

#### `expected_conflicts[]`

```json
{
  "group_id": "CNF-SEARCH-01",
  "signal_ids": ["SIG-GEOX-HOU", "SIG-CLS-NAT"],
  "conflict_metric": 0.75,
  "silent_average": false,
  "sensitivity_required": true
}
```

#### `expected_trust_report_fields`

```json
{
  "included_signals": ["SIG-GEOX-SF"],
  "stale_signals": [],
  "missing_se_signals": [],
  "estimand_excluded_signals": [],
  "conflicting_signals": [],
  "claim_levels": [
    {"scope_type": "dma", "scope_id": "sf", "channel_id": "search", "claim_level": "directly_observed_experimental_evidence"}
  ],
  "lateral_borrowing": "none",
  "propagation_path_min_edges": 1
}
```

#### `expected_release_gate_effect`

```json
{
  "prod_decisioning_allowed": false,
  "decision_safe": true,
  "optimization_blocked": false,
  "release_gate_recommendation": "conditional_not_approved",
  "attribution_safe": true
}
```

#### `expected_failures[]`

List of **anti-pattern IDs** that must **not** appear in validator output (negative assertions). Empty when world is positive-only.

```json
{"forbidden_pattern": "national_mu_point_mass_from_dma", "signal_ids": ["SIG-GEOX-SF"]}
```

---

## 8. World catalog integration

### 8.1 Catalog row fields (unchanged from WORLDS_001)

Each `WORLD-BAYES-*` row in `validation/world_catalog.index.json` must include:

```json
"expected_capabilities": [
  "VAL-BAYES-001", "VAL-BAYES-002", "VAL-BAYES-003", "VAL-BAYES-004",
  "VAL-BAYES-005", "VAL-BAYES-006", "VAL-BAYES-007", "VAL-BAYES-008",
  "VAL-BAYES-009", "VAL-BAYES-010", "VAL-BAYES-011", "VAL-BAYES-012",
  "VAL-BAYES-H2B-SMOKE"
],
"intended_certifications": ["BayesHierarchyEvidenceCertification"]
```

### 8.2 Certification module (planned)

**`BayesHierarchyEvidenceCertification`**

1. Resolve `bundle_path` from catalog.  
2. Run `hierarchy_evidence_validator.validate()`.  
3. Assert `status == pass` for positive worlds.  
4. Emit certification report referencing `validation_id` rows.

### 8.3 Registry alignment note

[BAYES_H2B_VALIDATION_WORLDS_001.md §4.4](BAYES_H2B_VALIDATION_WORLDS_001.md) used an earlier VAL-BAYES-001–008 numbering. **This document is canonical** for runner/registry. Update `validation_registry.md` when implementing stubs.

---

## 9. Validation assertions

### 9.1 Global assertion groups (`VAL-BAYES-001` – `012`)

| `validation_id` | Title | Validator check |
|-----------------|-------|-----------------|
| **VAL-BAYES-001** | CalibrationSignal-only ingress | Every evidence item has `signal_id`; sourced from `calibration_signals.json` only; no parallel experiment API objects |
| **VAL-BAYES-002** | Scope mapping correctness | `scope_type` + `scope_id` resolve to node in `hierarchy_spec`; targets match WORLDS_001 §6 table |
| **VAL-BAYES-003** | Influence-class correctness | Assigned `influence_class` matches fixture `expected_influence_class` |
| **VAL-BAYES-004** | Propagation eligibility | Each `expected_propagation` edge outcome matches; gates applied per ADR thresholds |
| **VAL-BAYES-005** | Inclusion/exclusion correctness | `included`, `decision_grade`, `exclusion_reason` per `expected_inclusion_exclusion` |
| **VAL-BAYES-006** | Conflict surfacing | Conflict groups match `expected_conflicts`; no silent merge |
| **VAL-BAYES-007** | TrustReport visibility | Required keys in §11 present; no silent drop of excluded signals |
| **VAL-BAYES-008** | Release-gate implications | `release_gate_results` match `expected_release_gate_effect` |
| **VAL-BAYES-009** | No silent averaging | No `merged_lift`, `averaged_lift`, or blended precision across conflict group |
| **VAL-BAYES-010** | No model-fit dependency | Validator did not read `fitted_coefs`, `posterior_summary`, or train artifacts |
| **VAL-BAYES-011** | No posterior dependency | Report contains no posterior samples, ESS, R-hat, or credible intervals as evidence inputs |
| **VAL-BAYES-012** | No alternate decision surface | No `BayesianDecisionSurface`, optimizer vectors, or allocation outputs in report |

### 9.2 Mapping MA-* → VAL-BAYES

| MA-* family | Primary VAL-BAYES |
|-------------|-------------------|
| Routing / mechanism | 001, 003 |
| Propagation | 004 |
| Inclusion | 005 |
| Conflict | 006, 009 |
| TrustReport fields | 007 |
| Release gate | 008 |
| Stale / missing SE / estimand | 005, 007 (+ world-specific MA) |

---

## 10. Pass/fail semantics

| `status` | Condition |
|----------|-----------|
| **pass** | All `mandatory_assertions` in fixture pass; VAL-BAYES-001–012 pass; no `expected_failures` triggered |
| **fail** | Any mandatory assertion fails; or forbidden pattern detected |
| **blocked** | Fixture unloadable; missing `calibration_signals.json`; validator internal error; policy version mismatch |

**WARN:** Optional assertions (`OA-*`) populate `warnings[]` but do not change `status` unless fixture sets `fail_on_warnings: true`.

**CI smoke:** Suite **pass** only if all seven worlds return `status: pass`.

---

## 11. TrustReport validation contract

Validator must emit `hierarchy_evidence` containing at minimum:

| Key | Required when |
|-----|----------------|
| `hierarchy_scope_map` | Always |
| `included_signals` / `excluded_signals` | Always |
| `stale_signals` | STALE world + any stale tier input |
| `missing_se_signals` | MISSING-SE world + null SE inputs |
| `estimand_excluded_signals` | ESTIMAND-EXCLUDE world |
| `conflicting_signals` | CONFLICT world |
| `propagated_evidence` or `propagation_path` | All |
| `claim_level` per affected scope/channel | All |
| `borrowed_strength_sources` | CLS-NATIONAL, SPARSE-GEO |
| `lateral_borrowing` | All |
| `unsupported_claims` | CLS-NATIONAL |
| `local_vs_national_alignment` | CONFLICT, CLS-NATIONAL |
| `pooling_diagnostics` | SPARSE-GEO |
| `signal_weight_summary` | STALE, MISSING-SE |

**VAL-BAYES-007 pass rule:** Every key listed in fixture `expected_trust_report_fields` is present with matching set equality (order-independent for arrays).

**Bayes-H2 fields** (`included_signals` at top level) may duplicate hierarchy block — validator must keep consistent.

---

## 12. Release-gate validation contract

Validator computes **implications** only — does not call promotion or `approved_for_prod`.

| Field | Type | Source logic |
|-------|------|--------------|
| `prod_decisioning_allowed` | bool | Always `false` for Bayes research path |
| `decision_safe` | bool | `false` if fail-closed conflict or missing decision-grade evidence when required |
| `optimization_blocked` | bool | `true` if `conflict_fail_closed` and conflict unresolved |
| `attribution_safe` | bool | `false` if any `claim_level` mislabel detected |
| `release_gate_recommendation` | enum | `block` \| `warn` \| `conditional_not_approved` per [trust_report_semantics.md](05_validation/trust_report_semantics.md) |

**VAL-BAYES-008:** Compare to fixture `expected_release_gate_effect` exactly for listed keys.

---

## 13. Cross-world invariant checks

Executed on **full suite** (`VAL-BAYES-H2B-SMOKE`):

| ID | Invariant |
|----|-----------|
| X-01 | Seven bundles load without error |
| X-02 | All reports `validator_version` compatible |
| X-03 | VAL-BAYES-010/011 pass on every world |
| X-04 | No report contains `posterior_draws` or `fitted_beta` |
| X-05 | ESTIMAND-EXCLUDE ⊂ excluded decision-grade globally |
| X-06 | MISSING-SE world has no decision-grade missing-SE signals |
| X-07 | CONFLICT world has `silent_average: false` in conflict_results |
| X-08 | Reports deterministic across two consecutive runs |

---

## 14. Failure taxonomy

| Code | Category | Example |
|------|----------|---------|
| `E-FIXTURE-001` | Fixture | Missing `hierarchy_evidence_fixture.json` |
| `E-FIXTURE-002` | Fixture | `world_id` mismatch bundle vs fixture |
| `E-INGRESS-001` | VAL-BAYES-001 | Non-CalibrationSignal path |
| `E-SCOPE-001` | VAL-BAYES-002 | Unresolved `scope_id` |
| `E-INFLUENCE-001` | VAL-BAYES-003 | Wrong influence class |
| `E-PROP-001` | VAL-BAYES-004 | Upward gate should block but passed |
| `E-INCL-001` | VAL-BAYES-005 | Excluded signal in decision-grade set |
| `E-CONF-001` | VAL-BAYES-006 | Missing conflict group |
| `E-CONF-002` | VAL-BAYES-009 | Silent average detected |
| `E-TR-001` | VAL-BAYES-007 | Missing TrustReport key |
| `E-GATE-001` | VAL-BAYES-008 | Release implication mismatch |
| `E-MODEL-001` | VAL-BAYES-010 | Train artifact read detected |
| `E-POST-001` | VAL-BAYES-011 | Posterior field in report |
| `E-DECIDE-001` | VAL-BAYES-012 | Alternate decision object in report |
| `E-STALE-001` | World STALE | Stale at full precision |
| `E-MSE-001` | World MISSING-SE | decision_grade true without SE |
| `E-EST-001` | World ESTIMAND | Included misaligned estimand |

---

## 15. CI smoke-test plan

### 15.1 Target ID

**`VAL-BAYES-H2B-SMOKE`**

### 15.2 Scope

| Step | Action |
|------|--------|
| 1 | Discover seven catalog rows / bundle paths |
| 2 | For each world: `hierarchy_evidence_validator.validate(bundle_path)` |
| 3 | Assert `status == pass` |
| 4 | Assert VAL-BAYES-001–012 each `pass` in `assertion_results` |
| 5 | Run cross-world invariants §13 |
| 6 | Write `validation/reports/bayes_h2b_smoke_summary.json` |

### 15.3 CI tier

| Tier | When |
|------|------|
| `smoke` | PR CI — required after validator stub lands |
| Duration budget | &lt; 30s total (no train) |

### 15.4 Failure policy

Any world `fail` or `blocked` → job **fail**. No retry without fixture change.

### 15.5 Entry command (planned)

```bash
python -m mmm.validation.synthetic.hierarchy_evidence_validator --smoke VAL-BAYES-H2B-SMOKE
```

*Not implemented in this deliverable.*

---

## 16. Implementation-readiness checklist

| # | Task | Status |
|---|------|--------|
| 1 | Accept `BAYES_H2B_VALIDATION_RUNNER_002` | ✅ This document |
| 2 | Add VAL-BAYES-001–012 + VAL-BAYES-H2B-SMOKE to `validation_registry.md` | Pending |
| 3 | Author seven `hierarchy_evidence_fixture.json` files | ✅ Materialized |
| 4 | Author shared `hierarchy_spec.json` (copy per bundle) | ✅ Materialized |
| 5 | Implement `hierarchy_evidence_validator` stub | Pending |
| 6 | Implement fixture diff / MA-* assertion engine | Pending |
| 7 | Wire `VAL-BAYES-H2B-SMOKE` in CI | Pending |
| 8 | Materialize `world_catalog.index.json` rows | ✅ Done |
| 9 | Bayes-H2d model spec ADR | **Blocked** until step 7 passes |
| 10 | Bayes-H3 | **Blocked** until step 9 |

---

## 17. Anti-patterns

| Anti-pattern | Blocked by |
|--------------|------------|
| Running PyMC inside validator | §3, VAL-BAYES-010/011 |
| Using coef recovery as pass/fail | §3 |
| Reading train artifact for evidence weights | VAL-BAYES-010 |
| Silent conflict averaging | VAL-BAYES-009 |
| Missing TrustReport entry for excluded signal | VAL-BAYES-007 |
| experiment_api.json sidecar | VAL-BAYES-001 |
| `optimize()` output in validator | VAL-BAYES-012 |
| Skipping fixture diff (validator-only golden) | Determinism §6.5 — forbidden for CI |
| Authorizing Bayes-H2d/H3 in this doc | §16 |

---

## 18. Open questions

| ID | Question | Phase |
|----|----------|-------|
| OQ-R01 | Single shared `hierarchy_spec.json` vs per-bundle copy | Materialize |
| OQ-R02 | `conflict_fail_closed` as fixture flag vs separate world variant | ESTIMAND / CONFLICT v1.1 |
| OQ-R03 | Whether validator lives in `certification_runner.py` or standalone module | Implement |
| OQ-R04 | Golden report commit policy (yes/no in repo) | CI policy |

---

## 19. Final recommendation

1. **Accept** this runner contract as the specification for `hierarchy_evidence_validator`.  
2. **Next:** Materialize seven fixture bundles per §7 + [BAYES_H2B_VALIDATION_WORLDS_001.md](BAYES_H2B_VALIDATION_WORLDS_001.md).  
3. **Then:** Implement validator **stub** + `VAL-BAYES-H2B-SMOKE` CI.  
4. **Only then:** Bayes-H2d hierarchical model spec ADR (architecture only).  
5. **Bayes-H3** remains blocked until H2d accepted and Bayes-H4 worlds defined separately.

This sequence forces Bayesian work through **reliability scaffolding** before modeling design — preventing an ungoverned MMM prototype.

**This document does not authorize Bayes-H2d, Bayes-H3, or any model implementation.**

---

## Appendix A — Per-world fixture expectations and core assertions

### A.1 WORLD-BAYES-GEOX-LOCAL

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | `SIG-GEOX-SF` — geox, dma/sf, lift 0.06, SE 0.015, fresh |
| `expected_influence_class` | `local_likelihood_style` on sf×search |
| `expected_propagation` | sf→CA **blocked** (coverage 0.33 &lt; 0.60) |
| `expected_trust_report_fields` | `included_signals: [SIG-GEOX-SF]`; no national direct claim |
| `expected_release_gate_effect` | `prod_decisioning_allowed: false` |
| `mandatory_assertions` | MA-GEOX-01 … MA-GEOX-05 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-GEOX-01 | 005 |
| MA-GEOX-02 | 004 |
| MA-GEOX-03 | 004, 007 |
| MA-GEOX-04 | 003, 007 |
| MA-GEOX-05 | 004, 008 |

---

### A.2 WORLD-BAYES-CLS-NATIONAL

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | `SIG-CLS-NAT` — cls, national/US, search +3% |
| `expected_influence_class` | `national_calibration_evidence` on μ_c |
| `expected_propagation` | US→each DMA: shrinkage only, `borrowed_strength_from_parent: true` |
| `expected_trust_report_fields` | `unsupported_claims` includes DMA causal CLS string |
| `mandatory_assertions` | MA-CLS-01 … MA-CLS-05 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-CLS-02 | 007 |
| MA-CLS-03 | 003, 007 |
| MA-CLS-05 | 004, 005 |

---

### A.3 WORLD-BAYES-CONFLICT

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | `SIG-GEOX-HOU` +8%; `SIG-CLS-NAT` +2% |
| `expected_conflicts` | Group with both signals; `silent_average: false`; `sensitivity_required: true` |
| `expected_failures` | Forbid `merged_lift` |
| `mandatory_assertions` | MA-CNF-01 … MA-CNF-05 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-CNF-01 | 006, 007 |
| MA-CNF-03 | 009 |
| MA-CNF-04/05 | 003, 004 |

---

### A.4 WORLD-BAYES-STALE

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | Fresh `SIG-GEOX-LA`; stale `SIG-CLS-STALE` (tier stale, age 200d) |
| `expected_inclusion_exclusion` | Stale: included false OR decision_grade false with downweight |
| `expected_trust_report_fields` | `stale_signals` contains CLS stale id |
| `mandatory_assertions` | MA-STL-01 … MA-STL-04 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-STL-01 | 007 |
| MA-STL-02 | 005 |

*Note: Map freshness to VAL-BAYES-005 inclusion rules; registry row `VAL-BAYES-004` in WORLDS_001 renamed to propagation in this contract.*

---

### A.5 WORLD-BAYES-MISSING-SE

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | `SIG-GEOX-NOSE` lift_se null; optional `SIG-GEOX-CI` with CI only |
| `expected_inclusion_exclusion` | NOSE: decision_grade false |
| `expected_trust_report_fields` | `missing_se_signals: [SIG-GEOX-NOSE]` |
| `mandatory_assertions` | MA-MSE-01 … MA-MSE-04 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-MSE-01 | 005 |
| MA-MSE-02 | 007 |
| MA-MSE-03 | 009 (no inflated precision) |

---

### A.6 WORLD-BAYES-SPARSE-GEO

| Fixture section | Expectation |
|-----------------|-------------|
| `hierarchy_spec.sparse_geo_ids` | phoenix, denver |
| `expected_trust_report_fields` | claim_levels: sf direct; phoenix/denver borrowed |
| `expected_propagation` | No edge sf→phoenix mechanism |
| `mandatory_assertions` | MA-SPR-01 … MA-SPR-05 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-SPR-05 | 004, 009 (no lateral unsupported calib) |

---

### A.7 WORLD-BAYES-ESTIMAND-EXCLUDE

| Fixture section | Expectation |
|-----------------|-------------|
| `calibration_signals` | `SIG-AB-PROD` segment, estimand `product_conversion_lift` |
| `estimand_allowlist` | **Excludes** product_conversion_lift |
| `expected_inclusion_exclusion` | included false |
| `expected_trust_report_fields` | `estimand_excluded_signals: [SIG-AB-PROD]` |
| `mandatory_assertions` | MA-EST-01 … MA-EST-04 |

| Core assertions | VAL-BAYES |
|-----------------|-----------|
| MA-EST-01 | 005, 006 (estimand gate) |
| MA-EST-03 | 004 (empty propagation path) |

---

## Appendix B — Example minimal `hierarchy_evidence_fixture.json` (GEOX-LOCAL)

```json
{
  "fixture_version": "bayes_h2b_fixture_v1",
  "world_id": "WORLD-BAYES-GEOX-LOCAL",
  "hierarchy_spec_ref": "hierarchy_spec.json",
  "calibration_signals_ref": "calibration_signals.json",
  "expected_routing": [
    {"signal_id": "SIG-GEOX-SF", "mechanism": "likelihood_term", "rule_id": "R6", "ingress": "calibration_signal"}
  ],
  "expected_influence_class": [
    {"signal_id": "SIG-GEOX-SF", "influence_class": "local_likelihood_style", "target": {"scope_type": "dma", "scope_id": "sf", "channel_ids": ["search"]}}
  ],
  "expected_propagation": [
    {"edge": "sf→CA", "direction": "up", "gate": "coverage_ratio", "gate_value": 0.33, "threshold": 0.60, "outcome": "blocked", "allowed_influence": "parent_summary_of_child_evidence"}
  ],
  "expected_inclusion_exclusion": [
    {"signal_id": "SIG-GEOX-SF", "included": true, "decision_grade": true, "exclusion_reason": null}
  ],
  "expected_conflicts": [],
  "expected_trust_report_fields": {
    "included_signals": ["SIG-GEOX-SF"],
    "lateral_borrowing": "none",
    "claim_levels": [
      {"scope_type": "dma", "scope_id": "sf", "channel_id": "search", "claim_level": "directly_observed_experimental_evidence"}
    ]
  },
  "expected_release_gate_effect": {
    "prod_decisioning_allowed": false,
    "decision_safe": true,
    "optimization_blocked": false,
    "release_gate_recommendation": "conditional_not_approved"
  },
  "expected_failures": [
    {"forbidden_pattern": "national_mu_point_mass_from_dma", "signal_ids": ["SIG-GEOX-SF"]}
  ],
  "mandatory_assertions": ["MA-GEOX-01", "MA-GEOX-02", "MA-GEOX-03", "MA-GEOX-04", "MA-GEOX-05"]
}
```

---

## References

- [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)  
- [BAYES_H2B_VALIDATION_WORLDS_001.md](BAYES_H2B_VALIDATION_WORLDS_001.md)  
- [bayes_h2_calibration_signal_mapping_adr.md](05_validation/bayes_h2_calibration_signal_mapping_adr.md)  
- [bayes_h1_decision_surface_preservation_adr.md](05_validation/bayes_h1_decision_surface_preservation_adr.md)
