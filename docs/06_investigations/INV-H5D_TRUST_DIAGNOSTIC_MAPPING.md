# INV-H5D — Bayes-H5 TrustReport Diagnostic Mapping (Research Only)

**Investigation ID:** INV-H5D  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5c (`60ff381`) extended MCMC confirmation  
**Code:** `mmm/research/bayes_h3_sandbox/h5_trust_diagnostics.py`  
**Source:** [BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json](../05_validation/archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json)  
**Artifact:** [BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json](../05_validation/archives/BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json)

---

## 1. Purpose

Define a **research-only** TrustReport-shaped diagnostic payload for Bayes-H5 sandbox outputs. This is **not** production TrustReport integration. It specifies candidate fields, warning taxonomy, and artifact schema that a future production TrustReport could consume **only after separate promotion gates**.

---

## 2. Stable candidate fields (research evidence)

| Field | Stability | Notes |
|-------|-----------|--------|
| `transform_alignment_status` | High | Matches world design; 0% unexpected mismatch post-H5b |
| `transform_mismatch_detected` | High | 100% on mismatch worlds across H5b/H5c |
| `warning_codes` (taxonomy) | High | Deterministic from world role + aggregate rates |
| `beta_gc_mae_mean` / `mu_c_mae_mean` | Medium | Stable across seeds; toy panels only |
| `weak_identification_status` | High | Collinearity / weak-signal tags reliable |
| `sparse_recovery_status` | High | Always `report_only` on sparse world |
| `h5_classification` | High | Frozen in validation world catalog |
| `policy_outcome` | Medium | INV-071 report-only; not a gate |

---

## 3. Deterministic / reliable warnings

| Code | When emitted |
|------|----------------|
| `h5:transform_mismatch:adstock` | ADSTOCK-MISMATCH + mismatch rate ≥ 99% |
| `h5:transform_mismatch:saturation` | SATURATION-MISMATCH + mismatch rate ≥ 99% |
| `h5:weak_identification:collinearity` | CORRELATED-CHANNELS + collinearity rate ≥ 99% |
| `h5:weak_identification:weak_signal_generative` | WEAK-SIGNAL + weak-ID rate ≥ 99% |
| `h5:sparse_recovery:report_only` | SPARSE-RECOVERY always |
| `h5:recovery_candidate:stable_research_only` | Aligned recovery worlds (not sparse) |
| `h5:production:block` | **Every** world payload (research lane) |

---

## 4. Report-only fields (must not become hard gates)

- `beta_gc_mae` / `mu_c_mae` vs H4c baselines  
- `shrinkage_ratio_sparse` (pooling mechanics; not true-effect gate)  
- `policy_outcome` from INV-071  
- All `recovery_metric_summary` aggregates  
- `recommended_interpretation` text  

---

## 5. Explicitly not authorized

| Item | Status |
|------|--------|
| Production TrustReport wiring | **Not authorized** |
| Optimizer / DecisionSurface / recommendations | **Blocked** |
| `approved_for_prod` / `prod_decisioning_allowed` | **false** |
| INV-071 hard gates | **Not authorized** |
| Production Bayes promotion | **Blocked** |
| Ridge replacement | **Blocked** |

---

## 6. Evidence required before production TrustReport integration

1. **Promotion Gate** ADR accepting H5 spec on real panels (not toy worlds only).  
2. Repeated pilots on **client-representative** data with extended MCMC.  
3. TrustReport contract ADR mapping candidate fields → production schema with fail-closed behavior.  
4. Shadow mode: H5 diagnostic TrustReport side-by-side Ridge prod TrustReport **without** optimizer consumption.  
5. Explicit sign-off that warning codes are calibrated for false-positive rate on prod traffic.  

---

## 7. Per-world mapping summary (from H5c source)

| World | Alignment | Weak-ID | Key warning codes |
|-------|-----------|---------|-------------------|
| ADSTOCK-ALIGNED | aligned | none | stable_research_only, production:block |
| SATURATION-ALIGNED | aligned | none | stable_research_only, production:block |
| ADSTOCK-MISMATCH | intentional_mismatch | none | transform_mismatch:adstock, production:block |
| SATURATION-MISMATCH | intentional_mismatch | none | transform_mismatch:saturation, production:block |
| CORRELATED-CHANNELS | aligned | collinearity | weak_identification:collinearity, production:block |
| WEAK-SIGNAL | aligned | weak_signal | weak_identification:weak_signal_generative, production:block |
| SPARSE-RECOVERY | aligned | none | sparse_recovery:report_only, production:block |

---

## 8. Production impact

**None.** Mapping artifact and all payloads carry `hard_gate=false`, `approved_for_prod=false`, `prod_decisioning_allowed=false`.
