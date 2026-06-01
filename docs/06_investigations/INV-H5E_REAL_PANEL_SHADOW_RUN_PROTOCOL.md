# INV-H5E — Bayes-H5 Real-Panel Shadow-Run Protocol (Research Only)

**Investigation ID:** INV-H5E  
**Status:** **Protocol defined** (no authorized shadow execution in this deliverable)  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5c (`60ff381`) · Bayes-H5d (`bcebfc8`) TrustReport candidate mapping  
**Schema:** [BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json](../05_validation/archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json)  
**Code:** `mmm/research/bayes_h3_sandbox/h5_shadow_protocol.py`

---

## 1. Purpose

Define how to run **Bayes-H5** (`bayes_h5_sandbox_spec_v1`) on **historical MMM panels** in a **shadow** lane:

- Compare diagnostic posteriors and H5d TrustReport **candidate** fields against **Ridge production** fits on the same snapshot.
- Optionally cross-reference **historical GeoX / CLS / experiment evidence** when documented as `CalibrationSignal` or replay metadata.
- **Never** feed H5 outputs into optimizer, DecisionSurface, production TrustReport, or budget recommendations.

This protocol is **design-only** in H5e. Executing shadow runs requires a separate authorization and ops checklist.

---

## 2. Eligible input panels

| Criterion | Requirement |
|-----------|-------------|
| **Environment** | `run_environment=research` only |
| **History** | Client MMM panels already used for Ridge prod or validation (not synthetic H4/H5 worlds) |
| **Grain** | Geo × week panel matching `PanelSchema` |
| **QA** | Passes `validate_panel` + integrity QA at **report** severity (blocks excluded unless waived in research) |
| **Lineage** | Immutable `dataset_snapshot_id` with content hash |
| **Exclusions** | Stress-only SPARSE-GEO worlds, toy recovery worlds, production decisioning panels tied to live optimizer loops |

---

## 3. Required data fields

| Field group | Columns / objects |
|-------------|-------------------|
| **Panel** | `geo_id`, `week`, target `y`, channel spend/exposure columns, optional controls |
| **Config** | `MMMConfig` with `framework=bayesian`, `run_environment=research`, `pooling=partial` |
| **Transforms** | Explicit `transform_config` per H5 registry (not prod FE pipeline) |
| **Hierarchy** | `geo_hierarchy_mapping` when sparse/partial pooling geos exist |
| **Calibration** | `calibration_signals_stub` list (stubs in H5e v1 — no integrated likelihood) |

---

## 4. Required lineage fields

Every shadow-run artifact **must** include:

| Field | Description |
|-------|-------------|
| `run_id` | Unique shadow run identifier |
| `dataset_snapshot_id` | Immutable data snapshot reference |
| `panel_id` | Logical panel name / client slice |
| `data_snapshot_hash` | Hash of panel bytes at run time |
| `mmm_config_hash` | Hash of serialized config |
| `run_environment` | Must be `research` |
| `sandbox_entrypoint` | `mmm.research.bayes_h3_sandbox.run_sandbox_fit` |
| `model_spec_version` | `bayes_h5_sandbox_spec_v1` |
| `enable_h5_sandbox` | `true` |
| `research_only` | `true` |

---

## 5. Transform configuration policy

1. **Declare** `media_transforms_by_channel` explicitly (identity, geometric_adstock, hill_saturation, adstock_then_saturation).
2. **Do not** silently reuse production feature-engineering transforms without documentation.
3. Record `transform_mismatch_mode`: `aligned` (client spec match) vs `intentional_mismatch` (diagnostic probe only).
4. Document `transform_params_by_channel` (e.g. adstock decay) in artifact.
5. Map outputs through H5d `trust_report_candidate_diagnostics` after fit.

---

## 6. Calibration-signal handling

| Rule | H5e v1 |
|------|--------|
| **Ingress** | `CalibrationSignal` metadata / stubs only |
| **Likelihood** | `likelihood_integrated=false` |
| **GeoX/CLS** | Reference as external experiment evidence IDs in `geox_cls_comparison`; no GeoX estimator promotion |
| **Conflicts** | Record in `calibration_signal_summary.conflict_warnings` (diagnostic) |

---

## 7. Output artifact schema

See [BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json](../05_validation/archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json).

Per-run record includes:

- Lineage block  
- `transform_config`  
- `posterior_diagnostics` (convergence, pooling — diagnostic only)  
- `trust_report_candidate_diagnostics` (H5d mapping)  
- `ridge_comparison` (diagnostic diff vs Ridge on same snapshot)  
- `geox_cls_comparison` (optional)  
- `excluded_fields` list  
- `production_flags` all false  

---

## 8. Comparison against Ridge production path

| Aspect | Shadow rule |
|--------|-------------|
| **Ridge fit** | Run prod Ridge estimator on **same** `dataset_snapshot_id` for contrast only |
| **Optimizer** | Ridge coefs may feed prod optimizer; **H5 posterior must not** |
| **Metrics** | Compare MAPE/residuals, coef stability, channel ranking **directionally** — report-only |
| **Decision grade** | `ridge_comparison.decision_grade=false`, `used_for_optimizer=false` for H5 arm |

---

## 9. Comparison against historical GeoX / CLS evidence

| Aspect | Shadow rule |
|--------|-------------|
| **Availability** | Optional; set `geox_cls_comparison.available=true` when evidence catalog exists |
| **Usage** | Cross-check declared lift directions vs H5 posterior signs — **diagnostic** |
| **Not allowed** | Promote SCM/GeoX intervals as H5 gates; D5-POW null-monitor rules apply separately |
| **Repo boundary** | GeoX production code remains outside MMM; reference investigation IDs only |

---

## 10. Failure modes

| Failure | Response |
|---------|----------|
| Missing lineage | **Fail closed** — do not write shadow artifact |
| Missing `transform_config` | **Fail closed** |
| `enable_h5_sandbox=false` | **Fail closed** |
| H5 fit emits `decision_surface` / optimizer fields | **Abort** — protocol violation |
| NUTS divergences / high R-hat | Record in `posterior_diagnostics`; **no hard fail** in shadow v1 |
| Transform mismatch undeclared | Warn via H5d codes; continue as diagnostic |
| Calibration conflict | Record; report-only |

---

## 11. Stop conditions

Stop the shadow **program** (not individual MCMC) if:

1. Any artifact has `approved_for_prod=true` or `prod_decisioning_allowed=true`.  
2. H5 output is wired to optimizer or production TrustReport.  
3. Repeated shadow runs show systematic transform mismatch without client spec update.  
4. Promotion Gate is not satisfied (see §12).

Individual runs with poor convergence may continue as diagnostic failures.

---

## 12. Promotion boundary

**H5e does not authorize:**

- Production Bayesian MMM  
- Production TrustReport integration  
- INV-071 hard gates on shadow metrics  
- Ridge replacement  

**Evidence required before any promotion:**

1. Completed shadow-run catalog (≥1 client panel, extended MCMC).  
2. Signed Promotion Gate ADR.  
3. TrustReport contract ADR for H5d field mapping.  
4. Shadow vs Ridge disagreement review.  
5. Explicit false-positive rate review on warning taxonomy.

---

## 13. Production impact

**None.** Protocol and schema are research-only. Ridge production path unchanged. Production Bayes remains blocked.
