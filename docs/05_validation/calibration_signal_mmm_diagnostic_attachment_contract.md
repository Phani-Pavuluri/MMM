# CalibrationSignal → MMM Ridge diagnostic attachment contract (MIP-C1)

**Status:** Accepted (contract / context-only)  
**Date:** 2026-06-01  
**Audit:** [AUDIT-MIP-C1](../audits/AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md)  
**Implementation:** `mmm/diagnostics/calibration_signal_attachment.py`

## Purpose

Govern how **CalibrationSignal** records attach to **`ridge_production_diagnostics_report`** as `calibration_evidence_context` without changing Ridge fit, optimizer, DecisionSurface, or recommendations.

Prerequisite: H7–H10 Ridge diagnostic chain ([ridge_production_diagnostics_contract.md](ridge_production_diagnostics_contract.md)).

---

## Attachment block identity

| Field | Value |
|-------|-------|
| `calibration_evidence_context.attachment_version` | `mip_calibration_signal_attachment_v1` |
| `calibration_evidence_context.milestone` | `MIP-C1` |
| `calibration_evidence_context.context_only` | `true` (required) |

Parent report field: `ridge_production_diagnostics_report.calibration_evidence_context`.

---

## Per-signal fields (binding)

Each attached signal **must** expose:

| Field | Type | Description |
|-------|------|-------------|
| `signal_id` | string | Unique evidence id |
| `source_system` | string | e.g. `geox`, `cls`, `ab_platform`, `holdout_registry`, `replay` |
| `source_modality` | string | e.g. `geo_experiment`, `cls_readout`, `ab_test`, `holdout`, `replay_unit` |
| `experiment_id` | string | Experiment id (or use `study_id`) |
| `study_id` | string | Optional study id (CLS) |
| `channel` | string | MMM media channel key |
| `geo_scope` | object | Declared geographic scope (`kind`, `ids`) |
| `time_window` | object | Evidence window (`start`, `end`) |
| `estimand_id` | string | Platform estimand id |
| `measurement_instrument_id` | string | Instrument / export version |
| `lift_scale` | string | e.g. `incremental_roi`, `incremental_sales` |
| `effect_estimate` | number | Point lift / effect |
| `standard_error` | number | Required for calibration-informed Ridge interpretation |
| `interval` | array | Optional `[low, high]` credible interval |
| `freshness_status` | string | `fresh`, `stale`, `expired`, `unknown` |
| `eligibility_status` | string | `eligible`, `inconclusive`, `excluded`, `blocked` |
| `alignment_status` | string | `aligned`, `misaligned`, `inconclusive`, `not_applicable` |
| `conflict_status` | string | `none`, `directional_conflict`, `scope_mismatch`, `trust_report_only` |
| `trust_report_disposition` | string | Routing: `diagnostic_context`, `trust_report_only`, `trust_report_and_human_review`, `context_only_stale` |
| `allowed_use` | string[] | Governed uses for this signal |
| `forbidden_claims` | string[] | Claims blocked when this signal is present |
| `included_in_context` | boolean | false when `eligibility_status=excluded` |

---

## Evaluation rules

### Alignment (`evaluate_signal_alignment`)

- Compare `effect_estimate` sign (or `claimed_direction`) to Ridge `coefficient_stability.media_coef_by_channel[channel]`.  
- Skip alignment when `freshness_status ∈ {stale, expired, unknown}` or estimand mismatch.  
- `inconclusive` when uncertainty missing.

### Conflict (`evaluate_signal_conflict`)

| Condition | `conflict_status` |
|-----------|-------------------|
| `eligibility_status=inconclusive` | `trust_report_only` |
| Estimand ≠ report `world_metadata.estimand_id` | `scope_mismatch` |
| Aligned directions | `none` |
| Misaligned directions | `directional_conflict` |

### TrustReport disposition

| Case | Disposition |
|------|-------------|
| Estimand mismatch or missing SE | `trust_report_only` |
| Stale | `context_only_stale` |
| Directional conflict | `trust_report_and_human_review` |
| Aligned + eligible + fresh + SE | `diagnostic_context` |

---

## Global forbidden claims (attachment block)

Always appended to report `forbidden_claims` when context is attached:

- `automatic_mmm_recalibration_from_calibration_signal`
- `ridge_coefficient_override_from_external_evidence`
- `optimizer_input_from_calibration_signal`
- `decision_surface_input_from_calibration_signal`
- `budget_recommendation_from_calibration_signal`

Per-signal forbidden claims include:

- `external_evidence_overrides_mmm_coefficients`
- `geox_or_cls_silently_overrides_mmm`
- `external_evidence_authorizes_optimizer_use`
- `external_evidence_authorizes_budget_recommendation`
- Plus scenario-specific claims (conflict, stale, sparse channel, uncertainty).

---

## Artifact placement

| Location | Allowed content |
|----------|-----------------|
| `ridge_production_diagnostics_report.calibration_evidence_context` | Full attachment block + `summary` |
| `ridge_production_diagnostics_summary.md` | Short "Calibration evidence context" section (H8) |
| `extension_report.json` | Embedded `ridge_production_diagnostics_report` |
| CLI (`format_ridge_diagnostics_cli_block`) | Headline + conflict ids only |

**Forbidden placements:** Ridge refit input, optimizer input, DecisionSurface input, recommendation payloads, coefficient override fields.

---

## API (context-only)

```python
attach_calibration_evidence_context(report, signals) -> report
evaluate_signal_alignment(signal, report) -> str
evaluate_signal_conflict(signal, report) -> str
build_calibration_evidence_summary(context) -> dict
build_calibration_forbidden_claims(context) -> list[str]
```

`attach_calibration_evidence_context` returns a **deep copy** of the report; it does not mutate fit objects or trainer state.

---

## Train-boundary ingestion (MIP-C2)

| Input | Module |
|-------|--------|
| `ingest_calibration_signals_into_report` | `mmm/diagnostics/calibration_signal_ingestion.py` |
| Extension hook | `extension_runner._attach_ridge_production_diagnostics` |
| CLI | `mmm train --calibration-signals-path` |

Audit: [AUDIT-MIP-C2](../audits/AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md)

## Tests and fixtures

| Fixtures | `tests/fixtures/mip_calibration_signal_attachment/` |
| Ingestion fixtures | `tests/fixtures/mip_calibration_signal_ingestion/` |
| Tests | `tests/mip/test_calibration_signal_mmm_attachment_contract.py` |
| Train boundary tests | `tests/mip/test_calibration_signal_train_boundary_ingestion.py` |

---

## Related

- [Bayes-H2 CalibrationSignal mapping ADR](bayes_h2_calibration_signal_mapping_adr.md)  
- [Bayes-H2b scope rules ADR](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)  
- [trust_report_semantics.md](trust_report_semantics.md)  
- [ridge_production_diagnostics_contract.md](ridge_production_diagnostics_contract.md) — updated with `calibration_evidence_context` reference  
