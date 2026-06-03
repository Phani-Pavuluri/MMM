# AUDIT-MIP-C3 — GeoX/CLS CalibrationSignal Adapter Gate

**Audit ID:** AUDIT-MIP-C3  
**Date:** 2026-06-01  
**Scope:** Offline GeoX/CLS export → CalibrationSignal adapter (no live APIs)  
**Prerequisites:** MIP-C1 @ `df54dd1`, MIP-C2 @ `dd3b36d`, H11 @ `9d20c72`  
**Verdict:** **Pass (adapter contract)** — fixture exports convert to C2 shape and ingest as context only

---

## 1. Purpose

Validate that **real-shaped GeoX/CLS exports** can be converted to the **MIP-C2 JSON shape** before any live API integration. Adapters are export-only; train path unchanged when ETL emits `signals` files.

---

## 2. Where does signal ingestion happen?

| Layer | Responsibility |
|-------|----------------|
| **MIP-C3** | `geox_record_to_calibration_signal` / `cls_record_to_calibration_signal` |
| **MIP-C2** | `ingest_calibration_signals_into_report` at train/extension |
| **MIP-C1** | `attach_calibration_evidence_context` policy |

Live GeoX/CLS APIs are **not** called in MIP-C3.

---

## 3. How is lineage recorded?

Each adapted signal includes `adapter_metadata.source_lineage` (experiment id, export version, readout id).  
Batch exports use `adapt_mixed_batch_export` → `adapter_lineage.geox_count` / `cls_count`.  
C2 `evidence_attachment_lineage` records file ingest at train time (unchanged).

---

## 4. What happens with no signals?

No adapter invocation — operators pass no `--calibration-signals-path` (H11 absent lineage).

---

## 5. What happens with conflicts?

Adapter preserves `effect_estimate` sign and magnitude.  
MIP-C1 evaluates directional conflict vs Ridge coefficients at attach — **MMM not overridden**.

---

## 6. What happens with malformed / incomplete exports?

| Case | Adapter | C2 attach |
|------|---------|-----------|
| Missing SE | `eligibility_status=blocked`, adapter note | `trust_report_only` |
| CLS inconclusive | `eligibility_status=inconclusive` | `trust_report_only` |
| Stale as-of / flag | `freshness_status=stale` | `context_only_stale` |
| Estimand mismatch flag | `adapter_notes` + `brand_lift` id | `scope_mismatch` at attach |
| Invalid adapted row | `validate_adapter_output` fails | Excluded from batch |

---

## 7. How do artifacts / CLI surface evidence?

ETL writes C2 file → `mmm train --calibration-signals-path` → H8 JSON/MD/CLI (MIP-C2).  
Representative adapted bundle: [MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json](../05_validation/archives/MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json).

---

## 8. Why optimizer / DecisionSurface / recommendations remain unchanged

Adapters emit diagnostic signals only — no `optimizer_*`, `decision_surface`, or `recommendation` fields.  
`validate_adapter_output` rejects forbidden keys.  
No changes to `RidgeBOMMMTrainer`, `decide`, or extension optimizer paths.

---

## 9. What remains before live GeoX/CLS ETL?

| Item | Status |
|------|--------|
| Adapter contract + module | ✅ |
| Fixture exports + tests | ✅ |
| C2 train-boundary ingest | ✅ (MIP-C2) |
| MIP-C4 ETL dry-run | ✅ [AUDIT-MIP-C4](AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md) |
| Scheduled production ETL job | **Not in scope** (MIP-C5) |
| Live API client | **Blocked** until ETL + TrustReport governance |
| Channel rename mapping table (client-specific) | **Operator-owned** |
| Production evidence approval | **Governance elsewhere** |

---

## 10. Verification checklist

| # | Requirement | CI |
|---|-------------|-----|
| 1 | GeoX valid → C2-compatible | `test_geox_valid_converts_to_c2_compatible_signal` |
| 2 | CLS valid → C2-compatible | `test_cls_valid_converts_to_c2_compatible_signal` |
| 3 | Missing uncertainty → blocked | `test_geox_missing_uncertainty_marked_blocked` |
| 4 | Stale CLS | `test_cls_stale_marked_stale` |
| 5 | Estimand mismatch preserved | `test_geox_estimand_mismatch_preserved_for_trust_report` |
| 6 | Mixed batch | `test_mixed_batch_normalizes_safely` |
| 7 | C2 ingest path | `test_adapter_output_ingests_via_c2_path` |
| 8 | No decision fields | `test_no_optimizer_decision_surface_recommendation_fields` |
| 9 | MIP-C2/C1 regression | `test_calibration_signal_train_boundary_ingestion.py`, etc. |

---

## 11. Related

- [geox_cls_to_calibration_signal_adapter_contract.md](../05_validation/geox_cls_to_calibration_signal_adapter_contract.md)
- [AUDIT-MIP-C2](AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md)
- [AUDIT-MIP-C1](AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md)
