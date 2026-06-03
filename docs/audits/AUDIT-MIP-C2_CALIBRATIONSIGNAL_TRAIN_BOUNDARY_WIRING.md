# AUDIT-MIP-C2 — CalibrationSignal Train/Extension Boundary Wiring

**Audit ID:** AUDIT-MIP-C2  
**Date:** 2026-06-01  
**Scope:** Optional CalibrationSignal ingestion into Ridge production diagnostics at train/extension boundary  
**Prerequisites:** MIP-C1 @ `df54dd1`; H11 @ `9d20c72`  
**Verdict:** **Pass (wiring)** — context attaches before artifact export; fit/optimizer/DecisionSurface unchanged

---

## 1. Purpose

Turn MIP-C1 from contract-only into a **safe train-boundary feature**: operators may supply CalibrationSignal JSON so external evidence appears in `ridge_production_diagnostics_report` before H8 export and CLI surfacing — **without** decision-path mutation.

---

## 2. Where does signal ingestion happen?

| Stage | Location |
|-------|----------|
| **Config** | `MMMConfig.extensions.ridge_diagnostics.calibration_signals_path` |
| **CLI** | `mmm train CONFIG.yaml --calibration-signals-path path/to/signals.json` (overrides YAML) |
| **Extension hook** | `mmm.evaluation.extension_runner._attach_ridge_production_diagnostics` |
| **Compose + attach** | `attach_ridge_diagnostics_to_extension_report` → `compose_ridge_diagnostic_report` → `ingest_calibration_signals_into_report` |
| **Artifact export** | `persist_training_artifacts` → `export_ridge_diagnostic_artifacts` (unchanged order; report already includes context) |

**Modules:** `mmm/diagnostics/calibration_signal_ingestion.py`, `mmm/diagnostics/calibration_signal_attachment.py`

---

## 3. How is lineage recorded?

Every Ridge diagnostic report includes `evidence_attachment_lineage` (MIP-C2):

| Field | Meaning |
|-------|---------|
| `attempted` | Ingestion path was invoked |
| `source_type` | `none` \| `file` \| `list` |
| `source_path` | File path when `source_type=file` |
| `source_ref` | Optional ref from JSON payload |
| `signals_count` | Records parsed (valid + rejected) |
| `attached_count` | Signals included in context |
| `rejected_count` | Records failing validation |
| `attachment_errors` | Parse/validation errors (fail-closed) |
| `context_only` | Always `true` |
| `optimizer_unchanged` / `decision_surface_unchanged` / `recommendations_unchanged` | Always `true` |

---

## 4. What happens with no signals?

- `attempted=false`, `source_type=none`
- No `calibration_evidence_context`
- H11 absent-lineage behavior preserved
- Markdown/CLI: explicit “not attached” section (H11 + MIP-C2)

---

## 5. What happens with conflicts?

- Valid signals still attach via MIP-C1 policy
- `conflict_status=directional_conflict` → TrustReport + human review disposition
- Forbidden claims include `mmm_direction_validated_by_external_evidence`
- **Ridge coefficients unchanged** (context-only)

---

## 6. What happens with malformed signals?

- Per-record validation (`signal_id` required)
- Invalid records → `attachment_errors` + `rejected_count`
- Valid records still attach when present
- All-invalid file: no context attach; errors in lineage + report warnings
- **Fit output unchanged** (ingestion runs post-fit on diagnostic report only)

---

## 7. How do artifacts / CLI surface evidence?

- `ridge_production_diagnostics_report.json` — full report + `calibration_evidence_context`
- `ridge_production_diagnostics_summary.md` — MIP-C1 context section when present
- `extension_report.json` — embedded report (+ summary when H11/H8 path sets it)
- `mmm train` — `format_ridge_diagnostics_cli_block` includes calibration headline when context present

**Representative archive:** [MIP_C2_RIDGE_DIAGNOSTICS_WITH_CALIBRATION_SIGNAL_CONTEXT_20260601.json](../05_validation/archives/MIP_C2_RIDGE_DIAGNOSTICS_WITH_CALIBRATION_SIGNAL_CONTEXT_20260601.json)

---

## 8. Why optimizer / DecisionSurface / recommendations remain unchanged

- Ingestion only mutates the **diagnostic report dict** after `trainer.fit()` completes
- `production_flags` still forbid optimizer / DecisionSurface / recommendations
- No calls into `decide`, optimizer certification, or curve-based budget paths
- CalibrationSignal does not set `calibration.enabled` or replay loss

---

## 9. What remains before live GeoX/CLS ETL?

| Item | Status |
|------|--------|
| MIP-C1 attachment contract | ✅ |
| H11 real-bundle diagnostics | ✅ |
| MIP-C2 file/list ingestion at train boundary | ✅ |
| Live GeoX/CLS API / registry ETL | **Not in scope** |
| TrustReport auto-population from signals | **Deferred** |
| Production approval of evidence | **Blocked** (governance elsewhere) |
| Bayes-H5 promotion | **Blocked** (research-only) |

---

## 10. Verification

| Check | CI |
|-------|-----|
| No-signal lineage | `test_no_signal_path_preserves_h11_absent_lineage` |
| Aligned file attach | `test_aligned_signal_file_attaches_context` |
| Conflict forbidden claims | `test_conflict_signal_attaches_forbidden_claims_on_stub` |
| Malformed fail-closed | `test_malformed_signal_file_records_errors_without_changing_fit` |
| Empty file | `test_empty_signal_file_attempted_zero_attached` |
| No decision fields | `test_no_optimizer_decision_surface_recommendation_fields` |
| MIP-C1 regression | `test_calibration_signal_mmm_attachment_contract.py` |

---

## 11. Related

- [AUDIT-MIP-C1](AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md)
- [calibration_signal_mmm_diagnostic_attachment_contract.md](../05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md)
- [INV-H11](../06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md)
