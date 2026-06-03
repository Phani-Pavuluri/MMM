# AUDIT-MIP-C1 — CalibrationSignal → MMM Diagnostic Integration Gate

**Audit ID:** AUDIT-MIP-C1  
**Date:** 2026-06-01  
**Scope:** Contract and attachment audit for external causal evidence (GeoX, CLS, A/B, holdout, replay) on Ridge production diagnostics  
**Prerequisites:** H10 Ridge diagnostic E2E audit **Pass** @ `3ed159d`  
**Verdict:** **Pass (contract)** — CalibrationSignal may attach to Ridge diagnostics as governed context only; fitting, optimizer, DecisionSurface, and recommendations unchanged

---

## 1. Purpose

Define and verify how **CalibrationSignal** evidence from GeoX, CLS, A/B tests, holdouts, and replay should attach to **Ridge production diagnostic artifacts** for:

- operator interpretation context  
- conflict detection (external vs MMM direction)  
- TrustReport boundary preservation  

This gate is **audit/contract work first**. It does **not** authorize automatic model correction, coefficient override, or decision-path promotion.

**Core principle:** CalibrationSignal enters MMM as **governed evidence context**, not as automatic Ridge correction.

---

## 2. Scope

| In scope | Out of scope |
|----------|--------------|
| Attachment contract + field schema | Feeding CalibrationSignal into Ridge coefficients |
| `calibration_evidence_context` on Ridge reports | Automatic MMM recalibration |
| Operator Markdown / CLI context blocks | Optimizer or DecisionSurface input changes |
| Fixture-backed contract tests | Budget recommendations |
| Conflict / stale / estimand policy | GeoX/CLS silently overriding MMM |
| TrustReport boundary documentation | Bypassing TrustReport |
| | Bayes-H5 production promotion |
| | Evidence production-approval (unless governed elsewhere) |

---

## 3. Evidence sources

| Source | `source_system` (typical) | `source_modality` (typical) | Role in MIP-C1 |
|--------|---------------------------|-----------------------------|----------------|
| **GeoX** | `geox` | `geo_experiment` | DMA/geo lift vs Ridge channel coef direction |
| **CLS** | `cls` | `cls_readout` | National/regional readout; eligibility may be inconclusive |
| **A/B** | `ab_platform` | `ab_test` | Estimand-sensitive; scope mismatch → TrustReport-only |
| **Holdout** | `holdout_registry` | `holdout` | Requires uncertainty for calibration-informed use |
| **Replay** | `replay` | `replay_unit` | Sparse-channel context; does not remove MMM sparse forbidden claims |

All sources materialize as **CalibrationSignal** records per [Bayes-H2 ADR](../05_validation/bayes_h2_calibration_signal_mapping_adr.md).

---

## 4. Ridge diagnostics vs CalibrationSignal

| Aspect | Ridge production diagnostics (H7–H10) | CalibrationSignal attachment (MIP-C1) |
|--------|--------------------------------------|----------------------------------------|
| **Producer** | `compose_ridge_diagnostic_report` | External evidence registry / ETL |
| **Content** | Transform, controls, collinearity, sparse, fold/coef stability | Experiment lift, SE, scope, freshness, estimand |
| **Mutates fit** | No (metadata from existing fit) | **Must not** |
| **Decision paths** | Forbidden outputs blocked | **Must not** feed optimizer / DecisionSurface / recommendations |
| **Collinearity flag** | `calibration_evidence_available` (boolean) | Set true when governed signals attached — **interpretation only** |

---

## 5. Allowed uses

- Attach `calibration_evidence_context` to `ridge_production_diagnostics_report`  
- Surface summary in `ridge_production_diagnostics_summary.md` and CLI (short warning/context block)  
- Embed context pointer in `extension_report.json` (diagnostic report unchanged in fit semantics)  
- Flag **aligned** signals for operator interpretation alongside Ridge severity  
- Flag **directional_conflict** for TrustReport + human review  
- Mark **stale** / **estimand mismatch** / **missing uncertainty** as TrustReport-only or context-only  
- Merge calibration **forbidden_claims** into report forbidden list (additive, not subtractive)  

---

## 6. Forbidden uses

- Feed CalibrationSignal into Ridge refit or coefficient override  
- Automatic recalibration of MMM from external evidence  
- Optimizer, DecisionSurface, or budget recommendation inputs from signals  
- GeoX/CLS silently overriding MMM coefficients or severity  
- Bypassing TrustReport for conflict, freshness, or estimand failures  
- Promoting Bayes-H5 or marking evidence production-approved without existing gates  
- Claims: `mmm_direction_validated_by_external_evidence`, `external_evidence_overrides_mmm_coefficients`, `fresh_calibration_evidence_claim` (when stale), etc.  

---

## 7. TrustReport boundary

CalibrationSignal attachment **complements** TrustReport; it does not replace it.

| Disposition | Meaning |
|-------------|---------|
| `diagnostic_context` | May appear in Ridge diagnostic context when aligned and eligible |
| `trust_report_only` | Record in TrustReport; not calibration-informed Ridge interpretation |
| `trust_report_and_human_review` | Directional conflict — disclose in TrustReport + operator review |
| `context_only_stale` | Stale warning in operator context only |

TrustReport remains authoritative for evidence quality, replay certification, freshness (VAL-007), and release gates.

---

## 8. Conflict-handling policy

1. **Directional conflict** (external sign ≠ Ridge channel coef sign): `conflict_status=directional_conflict`; MMM coefficients **unchanged**; forbidden claims include `mmm_direction_validated_by_external_evidence`.  
2. **Scope / estimand mismatch**: `conflict_status=scope_mismatch`; TrustReport-only.  
3. **Inconclusive eligibility** (e.g. CLS): `conflict_status=trust_report_only` even if directions agree.  
4. **Stale**: alignment not evaluated; `fresh_calibration_evidence_claim` forbidden.  
5. **No silent merge** — consistent with Bayes-H2b validator policy (`SILENT_AVERAGE_FORBIDDEN`).  

---

## 9. Artifact placement

| Artifact | Allowed | Forbidden |
|----------|---------|-----------|
| `ridge_production_diagnostics_report.calibration_evidence_context` | ✓ | — |
| `ridge_production_diagnostics_summary.md` | ✓ (context section) | Optimizer/recommendation narrative |
| `extension_report.json` | ✓ (embedded report) | Coef override fields |
| CLI summary | ✓ (short block) | DecisionSurface / optimizer lines |
| Ridge refit input | — | ✓ |
| Optimizer / DecisionSurface / recommendation input | — | ✓ |

**Implementation:** `mmm/diagnostics/calibration_signal_attachment.py`  
**Contract:** [calibration_signal_mmm_diagnostic_attachment_contract.md](../05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md)  
**Fixtures:** `tests/fixtures/mip_calibration_signal_attachment/`  
**CI:** `tests/mip/test_calibration_signal_mmm_attachment_contract.py`

---

## 10. Production boundary

| Boundary | Status |
|----------|--------|
| Ridge remains production baseline | ✓ |
| Bayes-H5 research-only | ✓ |
| CalibrationSignal context-only attachment | ✓ |
| No optimizer / DecisionSurface / recommendation changes | ✓ |
| TrustReport not bypassed | ✓ |
| Diagnostics ≠ automatic recalibration | ✓ |

---

## 11. Verification checklist

| # | Requirement | CI |
|---|-------------|-----|
| 1 | Attachment contract documented | contract doc |
| 2 | Seven fixture scenarios | `test_fixture_attachment_matches_contract` |
| 3 | Aligned signal → context only | `test_aligned_signal_attaches_as_context` |
| 4 | Conflict does not override MMM coefs | `test_conflict_does_not_override_mmm` |
| 5 | Stale → context-only stale | `test_stale_signal_context_only` |
| 6 | Estimand mismatch → TrustReport-only | `test_estimand_mismatch_trust_report_only` |
| 7 | Missing SE blocks calibration use | `test_missing_uncertainty_blocks_calibration_use` |
| 8 | Sparse channel forbidden claim persists | `test_sparse_channel_forbids_clean_mmm_only_claim` |
| 9 | No forbidden decision-path fields | `test_no_optimizer_decision_surface_recommendation_fields` |
| 10 | H10 regression tests still pass | `test_ridge_diagnostic_e2e_audit.py` |

---

## 12. Related artifacts

| Item | Path |
|------|------|
| H10 audit | [AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md](AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md) |
| Ridge diagnostics contract | [ridge_production_diagnostics_contract.md](../05_validation/ridge_production_diagnostics_contract.md) |
| Bayes-H2 CalibrationSignal ADR | [bayes_h2_calibration_signal_mapping_adr.md](../05_validation/bayes_h2_calibration_signal_mapping_adr.md) |
| Helper module | `mmm/diagnostics/calibration_signal_attachment.py` |
