# INV-H11 — Real-Bundle Ridge Diagnostic Hardening

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-H11 |
| **Title** | Ridge diagnostics on real/realistic training bundles |
| **Status** | **closed** |
| **Date** | 2026-06-01 |
| **Manifest** | [H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_MANIFEST.md](H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_MANIFEST.md) |
| **Archive** | [H11 JSON](../05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) · [Summary MD](../05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md) |

---

## 1. Did the diagnostic stack work outside H6 synthetic worlds?

**Yes.** Ridge fit + H7 report + H9 severity + H8 Markdown/CLI completed on `MMM-BENCHMARK-GEO-PANEL-V1` (`examples/benchmark_geo_panel_v1.csv`) without crash.

- **Severity:** `diagnostic_only` (expected — omitted retail controls).  
- **Completeness checklist:** all H11 checks passed (see archive `completeness.checks`).  
- **H6 regression:** unchanged; H10 e2e tests still pass.

---

## 2. Which fields were missing?

| Gap | Handling |
|-----|----------|
| **Control columns on panel** | None in schema — recorded in `control_completeness.missing_required_controls` |
| **CalibrationSignal / MIP-C1 context** | Not wired — now explicit in `evidence_attachment_lineage` |
| **Client vertical truth** | Assumed `retail` for stress test — documented in manifest `vertical_assumption` |
| **Production replay calibration** | `calibration_evidence_available=false` in config |

No silent suppression: unknown verticals emit `control_completeness:unknown_vertical:*`; missing transform BO params emit `ridge_transform:missing_best_params_metadata`.

---

## 3. Which assumptions were required?

1. **Vertical = retail** on a panel with zero control columns — intentional stress (same class as H6 omitted-controls).  
2. **Research Ridge config** with reduced BO trials for CI/archive regeneration.  
3. **Public illustrative panel** — safe to commit lineage + redacted coef archive.  
4. **Coef redaction** in archive JSON only — in-memory reports used in tests retain numeric coefs for contract checks.

---

## 4. Were any forbidden claims emitted?

**Yes** (expected):

- `no_clean_media_attribution_claim`
- `no_channel_level_causal_claim_without_caveat`
- `no_budget_reallocation_claim_based_only_on_this_run`

Plus global MIP-C1 boundary claims are absent until `attach_calibration_evidence_context` is called.

---

## 5. Did operator summary remain useful?

**Yes.** Summary Markdown includes:

- H9 severity / eligibility / forbidden claims  
- Control completeness (missing required retail controls)  
- Collinearity and sparse-channel blocks  
- **Explicit MIP-C1 absence** section  
- Production boundary footer (optimizer/DecisionSurface off)

CLI block includes severity and forbidden-claims line.

---

## 6. What must be fixed before broader rollout?

| Item | Priority | Owner lane |
|------|----------|------------|
| Wire **vertical_id** from product metadata (not hard-coded retail) | High | Train / extension_runner |
| **Control column** ingestion on real client bundles | High | Data contract |
| **MIP-C2** live CalibrationSignal ETL (optional context attach) | Medium | After H11 sign-off |
| H11b **triangulation panel** (sparse radio + calibration stub) | Medium | Follow-on bundle |
| Real client bundles with confidentiality — redacted archives only | High | Ops |

Do **not** treat H11 illustrative panel as production promotion evidence.

---

## 7. Production boundaries preserved

| Boundary | Status |
|----------|--------|
| Ridge fitting unchanged on prod path | ✓ (H11 uses research runner only) |
| Optimizer / DecisionSurface / recommendations | ✓ not emitted |
| MIP-C1 context-only | ✓ explicit absence recorded |
| Bayes-H5 research-only | ✓ `production_flags` |
| Diagnostics not hard gates | ✓ |

---

## 8. Schema hardening delivered (H11)

| Change | Module |
|--------|--------|
| `resolve_vertical_profile` — unknown vertical → explicit warning, no crash | `vertical_control_profiles.py` |
| `vertical_profile_known` on control block | `ridge_diagnostics.py` |
| `evidence_attachment_lineage` on every report | `ridge_diagnostics.py` |
| Real-bundle runner + completeness validator | `ridge_real_bundle_hardening.py` |
| Operator summary: calibration absent section | `ridge_diagnostic_summary.py` |

**Tests:** `tests/diagnostics/test_ridge_diagnostics_real_bundle_compat.py`

---

## 9. Related

- [AUDIT-H10](../audits/AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md)  
- [AUDIT-MIP-C1](../audits/AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md)  
- [ridge_production_diagnostics_contract.md](../05_validation/ridge_production_diagnostics_contract.md)
