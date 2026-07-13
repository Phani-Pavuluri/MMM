# MMM-EXPORT-002 — Typed MMM Export Schemas and Fixture Bundles

**Status:** Accepted (schema + fixtures only)  
**Date:** 2026-05-22  
**Prerequisites:** [MMM-EXPORT-001 inventory](mmm_to_mip_export_contract_inventory.md) @ `04dbc51`  
**Module:** `mmm/contracts/mip_export.py`  
**Fixtures:** `tests/fixtures/mip_export/`  
**Tests:** `tests/contracts/test_mmm_mip_export_contracts.py`

---

## 1. Purpose

Provide **typed schemas**, **claim-safety validators**, and **synthetic fixture `MMMExportBundle`s** so MIP can later:

- test readiness / blocked-answer paths
- demo synthetic ROI with explicit non-production labels
- refuse budget recommendations without a governed recommendation contract

**This does not** adapt live train/decide artifacts, emit real recommendations, integrate MIP, or mark ROI as production-safe.

Runtime adapter is deferred to **MMM-EXPORT-003**.

---

## 2. Schema module

| Symbol | Role |
|--------|------|
| `MMMClaimSafety` | Claim-safety rollup |
| `MMMArtifactLineage` | Fingerprint / source / version lineage block |
| `MMMUncertaintySummary` | Uncertainty status + notes |
| `MMMDiagnosticGateSummary` | Diagnostic pass/warn/fail summary |
| `MMMModelFitArtifact` | Fit readiness export shape |
| `MMMModelDiagnosticArtifact` | Diagnostic export shape |
| `MMMChannelContributionArtifact` | Contribution export shape |
| `MMMChannelROIArtifact` | ROI/ROAS export shape (gated) |
| `MMMResponseCurveArtifact` | Curve export shape |
| `MMMSimulationResultArtifact` | Simulation export shape |
| `MMMOptimizerResultArtifact` | Optimizer result shape (not a recommendation) |
| `MMMRecommendationContract` | Budget-shift authorization wrapper |
| `MMMExportBundle` | Multi-artifact envelope for one `model_run_id` |

**Schema version:** `mmm_mip_export_v1`

Implementation uses **Pydantic** models (same pattern as planning / quantity contracts).

---

## 3. Enums / taxonomies

| Taxonomy | Values |
|----------|--------|
| Inventory status | `EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP`, `EXISTS_PARTIAL_NOT_CONSUMABLE`, `EXISTS_RESEARCH_ONLY`, `PLANNED_NOT_IMPLEMENTED`, `MISSING` |
| Claim-safety codes | `production_claim_allowed`, `diagnostic_explanation_allowed`, `readiness_explanation_allowed`, `demo_fixture_only`, `blocked_until_*` |
| Artifact safety | `production_safe`, `diagnostic_only`, `readiness_only`, `demo_fixture_only`, `blocked` |

---

## 4. Validators

| Function | Behavior |
|----------|----------|
| `validate_mmm_export_artifact` | Structural + claim-safety; returns error list |
| `validate_mmm_export_bundle` | Bundle + all children |
| `validate_claim_safety` | Flag consistency (promo, demo, LLM, ROI, recs) |
| `validate_recommendation_contract` | Required source optimizer, claims, gates |
| `artifact_is_mip_exposable` | Production-consumable only (none of shipped fixtures) |
| `artifact_is_demo_safe` | Demo flags + non-production |
| `artifact_is_readiness_exposable` | Readiness language only |
| `bundle_is_mip_consumable` | Strict production bundle gate |

### Core rules

- Missing lineage / `schema_version` / `model_run_id` fails.
- Fingerprints required unless `is_docs_planned_placeholder=true`.
- `production_claim_allowed=true` requires `promotion_status=approved_for_prod`.
- `llm_exposure_allowed=true` requires `diagnostic_status` and **non-empty** `forbidden_claims`.
- `demo_fixture_allowed=true` requires `production_claim_allowed=false`.
- ROI values require uncertainty + diagnostic; missing/none uncertainty ⇒ not production-safe; prefer `blocked_until_uncertainty`.
- Response curves cannot set `recommendation_allowed=true`.
- Optimizer cannot set `recommendation_allowed=true` without a RecommendationContract.
- RecommendationContract demo / non-promoted sources stay blocked from production.
- Bayes / `research_only` cannot be production-safe.
- Bundle is not MIP-consumable if any child violates claim safety or inventory is not governed-consumable.

---

## 5. Fixtures

| File | Meaning |
|------|---------|
| `readiness_only_bundle.json` | Fit + diagnostic; readiness explainable; no ROI/recs |
| `diagnostic_roi_blocked_bundle.json` | ROI present, uncertainty missing → blocked until uncertainty |
| `demo_fixture_roi_bundle.json` | Synthetic ROI; demo only; not business truth |
| `blocked_budget_recommendation_bundle.json` | Optimizer result, no RecommendationContract |
| `valid_recommendation_contract_shape_blocked_fixture.json` | Valid **shape** of RecommendationContract; still demo / not promoted |

**None** of these fixtures set `EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP` or `production_claim_allowed=true`.

---

## 6. How MIP should interpret fixtures

| Bundle | MIP may | MIP must not |
|--------|---------|--------------|
| Readiness only | Say fit/diagnostics exist; exports for ROI/recs still blocked | Report channel ROI or budget advice |
| ROI blocked | Explain blocked_until_uncertainty | Present Meta/Search ROI as truth |
| Demo ROI | Show labeled synthetic demo | Treat as production measurement |
| Budget blocked | Explain need for RecommendationContract | “Move $200K Meta → Search” |
| Rec shape blocked | Validate schema wiring | Expose as approved recommendation |

---

## 7. Demo-safe vs production-safe

| | Demo-safe | Production-safe |
|--|-----------|-----------------|
| Flags | `demo_fixture_allowed=true`, `production_claim_allowed=false` | `artifact_safety_status=production_safe`, `approved_for_prod`, not demo |
| Shipped fixtures | Demo ROI + rec-shape | **None** |
| UI | Must label synthetic/demo | TrustReport + promotion required |

---

## 8. Why EXPORT-003 is next

Schemas + fixtures define the **boundary**. Real package artifacts (`extension_report`, decide JSON, diagnostics, curves) still need a **runtime adapter** to populate these types without inventing claim flags.

Recommended sequence:

1. **MMM-EXPORT-003** — runtime adapter (claim-gated, still conservative)  
2. **MIP-EXPORT-001** — MIP ingestion + answerability gates on fixtures then live bundles  

---

## 9. Related

- [mmm_to_mip_export_contract_inventory.md](mmm_to_mip_export_contract_inventory.md)  
- INV-MMM-EXPORT-CONTRACTS-001  
