# MMM-EXPORT-001 — Governed MMM-to-MIP Export Contract Inventory

**Status:** Accepted (inventory / contract boundary — docs only)  
**Date:** 2026-05-22  
**Investigation:** INV-MMM-EXPORT-CONTRACTS-001  
**Prerequisites:** MIP-C1–C5 file bridge ✅ · AUDIT-MIP-C6 pause before live scheduler ✅ · package-side agents deferred ✅  

---

## 1. Purpose

Define the **first governed inventory** of MMM → MIP export artifact families so MIP can decide — without guessing from internal train/decide JSON — what it may:

- explain as readiness / diagnostics
- demo as synthetic fixtures
- surface as ROI / contribution / response curves
- recommend as budget shifts

**This is contract/inventory work only.** No runtime adapters, optimizer/simulator changes, recommendation emission, MIP integration code, LLM/orchestration, or Bayes-H5 promotion.

### Guiding question

> Until MMM emits a governed, typed, MIP-consumable export with uncertainty, diagnostic/trust status, and allowed/blocked claims, MIP must **not** treat internal package artifacts as safe answers for ROI, ROAS, contribution, curves, optimizer results, or budget recommendations.

---

## 2. Architecture boundary

| Layer | Role |
|-------|------|
| **MMM package (internal)** | Fit, diagnostics, CalibrationSignal *ingestion*, simulate/optimize DecisionSurface paths, research/Bayes artifacts |
| **MMM export contracts (this inventory)** | Typed MIP-facing artifact families with claim gates |
| **MIP** | Ingestion, answerability gates, TrustReport, user-facing demo/LLM exposure |
| **Future package-side agents** | Interpret exports after contracts exist — [mmm_package_side_agents_roadmap.md](mmm_package_side_agents_roadmap.md) |

**CalibrationSignal direction:** MIP-C1–C5 govern **external evidence → MMM diagnostics** (context-only). That is **not** an MMM → MIP recommendation/ROI export.

---

## 3. Status taxonomy

| Code | Meaning |
|------|---------|
| `EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP` | Typed export + lineage + claim gates; MIP may consume under documented flags |
| `EXISTS_PARTIAL_NOT_CONSUMABLE` | Internal or package-facing artifacts exist; **not** safe for MIP ROI/recommendation exposure without a governed adapter |
| `EXISTS_RESEARCH_ONLY` | Research/Bayes/sandbox; never MIP-production claims |
| `PLANNED_NOT_IMPLEMENTED` | Named in this inventory; schema/runtime not yet built |
| `MISSING` | Required for a claim class; not present even as planning name until this doc |

---

## 4. Claim-safety taxonomy

| Code | Meaning |
|------|---------|
| `production_claim_allowed` | May support production decision language when flags and TrustReport gates pass |
| `diagnostic_explanation_allowed` | May explain model/diagnostic status to operators |
| `readiness_explanation_allowed` | May state whether fit/export/gates exist (meta) |
| `demo_fixture_only` | Synthetic demo only; must label non-production |
| `blocked_until_contract` | Blocked until typed export family exists |
| `blocked_until_uncertainty` | Blocked until uncertainty fields are populated and status ≠ missing |
| `blocked_until_promotion` | Blocked until promotion/release gates allow |
| `blocked_until_recommendation_contract` | Blocked until `MMMRecommendationContract` + `recommendation_allowed=true` |

---

## 5. Common required fields (all export families)

Every governed MMM → MIP export artifact (or future schema) **must** carry:

| Field | Role |
|-------|------|
| `artifact_type` | Family id (e.g. `MMMChannelROIArtifact`) |
| `schema_version` | Contract version string |
| `model_run_id` | Train/refresh run id |
| `training_data_fingerprint` | Panel/data fingerprint (e.g. `sha256_combined`) |
| `model_artifact_fingerprint` | Fit artifact fingerprint / lineage hash |
| `source_artifacts` | Pointers to internal package files (`extension_report`, decide JSON, diagnostics) |
| `model_form` | e.g. `semi_log`, bayesian research form |
| `estimand` | What is being measured / optimized |
| `time_window` | Training or scoring window |
| `geo_scope` | National / geo set |
| `channel_scope` | Channels covered |
| `outcome_metric` | KPI name |
| `spend_metric` | Spend/exposure metric name |
| `currency` | Currency code if monetary |
| `uncertainty_status` | e.g. `present` / `partial` / `missing` / `not_applicable` |
| `diagnostic_status` | e.g. `pass` / `warn` / `fail` / `unknown` |
| `promotion_status` | e.g. `research_only` / `diagnostic_only` / `planning_candidate` / `approved_for_prod` |
| `calibration_status` | Calibration / CalibrationSignal attachment summary |
| `planning_allowed` | bool — DecisionSurface planning eligibility (package gate mirror) |
| `llm_exposure_allowed` | bool — MIP/LLM may narrate this artifact |
| `demo_fixture_allowed` | bool — synthetic demo may use this payload |
| `recommendation_allowed` | bool — budget-shift / reallocation language allowed |
| `allowed_claims` | Explicit claim codes |
| `forbidden_claims` | Explicit blocked claim codes |
| `generated_at` | ISO timestamp |
| `package_version` | MMM package version |
| `git_commit` | Repo commit of producer |

**Default for this inventory (until EXPORT-002/003):** treat all MIP exposure flags as **false** unless an artifact family is explicitly marked otherwise — today **none** are `EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP`.

---

## 6. Artifact families

### 6.1 MMMModelFitArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Governed summary of a completed MMM fit for MIP readiness language (“a Ridge model exists for run X”), not ROI. |
| **Source artifacts / modules** | Train outputs; `extension_report.json`; `ridge_fit_summary`; `transform_policy`; `mmm/governance/decision_ridge_summary.py`; trainer result bundles |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` |
| **Required fields** | Common fields + `framework` (`ridge`/`bayesian`), `model_release.state`, `ridge_fit_summary` subset or bayesian meta pointer, `run_environment` |
| **Required lineage** | `model_run_id`, fingerprints, `source_artifacts`, `git_commit`, `package_version` |
| **Required uncertainty** | `uncertainty_status` (often `not_applicable` for point Ridge; Bayesian: draws availability) |
| **Required diagnostic/trust** | `diagnostic_status`, pointer to `MMMModelDiagnosticArtifact` when present, forbidden_claims from diagnostics |
| **Allowed claims** | `readiness_explanation_allowed` — fit exists / model_form / channels trained |
| **Forbidden claims** | Causal ROI/ROAS; “best channel”; budget recommendation; Bayes production-ready |
| **MIP exposure** | Not consumable until typed export + adapter |
| **Demo fixture** | Allowed later only with `demo_fixture_allowed=true` and synthetic labeling |
| **Required adapter** | Fit → `MMMModelFitArtifact` schema (EXPORT-002/003) |

---

### 6.2 MMMModelDiagnosticArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | MIP-facing Ridge (and later) diagnostics: transforms, collinearity, sparse channels, stability, CalibrationSignal context, severity. |
| **Source artifacts / modules** | H7–H11: `mmm/diagnostics/ridge_diagnostics.py`, `ridge_diagnostic_summary.py`, `ridge_severity_policy.py`; `ridge_production_diagnostics_report`; CalibrationSignal context (MIP-C1–C5) |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` — rich internally; not yet a MIP export type |
| **Required fields** | Common fields + severity summary, forbidden_claims list, calibration_evidence_context pointer, eligibility flags |
| **Required lineage** | `evidence_attachment_lineage` when signals attached; diagnostic `report_version` |
| **Required uncertainty** | Uncertainty of external evidence when present; diagnostic completeness flags |
| **Required diagnostic/trust** | H9 severity, TrustReport boundary notes, `calibration_status` |
| **Allowed claims** | `diagnostic_explanation_allowed`, `readiness_explanation_allowed` |
| **Forbidden claims** | Override DecisionSurface; promote Bayes; invent causal ROI from conflict flags |
| **MIP exposure** | **Closest** to first MIP-safe readiness path once adapter exists |
| **Demo fixture** | Synthetic diagnostic reports only with explicit demo flags |
| **Required adapter** | Diagnostics report → `MMMModelDiagnosticArtifact` |

---

### 6.3 MMMChannelContributionArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Channel-level contribution / decomposition for explanation — not production budget authority. |
| **Source artifacts / modules** | Decomposition / contribution paths under `mmm/decomposition/`; extension report contribution sections if present |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` |
| **Required fields** | Common fields + per-channel contribution values, aggregation method, baseline definition |
| **Required lineage** | Fit fingerprints; estimand for contribution |
| **Required uncertainty** | Interval or explicit `uncertainty_status=missing` with forbidden contribution-certainty claims |
| **Required diagnostic/trust** | Collinearity / sparse warnings affecting attribution; forbidden over-claim |
| **Allowed claims** | Diagnostic contribution discussion when gated |
| **Forbidden claims** | “True incremental share without TrustReport”; recommendation from contribution ranks alone |
| **MIP exposure** | Blocked for LLM/demo production claims |
| **Demo fixture** | Synthetic only |
| **Required adapter** | Decomposition → contribution export + claim gates |

---

### 6.4 MMMChannelROIArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Governed channel ROI / ROAS / marginal ROI for MIP only when uncertainty and diagnostics allow. |
| **Source artifacts / modules** | `mmm/reporting/roi_sections.py`; curve/marginal ROI grids; train reporting — **not** a DecisionSurface estimand by itself |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` (reporting/diagnostic) → treated as **not MIP-consumable** |
| **Required fields** | Common fields + ROI/ROAS definition, numerator/denominator metrics, channel rows, estimand |
| **Required lineage** | Fit + panel fingerprints; spend/outcome metric ids |
| **Required uncertainty** | Required for `llm_exposure_allowed`; else `blocked_until_uncertainty` |
| **Required diagnostic/trust** | Link to diagnostics; promotions; sparse/collinearity forbidden claims |
| **Allowed claims** | None for MIP until contract + uncertainty + promotion status |
| **Forbidden claims** | “Highest ROI channel” narrative; causal field lift ≈ MMM ROI; budget shift from ROI rank |
| **MIP exposure** | **Blocked** |
| **Demo fixture** | Only with `demo_fixture_allowed=true` and synthetic UI labeling |
| **Required adapter** | ROI report → `MMMChannelROIArtifact` with gates |

---

### 6.5 MMMResponseCurveArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Univariate response / saturation curves for explanation. Curves explain; full-panel simulation decides. |
| **Source artifacts / modules** | [response_curves.md](../02_concepts/response_curves.md); `mmm/decomposition/curves.py`; `curve_bundles` in train artifacts |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` |
| **Required fields** | Common fields + channel, grid, transform path (adstock→Hill), alignment note to full-panel Δμ |
| **Required lineage** | Transform policy + fit fingerprints |
| **Required uncertainty** | Status for curve/marginal estimates |
| **Required diagnostic/trust** | Explicit `forbidden_claims` that curves are not allocators |
| **Allowed claims** | Diagnostic shape / saturation discussion |
| **Forbidden claims** | Budget allocation from curve increments; replace DecisionSurface |
| **MIP exposure** | Blocked for recommendations |
| **Demo fixture** | Synthetic curves with demo flags only |
| **Required adapter** | `curve_bundles` → `MMMResponseCurveArtifact` |

---

### 6.6 MMMSimulationResultArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Full-panel Δμ simulation result under DecisionSurface governance for scenario explanation. |
| **Source artifacts / modules** | `mmm decide simulate`; `mmm/decision/service.py`; decision payload JSON; [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md) |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` — internal/governance-gated; no MIP export type |
| **Required fields** | Common fields + scenario id, BAU baseline, Δμ summary, `planning_allowed` mirror, gate outcomes |
| **Required lineage** | Fingerprint match train↔decide; extension_report pointer |
| **Required uncertainty** | Posterior quantiles only when bayesian planning rules allow; else explicit NA |
| **Required diagnostic/trust** | Model release, fingerprinted alignment, optimization-safety adjacent disclosures |
| **Allowed claims** | Scenario Δμ under stated assumptions when `planning_allowed` and export flags true |
| **Forbidden claims** | Recommendations without `MMMRecommendationContract`; curve-proxy as Δμ |
| **MIP exposure** | Blocked until export + gates mapped |
| **Demo fixture** | Synthetic Δμ only with demo flags |
| **Required adapter** | Decide simulate payload → `MMMSimulationResultArtifact` |

---

### 6.7 MMMOptimizerResultArtifact

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Constrained optimizer output (recommended spend vector as **optimizer result**, not yet a recommendation contract). |
| **Source artifacts / modules** | `mmm decide optimize-budget`; simulation-based optimizer; curve-local optimizer (unsafe / non-prod) |
| **Status** | `EXISTS_PARTIAL_NOT_CONSUMABLE` |
| **Required fields** | Common fields + objective, constraints, solution spend vector, gate checklist, `allow_unsafe_decision_apis` flag |
| **Required lineage** | Same as simulation + optimization config fingerprint |
| **Required uncertainty** | Risk-aware / posterior modes when used; else `missing` + blocked certainty claims |
| **Required diagnostic/trust** | `governance.approved_for_optimization`, replay calibration evidence, model_release |
| **Allowed claims** | None for MIP LLM until recommendation contract wraps this with `recommendation_allowed` |
| **Forbidden claims** | Auto-approve budgets; Bayesian prod optimize; unsafe curve-local as prod |
| **MIP exposure** | Blocked |
| **Demo fixture** | Synthetic optimizer output with demo flags only |
| **Required adapter** | Optimize JSON → `MMMOptimizerResultArtifact` + gate mapping |

---

### 6.8 MMMRecommendationContract

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Explicit, claim-gated wrapper authorizing budget-shift / reallocation **language** for MIP (and humans). |
| **Source artifacts / modules** | **None today** as a first-class type; would wrap simulation + optimizer + TrustReport + diagnostics |
| **Status** | `PLANNED_NOT_IMPLEMENTED` / effectively `MISSING` for MIP |
| **Required fields** | Common fields + proposed shifts, rationale codes, assumption set, TrustReport refs, `recommendation_allowed=true` only when all gates pass |
| **Required lineage** | Fit, sim, opt, TrustReport, diagnostics source_artifacts |
| **Required uncertainty** | Required |
| **Required diagnostic/trust** | Must fail-closed if diagnostics forbid claims |
| **Allowed claims** | Only when fully populated and promotion allows |
| **Forbidden claims** | Recommendations from ROI rank, curves alone, research Bayes, or missing TrustReport |
| **MIP exposure** | Blocked until implemented |
| **Demo fixture** | Demo recommendations must set `production_claim_allowed=false` |
| **Required adapter** | New producer after EXPORT-002 schemas |

---

### 6.9 MMMExportBundle

| Dimension | Definition |
|-----------|------------|
| **Purpose** | Package of export artifacts for a single `model_run_id` that MIP ingests as one unit. |
| **Source artifacts / modules** | **Missing** — planned envelope over families above |
| **Status** | `PLANNED_NOT_IMPLEMENTED` |
| **Required fields** | Common fields + `artifacts[]` inventory, per-artifact exposure flags, overall `llm_exposure_allowed` / `recommendation_allowed` roll-up (min/strict) |
| **Required lineage** | Bundle id, all child fingerprints, git_commit |
| **Required uncertainty** | Roll-up of child `uncertainty_status` |
| **Required diagnostic/trust** | Roll-up diagnostic/promotion/calibration |
| **Allowed claims** | Intersection of child allowed_claims |
| **Forbidden claims** | Union of child forbidden_claims (additive) |
| **MIP exposure** | Required before broad MIP consumption |
| **Demo fixture** | Demo bundles must be labeled and non-production |
| **Required adapter** | EXPORT-003 runtime assembler |

---

## 7. Related internal inputs (not MMM→MIP recommendation exports)

| Item | Role | Export status note |
|------|------|--------------------|
| **CalibrationSignal ingestion (MIP-C1–C5)** | External evidence → Ridge diagnostic context | Governed **input** bridge; not ROI/recommendation output |
| **Bayes-H5 / research fits** | Research-only | `EXISTS_RESEARCH_ONLY` — never MIP production claims |
| **TrustReport / release gates** | Platform governance | Consumed by future recommendation contract; not inventable by export |
| **Nested `extension_report.decision_bundle`** | Research tier (INV-014) | Must not substitute for governed decide JSON or export |

---

## 8. Current verdict table

| Item | Current status | MIP-safe? | Demo-safe? | Required next step |
|------|----------------|-----------|------------|--------------------|
| CalibrationSignal ingestion | Governed diagnostic/context **input** (MIP-C1–C5) | As **input context** only — not output ROI | N/A as MMM export | Keep C6 pause; GeoX OC for producer quality |
| Ridge diagnostics / trust artifacts | `EXISTS_PARTIAL_NOT_CONSUMABLE` | **Not yet** — nearest readiness candidate after adapter | Only synthetic with flags | EXPORT-002 schema + diagnostic adapter |
| Model fit | `EXISTS_PARTIAL_NOT_CONSUMABLE` | No | Synthetic only later | `MMMModelFitArtifact` contract + adapter |
| Channel contribution | `EXISTS_PARTIAL_NOT_CONSUMABLE` | No | Synthetic only later | Contribution export + uncertainty |
| ROI / ROAS | Partial diagnostic reporting / **not** governed export | **No** | Only if `demo_fixture_allowed` + synthetic UI | `MMMChannelROIArtifact` + uncertainty gates |
| Response curves | Partial / diagnostic | **No** for recommendations | Synthetic only later | `MMMResponseCurveArtifact` + forbidden allocator claims |
| Simulator / optimizer | Internal / governance-partial | **No** | Synthetic only later | Sim/opt export + gate mapping |
| RecommendationContract | `PLANNED_NOT_IMPLEMENTED` / missing | **No** | Demo fake recs only with labels | Design + implement contract |
| MMMExportBundle | `PLANNED_NOT_IMPLEMENTED` | **No** | Demo bundle later | EXPORT-002 fixture bundle |

**Overall inventory verdict:** **No** MMM family is currently `EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP`. MIP may only give **meta readiness** answers that cite this inventory (“fit internals exist; governed ROI/recommendation exports do not”).

---

## 9. Examples

### Example A — Safe readiness answer

**User:** “Do we have MMM results for Q1?”

**MIP may say:** A model fit artifact exists internally for the configured run, but **governed ROI / contribution / recommendation exports are blocked** pending `MMMExportBundle` and per-family contracts (`blocked_until_contract`).

**MIP must not say:** Channel ROIs, “Search wins,” or “increase Meta by 10%.”

---

### Example B — Blocked ROI answer

**User:** “Which channel has the highest ROI?”

**MIP must block** unless:

1. `MMMChannelROIArtifact` is governed and present in an `MMMExportBundle`,
2. `uncertainty_status` is acceptable,
3. diagnostics do not forbid the claim,
4. `llm_exposure_allowed=true`.

Otherwise claim code: `blocked_until_contract` / `blocked_until_uncertainty`.

---

### Example C — Blocked budget shift

**User:** “Move $200K from Meta to Search?”

**MIP must block** unless `MMMRecommendationContract` exists with `recommendation_allowed=true` (and TrustReport / promotion gates).  
Claim code: `blocked_until_recommendation_contract`.

Optimizer JSON alone is **insufficient**.

---

### Example D — Demo fixture

A fixture may show fake ROI **only if**:

- `demo_fixture_allowed=true`
- `production_claim_allowed=false` (and typically `llm_exposure_allowed=false` for production paths)
- UI explicitly labels **synthetic / demo**

Otherwise: blocked.

---

## 10. Next-lane plan

| ID | Scope | Output |
|----|-------|--------|
| **MMM-EXPORT-002** | Typed schemas + fixture `MMMExportBundle` | JSON Schema / pydantic placeholders + synthetic fixture with demo flags |
| **MMM-EXPORT-003** | Runtime adapter | Map existing package artifacts → populated export families (still claim-gated) |
| **MIP-EXPORT-001** | MIP-side | Ingest `MMMExportBundle`; answerability gates for LLM/demo/recommendation |

**Out of band (do not block on this inventory):** GeoX estimator/inference OC; C6 live scheduler; package-side agents; Bayes-H5 promotion.

---

## 11. Acceptance criteria (this commit)

| Criterion | Met? |
|-----------|------|
| Inventory doc answers what exists / governed / partial / blocked | ✅ |
| Nine artifact families documented | ✅ |
| Common fields listed | ✅ |
| Status + claim-safety taxonomies | ✅ |
| Conservative verdict table | ✅ |
| Examples A–D | ✅ |
| Next-lane EXPORT-002/003 + MIP-EXPORT-001 | ✅ |
| No runtime implementation | ✅ |

---

## 12. Related

- [decision_vs_research.md](../02_concepts/decision_vs_research.md)  
- [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md)  
- [ridge_production_diagnostics_contract.md](ridge_production_diagnostics_contract.md)  
- [response_curves.md](../02_concepts/response_curves.md)  
- [AUDIT-MIP-C6](../audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md)  
- [mmm_package_side_agents_roadmap.md](mmm_package_side_agents_roadmap.md)  
