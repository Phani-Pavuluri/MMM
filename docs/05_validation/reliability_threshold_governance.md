# Reliability threshold governance and metric semantics (Phase 5D)

**Version:** `reliability_threshold_governance_v1.0.0`  
**Status:** Active — encodes [INV-056](exact_recovery_investigation_report.md) findings into platform semantics  
**ADR:** [DR-04](synthetic_architecture_decisions.md#dr-04--threshold-ownership-process-draft-resolution) (draft resolution)  
**Registry:** [validation_registry.md](validation_registry.md) §2.1 governance columns

---

## 1. Purpose

Phase 5C ([exact recovery investigation](exact_recovery_investigation_report.md)) showed that **Δμ and decision-surface recovery can be materially stronger than coefficient recovery** because multiple (transform, β) combinations can produce similar counterfactuals. Ridge BO must therefore **not** be judged primarily by exact coefficient recovery.

This document defines:

1. **Metric classes** — decision-grade vs diagnostic vs trust modifiers vs structural  
2. **Threshold governance** — versioning, approval, and release-blocking rules  
3. **Scorecard interpretation** — how [ReliabilityScorecard](reliability_scorecard.md) scores map to readiness  
4. **TrustReport semantics** — how warnings downgrade confidence without conflating attribution failure with decision failure

**Non-goals (Phase 5D):** New estimators, Bayesian MMM, Nevergrad, new transforms, production decisioning changes, or numeric threshold finalization (`TBD_v1` remains provisional).

---

## 2. Core finding (INV-056)

> **Coefficient recovery is not the same as decision recovery.**

| Layer | Typical WORLD-008 / L5B exact-recovery behavior |
|-------|--------------------------------------------------|
| Coefficient / transform (BO) | Fail — shared transform + BO search + collinearity |
| Δμ (TBD_v1) | Often **pass** despite large per-parameter error |
| Optimizer / replay (where in scope) | Evaluated separately on dedicated worlds |
| Structural / contracts | High — not the bottleneck |

**Implication:** Production readiness and TrustReport must treat **decision-grade** evidence as primary and **attribution/diagnostic** evidence as secondary unless the product explicitly claims attribution accuracy.

---

## 3. Metric classes

### 3.1 Decision-grade metrics

May support **production decision readiness** after sufficient validation and governance approval.

| Metric / capability | `validation_id` (primary) | Notes |
|---------------------|---------------------------|--------|
| Δμ recovery | VAL-004 | Counterfactual on full panel; primary business-facing metric |
| Optimizer recovery | VAL-005 | Allocation vs true optimum / regret |
| Replay lift recovery | VAL-006, VAL-007 | Calibration path; experiment-linked |
| DecisionSurface compatibility | CERT-4A-009 | Structural gate on decision API |
| Estimand compatibility | CERT-4A-010 | Replay / Δμ estimand alignment |
| Release-gate compatibility | CERT-4A-013, VAL-008, VAL-011 | `approved_for_prod` semantics |
| Decision safety | VAL-008 | Logical gate match |

**Default release policy:** Failures in decision-grade metrics **may** block release when thresholds are **approved** (not while `TBD_v1_runtime`).

### 3.2 Diagnostic / attribution metrics

Must **not by themselves** block production decisions unless the product **explicitly** certifies attribution accuracy.

| Metric / capability | `validation_id` (primary) | Notes |
|---------------------|---------------------------|--------|
| Coefficient recovery | VAL-001 | β vs `media_truth.coefficients` |
| Transform recovery (adstock / Hill) | VAL-002, VAL-003 | Decay, half-max, slope |
| Decomposition alignment | (future) | Channel contribution parity |
| Marginal curve shape | (future) | Response curve shape vs truth |
| Channel attribution parity | (derived from VAL-001 + transforms) | Not a separate gate by default |

**Default release policy:** `can_block_release: false` — surface in scorecard and TrustReport as **attribution diagnostic**, not decision blocker.

### 3.3 Trust modifiers

Affect **TrustReport confidence** and scorecard `trust_modifier_status`; may block or downgrade even when Δμ passes.

| Signal | `validation_id` / capability | Notes |
|--------|------------------------------|--------|
| Identifiability warnings | VAL-013 (partial), identifiability_behavior | VIF, collinearity, unstable coef path |
| Drift warnings | VAL-012, drift_behavior | Degradation vs changepoint truth |
| Calibration freshness | VAL-007 | Stale or low-quality calibration |
| Replay miss | VAL-006 fail | Decision-grade if replay is on critical path |
| Severe collinearity | WORLD-012 pattern | Downgrade trust; skip unstable coef checks |
| Parameter instability | Recovery marked unstable | Do not interpret coef failures literally |

**Default release policy:** `can_block_release: conditional` — severe drift or identifiability may block **decision-usable** classification even when Δμ passes on a narrow scenario.

### 3.4 Structural metrics

Platform integrity — **release-blocking** when failed at scale.

| Metric / capability | Examples |
|---------------------|----------|
| Bundle / contract integrity | CERT-4A-001–013, VAL-009, VAL-014 |
| Artifact integrity | VAL-009, VAL-010 |
| Platform contract compatibility | contract_compatibility rollup |

---

## 4. Threshold governance (DR-04 draft resolution)

### 4.1 Principles

| Rule | Description |
|------|-------------|
| **Versioned thresholds** | Every numeric bound has `threshold_id`, `threshold_version`, and `effective_date` |
| **Class-specific** | Decision-grade thresholds are **tighter** than diagnostic attribution thresholds |
| **Provisional runtime** | `TBD_v1_runtime` in code is **not** a production gate until promoted via governance |
| **Traceable evidence** | Approved thresholds cite world families, lattice stratum, Monte Carlo tier, or external experiments |
| **No silent promotion** | Replacing `TBD_v1` requires owner sign-off per [validation_registry.md](validation_registry.md) §7 |

### 4.2 Threshold status lifecycle

| Status | Meaning |
|--------|---------|
| `TBD_v1_runtime` | Provisional bounds used in CI / synthetic runs; **not** production release gates |
| `research_only` | Documented for sandbox / Bayes-H worlds; no prod claim |
| `approved` | Governance-approved; may gate release per DR-06 policy |

### 4.3 Release-blocking matrix (default)

| `metric_class` | `can_block_release` (default) |
|----------------|-------------------------------|
| `decision_grade` | `true` (when threshold `approved`) |
| `diagnostic_attribution` | `false` |
| `trust_modifier` | `conditional` |
| `structural` | `true` |

### 4.4 Product overrides

A product surface that **markets channel-level attribution accuracy** may opt into stricter diagnostic gates via an explicit **attribution certification profile** (future ADR). Default Ridge BO profile does **not** opt in.

---

## 5. ReliabilityScorecard interpretation

Scorecard v1.1+ reports **separate scores** (see [reliability_scorecard.md](reliability_scorecard.md)):

| Field | Class aggregated |
|-------|------------------|
| `decision_reliability_score` | decision_grade capabilities |
| `attribution_diagnostic_score` | diagnostic_attribution capabilities |
| `structural_reliability_score` | structural capabilities |
| `trust_modifier_status` | trust_modifier capabilities (status object) |
| `overall_evidence_score` | Mean of all scored capabilities (legacy rollup) |
| `reliability_score` | **Deprecated alias** of `overall_evidence_score` — do not use alone for release |

### 5.1 Interpretation rules

1. **Decision-usable with weak coef:** If `decision_reliability_score` is high and decision-grade VAL rows pass under approved thresholds, the model may be **decision-usable** even when `attribution_diagnostic_score` is low — provided `trust_modifier_status` is acceptable.
2. **Attribution-unsafe when decision-usable:** Low coef/transform recovery means **do not** claim channel attribution parity in TrustReport or customer-facing copy.
3. **Blocked despite Δμ pass:** Severe drift (`VAL-012`) or identifiability (`WORLD-012` pattern) may set `trust_modifier_status: degraded` and block decision-usable classification.
4. **Curves / decomposition:** Diagnostic unless separately certified under an attribution profile.
5. **Do not collapse:** A single “reliability number” must not drive release without inspecting class scores.

### 5.2 Release readiness labels

`release_readiness_interpretation` prioritizes **structural pass** + **decision_reliability_score** + **trust_modifier_status**. Attribution failures alone must not yield “prod ready.”

---

## 6. TrustReport mapping (semantic)

| TrustReport signal | Source class |
|--------------------|--------------|
| `decision_safe` / optimizer gates | decision_grade + structural |
| Attribution confidence / coef warnings | diagnostic_attribution |
| Drift / collinearity / freshness warnings | trust_modifier |
| `approved_for_prod` | structural + governance (VAL-008, VAL-011, VAL-013) — not coef-only |

---

## 7. Evidence requirements by promotion

| Promotion | Minimum evidence |
|-----------|------------------|
| `TBD_v1_runtime` → `approved` (decision) | WORLD-008–012 + behavioral lattice + INV-056 report; optional 5F Monte Carlo |
| `TBD_v1_runtime` → `approved` (diagnostic) | Stricter coef/transform worlds; per-channel transform research worlds |
| Trust modifier thresholds | 5D/5E drift runner (VAL-012 full); WORLD-011 re-cert |

---

## 8. Related investigations

| ID | Title | Status |
|----|-------|--------|
| INV-056 | Exact recovery failure analysis | **Closed** — explained |
| INV-057 | Decision vs attribution threshold separation | Open — this document is primary deliverable |
| INV-058 | Transform/β compensation monitoring | Open — ongoing synthetic monitoring |
| INV-059 | ReliabilityScorecard metric-class refactor | **Closed** in code v1.1 |
| INV-060 | Monte Carlo threshold calibration | Partial — tier-0 recommendations published |

---

## 9. Roadmap

| Phase | Focus |
|-------|--------|
| **5D** ✅ | This document + registry + scorecard semantics |
| **5E** ✅ | [drift_detection.md](drift_detection.md), [trust_report_semantics.md](trust_report_semantics.md) |
| **5F** ✅ | [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md) |

---

## 10. References

- [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md)  
- [validation_registry.md](validation_registry.md)  
- [reliability_scorecard.md](reliability_scorecard.md)  
- [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) DR-04, DR-06
