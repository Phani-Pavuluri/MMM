# Bayes-H4 — Recovery Worlds for Bayes-H3 Research Sandbox

**ADR ID:** `bayes_h4_recovery_worlds_v1`  
**Title:** Bayes-H4 — Deterministic recovery worlds for hierarchical sandbox validation  
**Status:** **Accepted** (research validation scaffolding — does **not** authorize production Bayesian decisioning)  
**Date:** 2026-06-01  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisites:** [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md) · [bayes_h3_research_sandbox_backend_adr.md](bayes_h3_research_sandbox_backend_adr.md) · Bayes-H3 sandbox MVP fit ✅ · Bayes-H3 guardrails ✅  
**Governance:** [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md) · [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md)  
**Implementation:** `mmm.research.bayes_h3_sandbox.recovery_worlds` · `mmm.research.bayes_h3_sandbox.recovery_runner`  
**Related:** [BAYES_H2B_VALIDATION_WORLDS_001.md](../BAYES_H2B_VALIDATION_WORLDS_001.md) (evidence routing — orthogonal to H4 generative recovery)

---

## 1. Purpose

Validate whether the **Bayes-H3 research-only hierarchical sandbox** (`run_sandbox_fit` / PyMC MVP) can **recover known generative truth** on **deterministic synthetic worlds** before any discussion of production promotion.

Bayes-H3 answered: *can the sandbox run safely under fences?*  
Bayes-H4 answers: *does the sandbox behave scientifically on controlled truth?*

---

## 2. Non-goals

| Non-goal | Notes |
|----------|--------|
| Production Bayesian MMM | Ridge remains prod estimator |
| DecisionSurface / optimizer / recommendations | [Bayes-H1](bayes_h1_decision_surface_preservation_adr.md) |
| `approved_for_prod: true` | All H4 artifacts research-only |
| Replacing `WORLD-BAYES-*` hierarchy evidence worlds | H2b = routing contract; H4 = generative recovery |
| Calibrated pass/fail thresholds for promotion | Thresholds **TBD** until pilot characterization |
| Real-panel or client data | Synthetic only |

---

## 3. Recovery estimands

| Estimand | Symbol / object | Recovery question |
|----------|-----------------|-------------------|
| Channel hyper-mean | \(\mu_c\) | Posterior mean close to true \(\mu_c\)? |
| Pooling scale | \(\tau_c\) | Posterior \(\tau_c\) reflects true heterogeneity? |
| Geo-channel effect | \(\beta_{g,c}\) | Posterior mean close to true \(\beta_{g,c}\)? |
| Shrinkage | \(\|\beta_{g,c}-\mu_c\|\) vs truth | Sparse/noisy geo pulled toward \(\mu_c\)? |
| Evidence conflict | CalibrationSignal stubs | Diagnostic warning when stub conflicts with generative truth? |

All estimands are **diagnostic** — reported in `h4_recovery` block and TrustReport-shaped stub only.

---

## 4. Initial world catalog

| World ID | Intent |
|----------|--------|
| `WORLD-BAYES-H4-SIMPLE-POOLING` | Low heterogeneity; \(\beta_{g,c}\) near \(\mu_c\); baseline recovery |
| `WORLD-BAYES-H4-SPARSE-GEO` | One sparse DMA; large true local deviation; expect shrinkage toward \(\mu_c\) |
| `WORLD-BAYES-H4-CONFLICTING-EVIDENCE` | Generative truth vs conflicting calibration stub; expect conflict diagnostics |

Each world exposes: panel, geo hierarchy, channels, `true_mu_c`, `true_tau_c`, `true_beta_gc`, noise \(\sigma\), optional calibration stubs, `expected_diagnostic_behavior`.

---

## 5. Known-truth parameters

Truth is **fixed per world** (deterministic seeds). Stored in `RecoveryWorldSpec.known_truth` and used by `recovery_runner.compute_recovery_metrics`.

Generative sketch (log scale, aligned with H3 MVP):

\[
\log y_{g,t} = \alpha_g + \sum_c \beta_{g,c}\, \tilde{x}_{g,t,c} + \varepsilon_{g,t}, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\]

Media \(\tilde{x}\) are standardized per channel for numerical stability.

---

## 6. Acceptance metrics (research)

| Metric | Definition | Initial use |
|--------|------------|-------------|
| `beta_gc_mae` | Mean absolute error of posterior \(\mathbb{E}[\beta_{g,c}]\) vs true | Report only; threshold TBD |
| `mu_c_mae` | MAE of posterior \(\mathbb{E}[\mu_c]\) vs true | Report only |
| `beta_gc_coverage_90` | Fraction of \((g,c)\) where true \(\beta_{g,c}\) lies in posterior 90% interval | Report only |
| `shrinkage_ratio` | \(\mathbb{E}[\|\beta^{\text{post}}_{g,c}-\mu_c\|] / \mathbb{E}[\|\beta^{\text{true}}_{g,c}-\mu_c\|]\) for sparse geos | Expect \(< 1\) for sparse world (slow tests) |
| `conflict_warnings` | Strings when calibration stub direction opposes generative truth | Expect non-empty for conflict world |

**Pass/fail gates for promotion are not defined in this ADR.**

---

## 7. Failure and warning conditions

| Condition | Severity | Action |
|-----------|----------|--------|
| Missing research-only labels on artifact | **Fail** | Block merge |
| `production_decision_surface` or optimizer paths present | **Fail** | Block merge |
| `rhat_max` non-finite (slow runs) | **Warn** | Investigate sampling |
| `beta_gc_mae` above pilot threshold | **Warn** | Record in open investigations |
| No conflict warnings on conflict world | **Warn** | Evidence diagnostic gap |
| Shrinkage ratio \(\geq 1\) on sparse world | **Warn** | Pooling may be weak |

---

## 8. Governance boundary

- All H4 runs use `run_sandbox_fit` only.  
- Outputs: `label: RESEARCH ONLY — NOT DECISION GRADE`, `approved_for_prod: false`, `prod_decisioning_allowed: false`, `decision_grade: false`.  
- **Promotion remains blocked** until: H4 pilot thresholds calibrated, reproducible DecisionSurface adapter proof, TrustReport production mapping, Promotion Gate checklist — per [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md).

---

## 9. Consequences

- **Next implementation:** `recovery_worlds.py`, `recovery_runner.py`, `tests/research/test_bayes_h4_recovery_worlds.py`.  
- **Not authorized:** Bayes-H3 production promotion, NumPyro backend, prod CI Bayesian jobs without research labeling.

**This ADR does not authorize production Bayesian decisioning.**
