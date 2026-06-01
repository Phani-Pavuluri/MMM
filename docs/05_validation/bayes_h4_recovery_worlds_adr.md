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
| `WORLD-BAYES-H4-SPARSE-GEO` | One sparse DMA; large true local deviation; **stress diagnostic** (report-only) |
| `WORLD-BAYES-H4-CONFLICTING-EVIDENCE` | Generative truth vs conflicting calibration stub; expect conflict diagnostics |

### H4c extended recovery worlds (research reliability map)

| World ID | Role | Expected use |
|----------|------|----------------|
| `WORLD-BAYES-H4C-CLEAN-RECOVERY` | recovery_candidate | Good \(\mu_c\), \(\beta_{g,c}\) recovery under favorable conditions |
| `WORLD-BAYES-H4C-CORRELATED-CHANNELS` | weak_identification | Collinearity warning; weaker channel-level recovery |
| `WORLD-BAYES-H4C-ADSTOCKED-MEDIA` | transform_mismatch | Outcome adstocked; MVP fits raw media |
| `WORLD-BAYES-H4C-SATURATION` | transform_mismatch | Outcome saturated; MVP semi_log linear |
| `WORLD-BAYES-H4C-WEAK-SIGNAL` | weak_identification | Low SNR; poor \(\beta\) recovery expected |
| `WORLD-BAYES-H4C-SPARSE-RECOVERY` | recovery_candidate | Sparse geo with sufficient weeks; not the H4 stress world |

**Artifact:** [BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json](archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json)  
**Runner:** `h4c_extended_recovery_pilot.py`

H4c answers: *where does the sandbox recover synthetic truth, and where does it fail?* It does **not** answer whether Bayesian MMM is production-ready.

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
| `shrinkage_ratio_sparse` | Mean \(\|\hat\beta_{g,c}-\hat\mu_c\| / \|\beta^{\text{true}}_{g,c}-\hat\mu_c\|\) for sparse geos (pool center = **posterior** \(\hat\mu_c\)) | Expect \(< 1\) when partial pooling pulls toward estimated hyper-mean |
| `shrinkage_ratio_sparse_vs_true_mu` | Legacy H4a/H4b: same with **true** \(\mu_c^\*\) in denominator/numerator reference | Report-only; may exceed 1 when \(\hat\mu_c \neq \mu_c^\*\) |
| `sparse_shrinkage_decomposition` | Per \((g,c)\) distances, \(\hat\tau_c\), interval width | INV-H4-001 diagnostic |
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

## 9. Initial threshold pilot results (Bayes-H4a)

**Artifact:** [archives/BAYES_H4_THRESHOLD_PILOT_20260601.json](archives/BAYES_H4_THRESHOLD_PILOT_20260601.json)  
**Runner:** `mmm.research.bayes_h3_sandbox.h4_threshold_pilot`  
**Status:** Provisional — **warning/report-only**; does **not** authorize production.

| Pilot fact | Value / rule |
|------------|----------------|
| Thresholds | **Provisional** — derived from one PyMC pass per world (`draws=200`, `tune=200`, `chains=2`, `target_accept=0.92`, seed `4400`) |
| Hard gates | **None** — `hard_gate: false` in artifact |
| Production | **Blocked** — `production_promotion: false`, `approved_for_prod: false` on all rows |
| INV-071 | **Open** — requires repeated pilots + extended worlds before tightening bands |

### Observed bands (2026-06-01 pilot)

| Metric | Pilot observation | Provisional use |
|--------|-------------------|-----------------|
| `beta_gc_mae` | max ≈ **0.47** (warn band ≈ 0.70) | **Report** — monitor trend; no merge fail |
| `mu_c_mae` | max ≈ **0.39** (warn band ≈ 0.58) | **Report** — toy μ priors not production-calibrated |
| `beta_gc_coverage_90` | range **0.0–0.5** | **Directional only** — do not require exact 90% on tiny panels |
| `shrinkage_ratio_sparse` (legacy field in H4a JSON) | **2.57** vs true \(\mu^\*\) | **Superseded** — see primary-metric H4b-refresh; was legacy diagnostic, not pooling failure |
| `conflict_warnings` | **Non-empty** on conflict world | **Required** for conflict world (diagnostic) |
| Production flags | all **false** | **Fail** if any prod flag true |

### Governance reminder

Backend choice (PyMC) and recovery metrics remain **research-only**. Passing the pilot does not amend Bayes-H1/H3 promotion blocks.

---

## 10. Repeated pilot results (Bayes-H4b)

**Artifacts:**

| Artifact | Role |
|----------|------|
| [BAYES_H4_REPEATED_PILOT_20260601.json](archives/BAYES_H4_REPEATED_PILOT_20260601.json) | Original H4b (legacy metric conflated with primary field) |
| [BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json](archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json) | **H4b-refresh** — authoritative extended evidence with primary + legacy shrinkage |

**Runner:** `mmm/research/bayes_h3_sandbox/h4_repeated_pilot.py`  
**Sampler:** extended profile (`draws=600`, `tune=600`, `chains=4`, `target_accept=0.95`); fixed `panel_seed=4400`; `nuts_seeds` ∈ {4400, 4401, 4402}.

| Setting | Value |
|---------|--------|
| Production promotion | **Blocked** — all production flags false |
| Hard gates | **None** — `interpretation.hard_gate: false` |
| INV-071 | **Open** — true-effect recovery thresholds (pooling mechanics resolved per INV-H4-001 disposition) |

### Sparse shrinkage classification

See artifact `interpretation.sparse_shrinkage_summary.classification` and `sparse_shrinkage_distribution`.

| Classification | Meaning |
|----------------|---------|
| `resolved_under_longer_sampling` | Majority of extended runs show `shrinkage_ratio_sparse` &lt; 1 |
| `still_unstable` | High variance across seeds — likely fast/extended MCMC instability |
| `likely_model_prior_or_world_design` | Extended runs still ≥ H4a fast reference — MVP partial pooling or sparse world design |
| `likely_world_design_or_metric` | All runs ratio ≥ 1 — review sparse weeks and metric definition |
| `inconclusive` | Mixed or missing values — keep INV-071 open |

**H4a reference (legacy):** fast pilot `shrinkage_ratio_sparse_vs_true_mu ≈ 2.57`.

### Observed H4b-refresh outcome (2026-06-01, primary metric artifact)

| Metric | Extended pilot (3 seeds) | Classification |
|--------|--------------------------|----------------|
| `shrinkage_ratio_sparse` (primary vs \(\hat\mu_c\)) | **0.63, 0.68, 0.69** (mean ≈ **0.66**) — all &lt; 1 | **`pooling_toward_posterior_mu_stable`** |
| `shrinkage_ratio_sparse_vs_true_mu` (legacy) | **2.57, 2.65, 2.73** — all ≥ 1 | **`weak_recovery_vs_true_mu`** (not a pooling gate) |
| `beta_gc_mae` / `mu_c_mae` | report in artifact | true-effect recovery **open** |
| `conflict_warning_pass_rate` | **1.0** | conflict world OK |
| Production flags | all **false** | |

**Interpretation:** Extended sampling **confirms pooling toward learned \(\hat\mu_c\)** on the sparse world (primary). Legacy vs \(\mu_c^\*\) remains poor — judge recovery with `mu_c_mae`, `beta_gc_mae`, coverage, and legacy shrinkage separately. **Do not** treat primary &lt; 1 as proof of true-effect recovery. **Disposition C+A accepted** — see [INV-H4-001 §11](../06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md).

### H4b-disposition — metric and recovery posture (accepted)

| Policy | Rule |
|--------|------|
| **Authoritative pooling artifact** | [BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json](archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json) supersedes original H4b JSON for **pooling** interpretation |
| **Primary shrinkage** | Pooling diagnostic only (posterior \(\hat\beta\) vs learned \(\hat\mu_c\)); ratio &lt; 1 = mechanical pooling toward learned center |
| **Recovery evidence** | `beta_gc_mae`, `mu_c_mae`, `beta_gc_coverage_90`, `shrinkage_ratio_sparse_vs_true_mu` (legacy) |
| **Sparse world** | `WORLD-BAYES-H4-SPARSE-GEO` remains **report-only / stress-test** until worlds and thresholds recalibrated; **not** a hard recovery gate |
| **Production** | **Blocked** — no promotion from H4 pilots or disposition |

---

## 11. INV-H4-001 sparse pooling investigation

**Doc:** [INV-H4-001_SPARSE_POOLING_BEHAVIOR.md](../06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md)  
**Code:** `sparse_shrinkage_metrics.py` · `sparse_pooling_investigation.py` · diagnostic sparse variants in `recovery_worlds.py`

| Item | Status |
|------|--------|
| Metric audit (posterior \(\hat\mu_c\) vs true \(\mu_c\)) | **Complete** |
| Posterior geo/channel index metadata | **Complete** |
| Diagnostic world variants (4) | **Complete** (research-only) |
| INV-H4-001b variant sweep | **Closed** |
| INV-H4-001 disposition | **C+A accepted** — no production promotion |
| INV-071 true-effect thresholds | **Open** |
| Bayes-H4c extended pilot | **Complete** (reliability map) — see [H4C JSON](archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json) |
| INV-071 | **Open** — true-effect recovery thresholds |

---

## 13. H4c extended recovery pilot (reliability map)

**Artifact:** [archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json](archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json)

| Metric policy | Rule |
|---------------|------|
| Primary shrinkage | Pooling toward \(\hat\mu_c\) only (H4b-disposition) |
| Legacy shrinkage | True \(\mu_c^\*\) recovery diagnostic — not a pooling gate |
| Sparse stress | `WORLD-BAYES-H4-SPARSE-GEO` remains **outside** H4c catalog; report-only |

**Pilot role:** Produce a **reliability map** (classifications per world). No hard gates. No production promotion.

---

## 14. Consequences

- **Complete (scaffolding):** `recovery_worlds.py`, `recovery_runner.py`, `tests/research/test_bayes_h4_recovery_worlds.py`.  
- **Complete (H4a pilot):** `h4_threshold_pilot.py`, committed pilot JSON, `tests/research/test_bayes_h4_threshold_pilot.py`.  
- **Complete (H4b repeated pilot):** `h4_repeated_pilot.py`, original + [primary-metric refresh JSON](archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json), `tests/research/test_bayes_h4_repeated_pilot.py`.  
- **Complete (H4b-disposition):** metric policy C + sparse-world posture A accepted.  
- **Complete (H4c):** `h4c_recovery_worlds.py`, `h4c_extended_recovery_pilot.py`, H4c pilot JSON, `tests/research/test_bayes_h4c_extended_recovery_worlds.py`.  
- **Next:** Tune sparse/τ per disposition A; calibrate INV-071 true-effect bands.  
- **Not authorized:** Bayes-H3 production promotion, NumPyro backend, prod CI Bayesian jobs without research labeling.

**This ADR does not authorize production Bayesian decisioning.**
