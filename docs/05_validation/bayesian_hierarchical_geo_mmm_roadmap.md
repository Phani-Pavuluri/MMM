# Bayesian Hierarchical Geo-Level MMM — Research Sandbox roadmap

**Status:** Planning only — **not implemented**, **not production-ready**  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Bayes-H1 (complete):** [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md)  
**Bayes-H2 (complete):** [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md) · [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md)  
**Bayes-H2b (complete):** [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)  
**Catalog:** [WORLDS_001](../BAYES_H2B_VALIDATION_WORLDS_001.md) ✅ · **Runner:** [RUNNER_002](../BAYES_H2B_VALIDATION_RUNNER_002.md) ✅  
**Fixtures:** `validation/worlds/WORLD-BAYES-*/` ✅  
**Bayes-H2d (complete):** [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md) ✅  
**Next:** Bayes-H3 **research sandbox** only (`RESEARCH ONLY — NOT DECISION GRADE`)  
**Does not change:** existing Bayesian modules, prod gates, or Ridge production path

---

## Purpose

Build a **future** Bayesian MMM path that models **local geo effects** and **national/global effects jointly**, using **partial pooling**, **experiment-informed priors**, and **platform-contract compatibility**.

Posterior uncertainty feeds **TrustReport** and research artifacts by default — **not** direct production budget decisions until release gates explicitly approve.

---

## Core model direction

| Element | Description |
|---------|-------------|
| **Geo-level outcome model** | Outcome modeled at geo × time (and controls) |
| **Geo-specific media coefficients** | Channel effects vary by geo |
| **National/global hyper-distribution** | Channel-level population distribution over geos |
| **Partial pooling** | Local coefficients shrink toward national average |
| **Large / high-signal geos** | Can deviate more when data supports it |
| **Small / noisy geos** | Borrow strength from global distribution |
| **Experiment evidence** | Enters as priors or likelihood terms (scope-dependent) |
| **Uncertainty output** | Posterior feeds TrustReport; not prod decide by default |

### Target conceptual structure

For each **geo** \(g\) and **channel** \(c\):

- Local media coefficient \(\beta_{g,c}\) is geo-specific.
- \(\beta_{g,c} \sim \mathcal{N}(\mu_c, \tau_c^2)\) (or equivalent hierarchical parameterization).
- \(\mu_c\) = national/channel-level mean (global average effect).
- \(\tau_c\) = heterogeneity across geos.
- Experiments can inform **local** priors (geo-scoped evidence) or **global** priors (national lift) depending on estimand scope.

---

## Expected strengths

- Captures **regional heterogeneity** without independent per-geo overfitting.
- **Stabilizes** small/noisy geos through partial pooling.
- Supports **local experiment calibration** aligned with GeoX and replay semantics.
- Produces **posterior uncertainty** for TrustReport and diagnostics.
- Better foundation for **geo-specific planning** narratives (research tier).
- Aligns naturally with **GeoX** and **experiment-informed MMM** when evidence is declared via CalibrationSignal.

---

## Trade-offs and risks

| Trade-off | Implication |
|-----------|-------------|
| Geo-level panels required | Spend/outcome at geo granularity; materialization burden |
| Heavier computation | Sampling cost; CI/runtime policy separate from Ridge |
| More diagnostics | PPC, convergence, divergences, geo-level shrinkage tables |
| Harder identifiability | Collinearity and weak geos compound |
| Convergence risk | Posterior geometry, funnel, label switching |
| Not automatically causal | Observational hierarchy ≠ incrementality |
| **Not production-ready** | Until Bayes-H1–H4 + release-gate review pass |

---

## Platform contract requirements (binding)

The Bayesian path **may change the estimator**; it **may not change the production decision contract**.

| Contract | Requirement |
|----------|-------------|
| **DecisionSurface** | Same simulate/optimize interfaces; full-panel Δμ remains canonical prod surface |
| **Estimand** | Declared estimands (geo-time ATT, full-panel Δμ) unchanged in meaning |
| **CalibrationSignal** | Experiment/replay evidence via structured signals — no ad hoc coef targets |
| **TrustReport** | Posterior summaries, convergence, coverage flags in research/prod-candidate tiers |
| **Release-gate semantics** | `approved_for_prod`, promotion, PolicyError — unchanged rules |
| **Full-panel Δμ** | Production counterfactual on training panel geometry |

**Explicit rule:** Research Bayesian outputs must not bypass TrustReport, release gates, or alternate production decision APIs.

---

## Required validation before production candidacy

All items below are **future** reliability-program requirements (Track 2), not satisfied today:

| Validation | Intent |
|------------|--------|
| Hierarchical synthetic worlds | Known geo-level ground truth |
| Geo heterogeneity recovery | Recover \(\mu_c\), \(\tau_c\), and \(\beta_{g,c}\) where identified |
| Partial pooling recovery | Shrinkage toward national mean matches truth |
| Small-geo shrinkage validation | Noisy geos borrow strength; not spurious divergence |
| Experiment-prior recovery | Local/global experiment terms match injected evidence |
| Posterior calibration checks | Calibration of uncertainty statements |
| PPC diagnostics | Posterior predictive checks on held-out structure |
| Convergence diagnostics | \(\hat{R}\), ESS, divergences — gated thresholds TBD |
| Δμ recovery | Full-panel counterfactual vs truth |
| Optimizer recovery | Budget path uses same Estimand as Ridge worlds |
| TrustReport compatibility | Artifact tier, warnings, readiness fields |
| Release-gate compatibility | No silent prod promotion |
| Ridge baseline comparison | Same reliability worlds — Bayesian must beat or match on declared metrics |

**Non-goal:** Do **not** build Bayesian **production decisioning** before reliability-world validation and release-gate approval (see Bayes-H5).

---

## Research Sandbox phases (Bayes-H*)

### Bayes-H1 — DecisionSurface preservation ✅

| Deliverable | Status |
|-------------|--------|
| [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md) | **Accepted** |
| [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) | Architecture alignment |
| PyMC / priors / likelihood / samplers | **Out of scope** — explicitly deferred |

### Bayes-H2 — CalibrationSignal mapping ✅

| Deliverable | Status |
|-------------|--------|
| [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md) | **Accepted** |
| Scope → mechanism table, conflict/freshness/uncertainty rules | In ADR |
| Validation world specs (7 worlds) | Defined — materialize in Bayes-H2b |

### Bayes-H2b — Scope propagation ✅

| Deliverable | Status |
|-------------|--------|
| [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) | **Accepted** |
| Upward/downward/lateral rules, claim semantics, TrustReport hierarchy fields | In ADR |
| `WORLD-BAYES-*` specifications | ✅ [BAYES_H2B_VALIDATION_WORLDS_001.md](../BAYES_H2B_VALIDATION_WORLDS_001.md) |

### Bayes-H2c — Materialization + runner ✅

| Deliverable | Notes |
|-------------|--------|
| Materialize seven `WORLD-BAYES-*` bundles | ✅ `validation/worlds/world_catalog.index.json` |
| [BAYES_H2B_VALIDATION_RUNNER_002](../BAYES_H2B_VALIDATION_RUNNER_002.md) | ✅ Contract |
| `hierarchy_evidence_validator` + `VAL-BAYES-H2B-SMOKE` | ✅ |

### Bayes-H2d — Hierarchical model spec ✅

| Deliverable | Notes |
|-------------|--------|
| [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md) | **Accepted** — \(\mu_c\), \(\tau_c\), \(\beta_{g,c}\), observation model (conceptual), TrustReport mapping |
| Geo/channel hierarchy, pooling modes, estimand alignment | In ADR §2–§4 |
| TrustReport / DecisionSurface / Release Gate implications | In ADR §7–§10 |

**Exit:** ADR accepted; **no PyMC**; authorizes **research sandbox** only.

### Bayes-H2 — Synthetic hierarchical worlds

| World type | Truth encoded |
|------------|---------------|
| Known geo-level coefficients | Per-geo \(\beta_{g,c}\) in `coefficient_truth` |
| Small / noisy geos | Low signal, high noise axes |
| Large / high-signal geos | Stable deviation from \(\mu_c\) |
| Local experiment evidence | `experiment_truth` scoped to geo |
| Partial-pooling ground truth | \(\mu_c\), \(\tau_c\) in metadata / dedicated truth section |

**Exit:** GroundTruthWorld bundles materialize geo panels; catalog entries in world catalog.

### Bayes-H3 — Research-only PyMC implementation

| Deliverable | Notes |
|-------------|--------|
| Posterior sampling | PyMC (Stan remains stub per INV-049) |
| Convergence diagnostics | Emitted on extension artifact |
| PPC | Research-tier reports |
| TrustReport-compatible artifact | `prod_decisioning_allowed: false` |

**Exit:** Runs on Bayes-H2 worlds in CI (research job); no prod decide.

### Bayes-H4 — Reliability certification

| Check class | Examples |
|-------------|----------|
| Shrinkage recovery | Small geos pulled toward \(\mu_c\) |
| Heterogeneity recovery | \(\tau_c\) and ranked geo deviations |
| Posterior coverage | Credible intervals vs truth |
| Experiment-prior calibration | VAL-006 analog for Bayesian likelihood |
| Δμ recovery | VAL-004 on hierarchical worlds |
| Ridge comparison | Same worlds, documented deltas |

**Exit:** ReliabilityScorecard stratum for `framework:bayesian_hierarchical` (future).

### Bayes-H5 — Sandbox model-spec improvement ✅

| Deliverable | Status |
|-------------|--------|
| [bayes_h5_model_spec_improvement_adr.md](bayes_h5_model_spec_improvement_adr.md) | **Accepted** (2026-06-01) — transforms, priors, diagnostics, validation plan |
| Implementation (`bayes_h5_sandbox_spec_v1`) | **Not authorized** until H5 worlds + pilot |
| Production promotion | **Blocked** |

**Exit:** Authorizes **spec direction** only; H3 MVP remains running code until gated implementation lands.

### Bayes-H5n — Shadow-policy recommender ✅

| Deliverable | Status |
|-------------|--------|
| `h5_shadow_policy_recommender.py` + sample-panel artifact | **Complete** — [INV-H5N](../06_investigations/INV-H5N_SHADOW_POLICY_RECOMMENDER.md) |
| Production Bayes / optimizer / DecisionSurface | **Not in scope** |

Maps collinearity, sparsity, convergence, weak ID, and calibration availability into explicit channel + geometry + sampler policy recommendations with rationale, forbidden claims, and blocked options. Sample panel recommends H5m drop-tv frozen policy. See [H5 ADR § H5n](bayes_h5_model_spec_improvement_adr.md#h5n-shadow-policy-recommender-inv-h5n).

### Bayes-H5o — Second real-panel shadow ✅

| Deliverable | Status |
|-------------|--------|
| `benchmark_geo_panel_v1` + recommender → freeze → replay | **Complete** — [INV-H5O](../06_investigations/INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md) |
| Production Bayes | **Not in scope** |

One panel only (`examples_mmm_benchmark_geo_panel_v1`); keep-all policy under low collinearity; converged replay. Do not batch panels.

### Bayes-H5p — Shadow workflow audit gate ✅

| Deliverable | Status |
|-------------|--------|
| H5l–H5o evidence summary + expansion/stop criteria | **Complete** — [AUDIT-H5P](../audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md) |
| Production Bayes | **Still blocked** |

Checkpoint before H5q+ panel expansion: recommender → freeze → replay discipline; stop on `do_not_run` / failed convergence.

### Bayes-H5r — Sparse-channel remedy replay ✅

| Deliverable | Status |
|-------------|--------|
| Same triangulation panel; drop_sparse_channels (radio) | **Complete** — [INV-H5R](../06_investigations/INV-H5R_SPARSE_CHANNEL_REMEDY_REPLAY.md) |
| H5q failed → H5r converged | **Documented** in [comparison JSON](../05_validation/archives/BAYES_H5R_REMEDY_COMPARISON_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json) |

### Bayes-H5q — Third real-panel shadow ✅

| Deliverable | Status |
|-------------|--------|
| Triangulation panel (8 geos, 4 channels, calibration stub) | **Complete** — [INV-H5Q](../06_investigations/INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md) |
| Convergence | **failed_convergence** (14 div) — recorded honestly; not evidence-promotable |
| Production Bayes | **Still blocked** |

### Bayes-H5 — Production candidacy review

| Gate | Rule |
|------|------|
| Prerequisite | Bayes-H1–H4 pass with versioned thresholds |
| Prod decisioning | **Off** until explicit ADR + release-gate approval |
| DecisionSurface | Decide whether posterior **mean** vs **draw** policy feeds prod simulate/optimize |
| Default | Research-only TrustReport; Ridge remains prod canonical |

**Exit:** Optional promotion path documented — not automatic.

---

## Relationship to existing investigations

| ID | Link |
|----|------|
| INV-032 | Prod Bayesian decide blocked — remains until Bayes-H5 |
| INV-033 | Experiment likelihood research-only |
| INV-034 | Hierarchy partial pooling research-only — parent to Bayes-H track |
| INV-063–069 | Bayes-H design, validation, contracts (this track) |

---

## Explicit non-goals

- **No** Bayesian production budget decisioning before reliability-world validation and release-gate approval.
- **No** change to current Bayesian module behavior or prod gates in this planning phase.
- **No** claim of causal incrementality from hierarchical pooling alone.
- **No** replacement of full-panel Δμ as the production decision surface without contract ADR.

---

## Related documentation

| Document | Role |
|----------|------|
| [platform_roadmap.md](platform_roadmap.md) | Master 5-track roadmap |
| [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) | Track 2 phases; Bayes-H2/H4 tie-in |
| [bayesian.md](../02_concepts/bayesian.md) | Current research Bayesian surface (unchanged) |
| [hierarchical_borrowing.md](../02_concepts/hierarchical_borrowing.md) | Partial pooling concepts |
| [open_investigations.md](../06_investigations/open_investigations.md) | INV-063–069 |

---

## Immediate platform priority (unchanged)

**Reliability Program** priority: Phases **5C–5F** complete (core architecture). **Track 4** Bayes-H may proceed with roadmap refinement — models must plug into DecisionSurface, Estimand, CalibrationSignal, TrustReport, ReleaseGate without redefining them. Tier-1 Monte Carlo (N=100) recommended before DR-04 threshold approval.
