# Bayesian Hierarchical Geo MMM — Architecture refinement (Track 4)

**Version:** `bayes_h_geo_refinement_v1.0.0`  
**Status:** Architecture alignment only — **no implementation**  
**Binding ADRs:** [bayes_h1](bayes_h1_decision_surface_preservation_adr.md) · [bayes_h2](bayes_h2_calibration_signal_mapping_adr.md) · [bayes_h2b](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) · [bayes_h2d](bayes_h2d_hierarchical_model_spec_adr.md) (**Accepted**)  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisite:** Reliability Program Phases **5C–5F** complete ([monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md))  
**Related:** [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) · [trust_report_semantics.md](trust_report_semantics.md) · [reliability_threshold_governance.md](reliability_threshold_governance.md)

> **Bayes-H1–H2d ADRs** and **VAL-BAYES-H2B-SMOKE** are complete. **Bayes-H3** research sandbox may start (labeled `RESEARCH ONLY — NOT DECISION GRADE`). **Production** Bayesian decisioning remains blocked.

---

## Purpose

The platform is now **contract-centric**, not model-centric. Ridge BO proved that **decision recovery** and **attribution recovery** diverge; TrustReport, metric classes, and release gates are defined. The largest risk for Bayesian Geo MMM is **not** “can we sample a hierarchical model?” — it is **building advanced modeling before proving it fits the contracts the platform already stabilized**.

This document refines **architecture alignment** so a future implementation **plugs in** to existing ABIs. It does **not** authorize code, sampling, or production decisioning.

---

## Executive principle

| Layer | May change | Must not change |
|-------|------------|-----------------|
| **Estimator** | PyMC hierarchy, priors, pooling | — |
| **Production decision contract** | — | DecisionSurface semantics, Estimand meaning, gate names, full-panel Δμ geometry |
| **Evidence ABI** | — | CalibrationSignal shape and replay path |
| **Readiness ABI** | — | TrustReport + release-gate rules (extensions only) |

**Rule:** Bayesian Geo MMM adds an **estimator adapter** and **TrustReport extensions**. It does **not** introduce parallel decide APIs, alternate Δμ definitions, or posterior-driven budget commands that bypass gates.

---

## 1. Platform compatibility

### 1.1 Integration pattern (no new contracts)

```text
CalibrationSignal (input)
        ↓
Bayesian hierarchical estimator  →  posterior / diagnostics (internal)
        ↓
Contract adapters (existing)  →  DecisionSurface, Estimand, TrustReport, promotion record
        ↓
Release gates (unchanged PolicyError / approved_for_prod semantics)
```

The Bayesian module **never** exposes a second “decide” surface. Training produces the same **artifact tiers** Ridge uses, plus **research-only** posterior blocks. `mmm decide` / planning simulate consume **DecisionSurface** built from declared estimands and fitted state — not raw posterior objects.

### 1.2 DecisionSurface

| Requirement | How Bayesian satisfies it without new contracts |
|-------------|--------------------------------------------------|
| Same entry points | `simulate` / `optimize_budget_via_simulation` unchanged |
| Full-panel Δμ | Counterfactual uses **training panel geometry** (all geos × periods in estimand mask) |
| Point estimate for decisions | **Posterior mean** (or MAP) of generative parameters feeds the **same** feature construction + simulate path as Ridge |
| Uncertainty | **Not** embedded in DecisionSurface point outputs; carried in TrustReport / research extensions |
| `decision_safe` | Still computed by governance from TrustReport + gates — not “always true because Bayesian” |

**Adapter responsibility:** Map \((\hat{\beta}_{g,c}, \hat{\text{transform}})\) from posterior summaries into the structures `ridge_context_from_fit` / planning already expect. No new JSON schema for “BayesianDecisionSurface.”

### 1.3 Estimand

| Estimand (existing registry) | Bayesian role |
|------------------------------|---------------|
| `full_panel_transform_estimand_mask` | Unchanged — replay and Δμ use same mask semantics |
| Geo-time ATT / lift | Declared in `experiment_truth` + Estimand; hierarchy informs **prior/likelihood**, not estimand definition |
| Optimizer regret | Truth in `decision_truth`; optimizer still runs on **DecisionSurface** |

**Rule:** Changing the estimand registry requires an ADR. Bayesian work **maps** posteriors to existing estimand IDs — it does not invent `bayesian_posterior_estimand_v1`.

### 1.4 CalibrationSignal

| Requirement | Approach |
|-------------|----------|
| Single evidence ABI | GeoX, CLS, future A/B tests materialize as **CalibrationSignal** payloads (units, lift definitions, freshness, scope) |
| No ad hoc coef targets in prod | Signals translate to **prior/likelihood terms** inside the estimator adapter |
| Replay path preserved | `load_calibration_units_from_json` → signal registry → Bayesian update terms |
| Freshness / quality | TrustReport modifiers (Phase 5D) — stale signals downgrade trust, not silent coef overwrite |

**Not allowed:** Writing experiment lift directly into “true β” fields consumed by decide without passing through CalibrationSignal + TrustReport.

### 1.5 TrustReport

| Field class | Ridge today | Bayesian extension (same TrustReport object) |
|-------------|-------------|-----------------------------------------------|
| `decision_safe` / readiness | Governance scorecard | Same gates; add **convergence** / **divergence** blockers |
| Attribution diagnostics | Coef warnings | **Shrinkage tables**, \(\hat{\tau}_c\), geo-level flags |
| Uncertainty | Limited (Ridge CI research) | **Posterior intervals** on \(\beta_{g,c}\), **coverage** flags |
| Metric class (5D) | decision vs diagnostic | Posterior coef intervals = **diagnostic**; Δμ recovery = **decision_grade** |

**Default tier:** `prod_decisioning_allowed: false` until Bayes-H4 + release review.

### 1.6 Release gates

| Gate | Behavior |
|------|----------|
| `approved_for_prod` | Unchanged semantics — Bayesian runs start **research_only** |
| `require_promoted_model_for_prod_decision` | Promotion record still required; fingerprint rules unchanged |
| `PolicyError` | Same error types — no `BayesianBypassGate` |
| ReliabilityScorecard | Bayesian stratum **additive** — does not replace Ridge scorecard for prod |

---

## 2. Hierarchy design

### 2.1 Geographic levels (DMA, state, region, national)

The panel has a single **`geo_id` column** per row (see `geo_truth.geos`). Hierarchy is a **tree over geo_ids**, not simultaneous columns.

| Level | Typical role | Modeling recommendation |
|-------|--------------|-------------------------|
| **DMA** (or finest geo) | **Primary likelihood unit** — one outcome series per DMA × week | \(\beta_{g,c}\) defined at this \(g\) |
| **State** | Aggregation / partial pooling parent | Optional intermediate node in geo tree |
| **Region** | Broader aggregation | Optional parent of state |
| **National** | **Hyper-parameters** \(\mu_c\), \(\tau_c\) — not a row in panel | Population mean and spread **across geos** |

**v1 recommendation:** Fit at **finest geo in panel** (usually DMA). Pool \(\beta_{g,c}\) toward \(\mu_c\) with optional **nested** pooling (DMA → state → region) only if `HierarchyDefinition`-style tree is declared and identified. **Do not** fit independent national and DMA rows as duplicate likelihood units.

### 2.2 What is pooled

| Quantity | Pool across geos? | Notes |
|----------|-------------------|--------|
| **Media coefficients** \(\beta_{g,c}\) | **Yes** (core) | \(\beta_{g,c} \sim \mathcal{N}(\mu_c, \tau_c^2)\) or non-centered parameterization |
| **National means** \(\mu_c\) | Hyper-prior | Shared across all geos for channel \(c\) |
| **Heterogeneity** \(\tau_c\) | Hyper-prior | Channel-specific shrinkage intensity |
| **Intercept** \(\alpha_g\) | **Optional** partial pooling | Separate from media; weaker pooling if geo baselines differ |
| **Adstock decay** | **National / channel** (v1) | One decay per channel at national level; geo-specific decay only in research worlds that prove identifiability |
| **Hill half / slope** | **National / channel** (v1) | Same as Ridge BO “shared transform” lesson from INV-056 — per-geo saturation is tier-2 research |
| **Control coefficients** | **Optional** geo-specific or national | Declare in world truth; default national for stability |
| **Experiment lift parameters** | **Estimand-scoped** | Pool only when CalibrationSignal scope is national; local GeoX → local likelihood term |

### 2.3 What is not pooled

| Quantity | Reason |
|----------|--------|
| **Raw spend / KPI** | Data — not parameters |
| **Experiment estimand** (ATT window, unit definition) | Contract — declared per CalibrationSignal |
| **DecisionSurface geometry** | Full panel layout is fixed |
| **Release-gate outcomes** | Logical — not statistical |
| **Privacy / policy flags** | Governance — not hierarchical priors |
| **Posterior draws as budget vectors** | Decision policy — blocked (see §7) |

### 2.4 Pooling modes (research toggles)

| Mode | Use |
|------|-----|
| **Full pooling** | All geos share \(\mu_c\) exactly (\(\tau_c \to 0\)) — sanity / lower bound |
| **Partial pooling** | Default Bayes-H target |
| **No pooling** | Independent \(\beta_{g,c}\) — upper bound; small-geo overfit reference |
| **Nested geo tree** | DMA → state → region means — only when tree validated on synthetic worlds |

---

## 3. Calibration integration

### 3.1 Principle: CalibrationSignal is the only front door

GeoX, CLS, and future A/B tests **do not** call the sampler directly. They produce **CalibrationSignal** records that the platform already understands:

- unit scope (geo list, time window)
- lift definition / estimand reference
- uncertainty / quality tier
- freshness
- source (`geox`, `cls`, `ab_test`, `replay`, …)

### 3.2 Mapping to the hierarchy (binding)

Formal rules, scope table, mechanism assignment (R1–R10), uncertainty, freshness, conflict, and TrustReport fields: **[bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md)**.

Summary:

| Evidence type | Typical scope | Default mechanism |
|---------------|---------------|-------------------|
| **GeoX** | geo-level | `likelihood_term` or local prior on \(\beta_{g,c}\) |
| **CLS** | national / region | `hyper-prior` or national calibration likelihood on \(\mu_c\) |
| **A/B** | national / channel if aligned | `hyper-prior`; else TrustReport-only |
| **Replay** | panel-aligned | `likelihood_term` / `calibration_penalty` via replay invariants |

```text
CalibrationSignal  →  signal registry  →  {mechanism, weight, scope, trace}
                                              ↓
                                    Bayesian fit (Bayes-H3+, research only)
```

### 3.3 Hierarchy propagation (binding)

Upward, downward, lateral propagation, multi-geo handling, sparse-geo conflict precedence, claim semantics, and TrustReport hierarchy fields: **[bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)**.

**Rejected:** A parallel “experiment_coef” object consumed by decide without Estimand + CalibrationSignal alignment.

---

## 4. Decision semantics

### 4.1 Posterior ≠ DecisionSurface

| Object | Role |
|--------|------|
| **Posterior** \(p(\theta \mid \text{data})\) | Internal Bayesian state — \(\theta = \{\beta_{g,c}, \mu_c, \tau_c, \text{transform}, \ldots\}\) |
| **DecisionSurface** | **Operational** API for counterfactuals — point-valued simulate/optimize on full panel |
| **TrustReport** | Readiness + uncertainty disclosure |

### 4.2 Full-panel Δμ remains canonical

Production Δμ:

1. Fix **estimand mask** (same as Ridge).
2. Use **posterior mean** (default) or governance-approved point policy for generative parameters.
3. Run **existing** `simulate` / planning path.
4. Compare to truth on synthetic worlds (VAL-004 class — **decision_grade**).

**Uncertainty:** Report \(\Delta\mu\) credible intervals in **TrustReport / research artifacts** — not as the primary returned `delta_mu` for prod decide unless an explicit, gated policy is approved in Bayes-H5 (default: **no**).

### 4.3 Optimizer semantics

Optimizer uses **same** simulation optimizer as Ridge worlds:

- Input: DecisionSurface point state
- Output: allocation vs `decision_truth.true_optimal_budget`
- VAL-005 remains **decision_grade**

**Rejected:** Optimize expected utility over posterior draws as default prod behavior.

---

## 5. Validation requirements — new world families

Before implementation (Bayes-H3+), extend **GroundTruthWorld** / materializer with hierarchical truth blocks. Suggested `scenario_tags`: `framework:bayesian_hierarchical`, `bayes_h2:*`.

| World family | Truth declared | What it proves |
|--------------|----------------|----------------|
| **partial_pooling** | Known \(\mu_c\), \(\tau_c\), all \(\beta_{g,c}\) | Shrinkage magnitude vs truth |
| **sparse_dma** | Many small geos + few large geos | Small geos borrow; large geos deviate |
| **geo_heterogeneity** | High \(\tau_c\) vs low \(\tau_c\) strata | \(\tau\) recovery / ordering |
| **experiment_prior_local** | GeoX-like signal on subset of geos | Local prior/likelihood recovery |
| **experiment_prior_global** | National lift on \(\mu_c\) | Global prior recovery |
| **nested_geo_tree** | DMA → state → region indices | Nested pooling (tier 2) |
| **replay_calibration_bayes** | `experiment_truth` + replay units | CalibrationSignal → posterior shift |
| **identifiability_stress** | Collinear channels at geo level | Trust modifiers; no false prod ready |
| **convergence_negative** | Deliberately weak identification | TrustReport blocks; gates fail |

Each world must include:

- `geo_truth` with explicit `geos`, weights, optional `hierarchy` tree
- `media_truth.coefficients` or `hierarchical_media_truth` with per-geo \(\beta_{g,c}\) and national hyper-parameters
- `experiment_truth` aligned with CalibrationSignal certification
- `artifact_truth.expected_gates` and `expected_warnings`

**Materialization:** Rich DGP (not constant panel) for Bayes-H2 — see [dgp_materialization.md](dgp_materialization.md).

---

## 6. Reliability requirements (pre-implementation gate)

No Bayesian code in production paths until **all** rows below have a **tier-1+** (N≥100) Monte Carlo or dedicated Bayes-H4 certification plan. Metric classes per [reliability_threshold_governance.md](reliability_threshold_governance.md).

| Validation | Metric class | Pass intent |
|------------|--------------|-------------|
| **Pooling recovery** | diagnostic_attribution | \(\mu_c\), \(\tau_c\), \(\beta_{g,c}\) vs truth within MC bands |
| **Shrinkage monotonicity** | diagnostic_attribution | Small geos shrink more than large geos (stochastic ordering) |
| **Calibration recovery** | decision_grade / trust | Signal → posterior shift matches injected experiment truth |
| **Uncertainty calibration** | trust_modifier | 90% intervals cover ≥ 85% on synthetic (tier-dependent) |
| **Posterior coverage** | trust_modifier | PPC / coverage on held-out geo-periods |
| **Convergence** | structural / trust | \(\hat{R}\), ESS, divergences — fail closed |
| **Δμ recovery** | decision_grade | Full-panel Δμ vs `decision_truth` (primary) |
| **Optimizer recovery** | decision_grade | Same as VAL-005 on Bayes point state |
| **TrustReport compatibility** | structural | Required fields populated; `prod_decisioning_allowed` false until H5 |
| **Release-gate compatibility** | structural | No promotion without explicit approval |
| **Ridge baseline parity** | informational | Same worlds — Bayes vs Ridge on **decision_grade** metrics |

**Explicit:** Coef recovery may **fail** while Δμ passes — thresholds must follow 5D/5F policy (diagnostic vs decision).

### 6.1 Phased gate (aligned with Bayes-H*)

| Phase | Gate |
|-------|------|
| **Bayes-H1** | This document + ADR sign-off — no sampler |
| **Bayes-H2** | World families materialize + validate L1–L3 |
| **Bayes-H3** | Research sampler + TrustReport artifact |
| **Bayes-H4** | Reliability certifications on worlds above |
| **Bayes-H5** | Production candidacy review — optional posterior-in-decide policies |

---

## 7. Explicit non-goals

| Non-goal | Rationale |
|----------|-----------|
| **New decision surface** | Would fork platform ABI |
| **Attribution-first design** | INV-056 — decision metrics gate prod; coef diagnostic |
| **Posterior mean as default budget recommendation** | Decisions use DecisionSurface point policy; draws are research |
| **Bypassing TrustReport** | All paths through readiness + metric classes |
| **Nevergrad / state-space / dynamic coef** | Out of scope per platform direction |
| **Per-geo adstock/Hill in v1** | INV-056 shared-transform lesson |
| **Production Bayesian decide** | Before Bayes-H4 + release gates |
| **New VAL IDs without registry ADR** | Extend existing VAL rows + Bayes stratum |
| **Causal claims from hierarchy** | Observational pooling ≠ incrementality |

---

## 8. Recommended implementation order (after Bayes-H1 ADR)

1. ~~**Bayes-H1 ADR**~~ — ✅ [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md)  
2. ~~**Bayes-H2 — CalibrationSignal Mapping ADR**~~ — ✅ [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md)  
3. ~~**Bayes-H2b — Hierarchical Experiment-Prior Scope Rules**~~ — ✅ [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md)  
4. **Materialize `WORLD-BAYES-*`** + runner 002 — **next** ([catalog](../BAYES_H2B_VALIDATION_WORLDS_001.md) ✅)  
5. **Bayes-H2d** model design spec — blocked until VAL-BAYES CI smoke  
6. **Bayes-H3 prototype** — research PyMC; blocked until step 5  
7. **Bayes-H4 reliability** — §6 matrix on tier-1+ N  
8. **Bayes-H5** — production candidacy (if ever) — separate executive decision  

**Not in v1 path:** Nevergrad, state-space, dynamic coefficients, new transforms in prod.

---

## 9. Risk assessment (why refinement before code)

| Risk if implementation-first | Mitigation from this phase |
|------------------------------|----------------------------|
| Second decide API | §1 — adapter-only integration |
| GeoX breaks estimands | §3 — CalibrationSignal-only |
| Coef recovery becomes prod gate | §6 — 5D metric classes |
| Posterior replaces Δμ | §4 — DecisionSurface unchanged |
| Over-pooling / wrong tree | §2 — explicit pooled vs not pooled |
| Advanced model before contract proof | §5–6 — worlds before sampler |

The platform spent Phases **5C–5F** stabilizing **how** to judge reliability. Bayesian work should extend that grammar, not invent a new one.

---

## 10. References

- [platform_roadmap.md](platform_roadmap.md) — Track 4  
- [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) — Bayes-H1–H5  
- [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md) — shared transform / Δμ vs coef  
- [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md) — threshold policy  
- [trust_report_semantics.md](trust_report_semantics.md) — green/yellow/red  
- [groundtruth_contract.md](groundtruth_contract.md) — world truth extensions  
- [hierarchical_borrowing.md](../02_concepts/hierarchical_borrowing.md) — Ridge hierarchy (distinct from Bayes partial pooling)  
- Investigations: INV-063–069, INV-066 (experiment priors), INV-064 (pooling worlds)
