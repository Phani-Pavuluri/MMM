# Bayes-H2d — Hierarchical Bayesian MMM Model Specification

**ADR ID:** `bayes_h2d_hierarchical_model_spec_v1`  
**Title:** Bayes-H2d — Hierarchical Bayesian Geo MMM Model Specification (architecture only)  
**Status:** **Accepted** (model spec — does **not** authorize PyMC, samplers, model classes, or production Bayesian decisioning)  
**Date:** 2026-05-29  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisites:** [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md) · [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md) · [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) · [BAYES_H2B_VALIDATION_WORLDS_001.md](../BAYES_H2B_VALIDATION_WORLDS_001.md) · [BAYES_H2B_VALIDATION_RUNNER_002.md](../BAYES_H2B_VALIDATION_RUNNER_002.md) · `VAL-BAYES-H2B-SMOKE` ✅  
**Governance:** [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md) · [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md)  
**Related:** [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) · [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) · [trust_report_semantics.md](trust_report_semantics.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) Decision 13

---

## Roadmap alignment gate (pre-authoring)

| Gate question | Answer |
|---------------|--------|
| **Tier** | 1 — Contract / architecture |
| **MIP goal** | MMM calibration ecosystem; trust-aware measurement |
| **Contracts touched** | DecisionSurface, Estimand, CalibrationSignal, TrustReport, Release Gates |
| **Failure mode reduced** | Premature PyMC; posterior-as-decision; ABI drift; ungrounded hierarchy |
| **Proof artifact** | This ADR; builds on `WORLD-BAYES-*` + `VAL-BAYES-H2B-SMOKE` |
| **Gate level** | Architecture Gate |
| **Research allowed?** | Yes — architecture-only |
| **Production promotion** | **Blocked** — no prod Bayesian decisioning |

**Explicit non-goals:** No PyMC; no samplers; no model classes; no production release; no posterior-to-optimizer path; no coefficient-to-optimizer path; no alternate DecisionSurface; no experiment-specific APIs; no Bayes-H3 **production** promotion.

**Next authorized step after acceptance:** Bayes-H3 **research sandbox** implementation (labeled `RESEARCH ONLY — NOT DECISION GRADE`).

---

## 1. Model purpose

### 1.1 What this model is for

The **hierarchical Bayesian Geo MMM** is a **research-sandbox estimator** candidate that:

- Models **weekly geo-level outcomes** with **channel-specific media effects** that vary across geographies.
- Uses **partial pooling** so small or noisy geos borrow strength from national/channel structure without independent per-geo overfitting.
- Integrates **experiment and replay evidence** only through **CalibrationSignal** and the Bayes-H2 / H2b signal registry (mechanisms R1–R10, propagation gates, claim levels).
- Produces **posterior summaries and diagnostics** for **TrustReport** and research artifacts.
- Feeds **production decisioning** only through the existing **DecisionSurface** adapter (point generative state → full-panel Δμ via `simulate()`), never through raw posteriors or coefficients.

### 1.2 What this model is not for

| Not for | Reason |
|---------|--------|
| Replacing Ridge as default prod estimator | Until Bayes-H4+ certification and Promotion Gate |
| Direct budget optimization on posteriors or β | [Bayes-H1](bayes_h1_decision_surface_preservation_adr.md) Decision 3 |
| Proving causal incrementality from observational hierarchy alone | Requires experiment evidence + estimand discipline |
| Bypassing TrustReport or Release Gates | Platform ABI |
| National experiment claims from single-DMA GeoX without gates | [Bayes-H2b](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) §8 |

### 1.3 Relationship to prior ADRs

| ADR | Role relative to H2d |
|-----|----------------------|
| **Bayes-H1** | DecisionSurface supremacy; posterior separation; optimizer contract |
| **Bayes-H2** | CalibrationSignal ingress; mechanism assignment |
| **Bayes-H2b** | Scope graph, propagation, influence classes, TrustReport hierarchy fields |
| **Bayes-H2d (this)** | **Generative/statistical model shape** — parameters, pooling, observation model concept |

Bayes-H2d **does not** re-open H1–H2b decisions. Implementation must **compose** them.

---

## 2. Hierarchy structure

### 2.1 Geographic ladder (binding)

```text
Panel row unit: (geo_id, week)  — finest geo in materialized panel (usually DMA)

Statistical hierarchy (conceptual):

  β_{g,c}  (local media effect, geo g × channel c)
      ↑ partial pool
  μ_c, τ_c  (national/channel hyper-mean and heterogeneity scale)
      ↑ optional nested nodes
  state / region  (summary & pooling diagnostics — not duplicate likelihood rows)
  national  (hyper-parameters & calibration targets — not a panel row)
```

**Channel** and **segment** cross-cut geography. **Time** is the observation index within each geo series.

### 2.2 Scope graph (contract, not assumed tree)

The **scope graph** from [Bayes-H2b](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) governs evidence propagation. The **generative hierarchy** uses the **panel geo_id** as the likelihood unit; state/region/national nodes inform **pooling and TrustReport summaries**, not parallel outcome series unless explicitly materialized as separate panel geos (forbidden for national as a row).

### 2.3 Pooling modes (research toggles)

| Mode | Specification use |
|------|---------------------|
| **Partial pooling (default)** | \(\beta_{g,c} \sim \mathcal{N}(\mu_c, \tau_c^2)\) or non-centered equivalent |
| **Full pooling** | \(\tau_c \to 0\) — sanity lower bound |
| **No pooling** | Independent \(\beta_{g,c}\) — upper bound / overfit reference |
| **Nested geo tree** | DMA → state → region means — only when tree declared in `hierarchy_spec` and identified on synthetic worlds |

### 2.4 Identifiability constraints (v1 spec)

| Constraint | Rule |
|------------|------|
| One likelihood row per `(geo_id, week)` | No duplicate national rows in panel |
| Shared transforms (adstock, saturation) | **National per channel** in v1 unless world proves geo-specific identifiability |
| Collinear channels | TrustReport flag; may exclude channels from decision-grade surfaces |
| Sparse geos | Borrow via \(\tau_c\); `sparse_geo_ids` in hierarchy spec → enhanced pooling diagnostics |

---

## 3. Parameter groups

Parameters are **internal generative state**. None are **decision surfaces**. DecisionSurface consumes **implied μ(plan), μ(baseline)** on full panel geometry after adapter mapping.

### 3.1 Local (geo × channel)

| Symbol / group | Meaning | Decision-facing? |
|----------------|---------|------------------|
| \(\beta_{g,c}\) | Incremental media effect for geo \(g\), channel \(c\) on modeling scale | **No** — adapter input only |
| \(\alpha_g\) | Geo intercept (optional) | **No** |
| Geo-specific controls | Optional \(\theta_{g,k}\) for control features | **No** |

### 3.2 Hyper (national / channel)

| Symbol / group | Meaning | Decision-facing? |
|----------------|---------|------------------|
| \(\mu_c\) | Population mean media effect for channel \(c\) across geos | **No** |
| \(\tau_c\) | Heterogeneity scale for channel \(c\) | **No** — TrustReport pooling |
| National transform params | Adstock decay, Hill half/slope per channel (v1 shared) | **No** |

### 3.3 Observation / noise

| Group | Meaning |
|-------|---------|
| \(\sigma_{g}\) or \(\sigma\) | Outcome noise scale (geo-specific optional) |
| Overdispersion | Count outcomes — research extension only with explicit estimand |

### 3.4 Evidence-linked (not free parameters)

| Construct | Source |
|-----------|--------|
| Likelihood increments from experiments | CalibrationSignal → signal registry → terms on \(\beta_{g,c}\), \(\mu_c\) per Bayes-H2/H2b |
| Replay-implied targets | Same ingress — replay path invariants preserved |

**Prohibited:** Standalone `experiment_lift_coef` parameters writable from orchestration without CalibrationSignal.

### 3.5 Frozen / external (not estimated by Bayes fit)

| Quantity | Role |
|----------|------|
| Panel spend, KPI, controls | Data |
| Estimand masks | Contract |
| Release-gate flags | Governance |
| DecisionSurface scenario definitions | Planning input |

---

## 4. Observation model (conceptual)

**No implementation code in this ADR.** The following is the **target generative story** for sandbox exploration.

### 4.1 Structural form (Gaussian v1)

For each geo \(g\), week \(t\), outcome \(y_{g,t}\) (e.g. revenue, transformed):

\[
y_{g,t} = \alpha_g + \sum_c f_c(x_{g,c,t}; \beta_{g,c}, \phi_c) + \sum_k \theta_{g,k} z_{g,k,t} + \epsilon_{g,t}
\]

where:

- \(f_c\) is the **media contribution** (adstock + saturation applied to spend \(x_{g,c,t}\)).
- \(\phi_c\) are **national/shared transform parameters** per channel (v1).
- \(\epsilon_{g,t} \sim \mathcal{N}(0, \sigma^2_{g})\) or homoscedastic \(\sigma^2\) for v1 sandbox.

### 4.2 Hierarchy on media effects

\[
\beta_{g,c} \sim \mathcal{N}(\mu_c, \tau_c^2)
\]

Non-centered parameterization is **recommended** for sampling stability (implementation note for Bayes-H3 sandbox only).

### 4.3 Evidence terms (conceptual additive updates)

After Bayes-H2 mechanism assignment, evidence enters as **one of**:

| Mechanism | Conceptual effect on generative model |
|-----------|--------------------------------------|
| `likelihood_term` | Additional likelihood factor on targeted parameters (e.g. Gaussian on implied lift) |
| `local prior` / `hyper_prior_style` | Prior mean/variance shift on \(\beta_{g,c}\) or \(\mu_c\) |
| `calibration_penalty` | Soft constraint comparable to Ridge replay penalty |
| `excluded` / `trust_report_only` | No term in likelihood |

**No parallel observation model** for experiments outside this table.

### 4.4 What is not modeled in v1 spec

| Excluded from v1 spec | Deferred |
|-----------------------|----------|
| Multiplicative seasonality state-space | Bayes-H4+ worlds |
| Dynamic media stock latent states | Research extension |
| Causal forests / uplift modules | Separate track if promoted |
| Product-conversion estimands on geo panel | `WORLD-BAYES-ESTIMAND-EXCLUDE` pattern |

---

## 5. Calibration evidence integration (CalibrationSignal only)

### 5.1 Ingress pipeline (binding)

```text
Approved evidence registry / replay materialization
        ↓
CalibrationSignal[]  (sole ingress)
        ↓
build_signal_registry()  — Bayes-H2 R1–R10
        ↓
apply_scope_propagation()  — Bayes-H2b gates & claim levels
        ↓
attach_evidence_terms_to_model_spec()  — conceptual (Bayes-H3 sandbox)
        ↓
fit → posterior (sandbox only)
```

### 5.2 Mapping summary

| Evidence class | Typical target | Spec reference |
|----------------|----------------|----------------|
| GeoX local | \(\beta_{g,c}\) for treated geos | Bayes-H2 R6; Bayes-H2b §8.8 |
| CLS national | \(\mu_c\) / national calibration | Bayes-H2 R7; Bayes-H2b §8.3 |
| Replay | Panel-aligned likelihood | Bayes-H2 R5, R10 |
| Conflicting same-scope | Downweight / fail-closed — no merge | Bayes-H2b §9 |
| Stale / missing SE / estimand mismatch | Exclude or diagnostic tier | Worlds STALE, MISSING-SE, ESTIMAND-EXCLUDE |

### 5.3 Validation before any fit code

`hierarchy_evidence_validator` + `VAL-BAYES-H2B-SMOKE` must pass **before** Bayes-H3 sandbox code merges. Fitter must **not** assign mechanisms independently of the registry contract.

---

## 6. Local / regional / national pooling semantics

| Level | Pooling role | Evidence may claim (after gates) |
|-------|-------------|----------------------------------|
| **DMA / geo** | Primary \(\beta_{g,c}\) likelihood unit | `directly_observed_experimental_evidence` at native scope |
| **State** | Partial pool parent; TrustReport summary | Gated upward summary only (coverage ≥ 0.60) |
| **Region** | Regional pooling evidence class | No single-DMA → region point mass |
| **National** | \(\mu_c, \tau_c\) hyper level | National calibration; not DMA point mass on \(\mu_c\) |

**Downward (national → geo):** Shrinkage and calibration pressure only — not national experiment written as every \(\beta_{g,c}\) point mass.

**Lateral:** No silent borrowing across sibling DMAs except via shared \(\mu_c, \tau_c\).

---

## 7. Uncertainty outputs and TrustReport mapping

TrustReport **owns** uncertainty, confidence, conflicts, sensitivity, and diagnostics. DecisionSurface remains **point-valued** for production default.

### 7.1 TrustReport fields (Bayesian extensions)

| Field / block | Content | Metric class (5D) |
|---------------|---------|-------------------|
| `posterior_summary` | Means, SDs, optional intervals on \(\beta_{g,c}, \mu_c, \tau_c\) | **Diagnostic** |
| `hierarchy_evidence` | From Bayes-H2b validator / fit trace | **Diagnostic / governance** |
| `pooling_diagnostics` | \(\hat{\tau}_c\), shrinkage factors, sparse-geo flags | **Diagnostic** |
| `convergence_diagnostics` | R-hat, ESS, divergences (sandbox) | **Governance blocker** if failed |
| `conflicting_signals` | Per Bayes-H2b | **Governance** |
| `stale_signals` / `missing_se_signals` | Per Bayes-H2 | **Governance** |
| `claim_levels` | Per scope/channel | **Governance** |
| `decision_safe` | Composite readiness | **Decision-grade gate** — not posterior mean |
| `prod_decisioning_allowed` | Default **false** for Bayes sandbox | **Release** |

### 7.2 What TrustReport must not do

- Imply approval because “Bayesian is probabilistic.”
- Hide conflict, stale, or missing-SE signals included in fit.
- Present posterior intervals as **budget recommendations**.

---

## 8. Posterior artifacts (diagnostic only)

| Artifact | Allowed | Prohibited |
|----------|---------|------------|
| Posterior draws | Sandbox export, PPC, coverage studies | Optimizer input |
| Posterior mean \(\hat\beta_{g,c}\) | Adapter input to generative simulate after governance point policy | Direct coef budget tables in prod |
| Credible intervals | TrustReport, research PDFs | DecisionSurface primary output |
| Decomposition / attribution | Diagnostic tier | Prod decide without Δμ path |
| `posterior_summary` JSON block | Sibling to DecisionSurface in artifact bundle | Replacing DecisionSurface |

**Label:** All sandbox posterior artifacts carry `research_only: true` and `prod_decisioning_allowed: false` unless Promotion Gate explicitly passes.

---

## 9. DecisionSurface compatibility

### 9.1 Adapter contract (conceptual)

```text
posterior point policy (mean or MAP) → generative parameter bundle
        ↓
same feature construction as Ridge path
        ↓
simulate(plan), simulate(baseline) on full-panel geometry
        ↓
Δμ = μ(plan) − μ(baseline)  — unchanged estimand semantics
```

### 9.2 Requirements

| Requirement | Spec |
|-------------|------|
| Entry points | Unchanged `mmm decide` / planning simulate |
| Estimand masks | Existing registry IDs only |
| Optimizer | `optimize_budget_via_simulation` on DecisionSurface only |
| Uncertainty in surface | **Out of scope** — TrustReport |
| Certification | VAL-004 / VAL-005 on **Δμ**, not on coef recovery alone |

### 9.3 Point policy (open for H5 promotion review)

Default sandbox policy: **posterior mean** generative parameters. Draw-based optimization is **research-only** unless future ADR + Promotion Gate approves.

---

## 10. Release-gate implications

| Gate / field | Bayesian sandbox default | Promotion path |
|--------------|-------------------------|----------------|
| `approved_for_prod` | **false** | Bayes-H5 + Promotion Gate |
| `prod_decisioning_allowed` | **false** in TrustReport | Explicit ADR after H4 |
| `require_promoted_model_for_prod_decision` | Unchanged | Same fingerprint rules as Ridge |
| `PolicyError` | No `BayesianBypassGate` | — |
| `release_gate_recommendation` | `conditional_not_approved` or `warn` | Per [trust_report_semantics](trust_report_semantics.md) |

Release gates evaluate **TrustReport + artifact fingerprints**, not “convergence alone.”

---

## 11. Validation obligations before implementation

Bayes-H3 sandbox code may start **only after** this ADR is **Accepted** and:

| # | Obligation | Status |
|---|------------|--------|
| 1 | Bayes-H1, H2, H2b ADRs accepted | ✅ |
| 2 | Seven `WORLD-BAYES-*` fixtures materialized | ✅ |
| 3 | `hierarchy_evidence_validator` + `VAL-BAYES-H2B-SMOKE` pass | ✅ |
| 4 | This Bayes-H2d ADR accepted | ✅ (this document) |
| 5 | Bayes-H4 hierarchical **recovery** worlds specified (coef + Δμ) | Pending — before prod candidacy |
| 6 | Convergence / PPC acceptance criteria in validation registry | Pending |
| 7 | Negative tests: no fit without CalibrationSignal | Required in H3 PR |
| 8 | Negative tests: optimizer never reads posterior draws | Required in H3 PR |

**Implementation gate exit (Bayes-H3 sandbox):** Deterministic tests; explicit failure reasons; no forbidden APIs; sandbox label on all artifacts.

---

## 12. Research sandbox allowance

Upon **acceptance of this ADR**:

| Allowed in research sandbox | Condition |
|---------------------------|-----------|
| PyMC (or Stan) prototype fit | `RESEARCH ONLY — NOT DECISION GRADE` label |
| Posterior draws, coef tables, decomposition | Diagnostic exports only |
| Alternative priors, pooling modes | Documented in sandbox report |
| Comparison to Ridge on same worlds | Scorecard — not prod replacement |
| Prototype optimizers on posterior | **Only** if isolated from prod paths and labeled non-decision-grade |

| Still prohibited in sandbox | |
|-----------------------------|---|
| Writing prod `approved_for_prod: true` without Promotion Gate | |
| Shipping alternate `decide` API | |
| Experiment-specific Bayesian APIs bypassing CalibrationSignal | |

---

## 13. Production promotion requirements

Production promotion (Bayes-H3 **production** tier or Bayes-H5 candidacy) requires **all** of:

1. Accepted architecture through Bayes-H4 reliability certification worlds  
2. Contract mapping to binding ABI unchanged  
3. Estimand compatibility demonstrated on hierarchical worlds (VAL-004 Δμ, not coef-only)  
4. CalibrationSignal compatibility with negative ingress tests  
5. TrustReport semantics complete (convergence, conflicts, freshness, missing SE)  
6. Release Gate semantics unchanged; Bayesian stratum additive to scorecard  
7. Reproducible DecisionSurface (fingerprint, replay policy)  
8. Clear diagnostic vs decision-grade artifact separation  
9. [Promotion Gate](../ROADMAP_ALIGNMENT_GATE.md#level-3--promotion-gate) exit checklist  
10. Explicit ADR amending `prod_decisioning_allowed` default — **not** automatic after H3 sandbox  

**Bayes-H3 production promotion remains blocked** until the above are satisfied.

---

## 14. Anti-patterns

| Anti-pattern | Why blocked | Blocked by |
|--------------|-------------|------------|
| `BayesianDecisionSurface` with CI as optimize input | Fragments prod path | Bayes-H1 |
| Optimizer on posterior draws | Violates decision path | Bayes-H1, this ADR §9 |
| Optimizer on \(\hat\beta\) weights | Violates Δμ decisioning | Bayes-H1 |
| `set_beta_from_experiment(lift)` without CalibrationSignal | Bypasses ingress | Bayes-H2 |
| National \(\mu_c\) point mass from single DMA GeoX | Scope lie | Bayes-H2b |
| Silent conflict averaging | Trust fragmentation | Bayes-H2b, worlds CONFLICT |
| Stale evidence at full precision | Freshness lie | worlds STALE |
| Missing SE as decision-grade | Uncertainty lie | worlds MISSING-SE |
| “Bayesian approved” without TrustReport | Gate bypass | Bayes-H1, §10 |
| Coef recovery pass ⇒ prod ready | INV-056 lesson | Bayes-H4 required |
| Skipping hierarchy evidence validator | Unenforceable routing | VAL-BAYES-H2B-SMOKE |
| PyMC in prod CI without research job separation | Drift to prod | Implementation Gate |

---

## 15. Open questions

| ID | Question | Owner phase | Default until resolved |
|----|----------|-------------|------------------------|
| OQ-H2d-01 | Non-centered vs centered parameterization for \(\beta_{g,c}\) | Bayes-H3 sandbox | Non-centered preferred |
| OQ-H2d-02 | Geo-specific vs national adstock/saturation in v1 | Bayes-H3 | National per channel |
| OQ-H2d-03 | Homoscedastic vs geo-specific \(\sigma_g\) | Bayes-H3 | Homoscedastic v1 |
| OQ-H2d-04 | Count / lognormal likelihood family | Bayes-H4 | Gaussian v1 spec only |
| OQ-H2d-05 | Point policy: posterior mean vs MAP for adapter | Bayes-H5 | Posterior mean |
| OQ-H2d-06 | Whether draw-based robust optimization is ever prod-eligible | Bayes-H5+ | Research only |
| OQ-H2d-07 | Nested state/region hyperpriors vs flat \(\mu_c,\tau_c\) | Bayes-H4 worlds | Flat v1 |
| OQ-H2d-08 | CI→SE conversion defaults per source (GeoX vs CLS) | Calibration ETL | Per Bayes-H2 OQ |
| OQ-H2d-09 | Single shared `hierarchy_spec.json` vs per-run graph | Materialization | Shared v1 fixtures |
| OQ-H2d-10 | Artifact schema version for `posterior_summary` block | Bayes-H3 | Sibling to existing bundle |

---

## Decisions (summary)

| Decision | Statement |
|----------|-----------|
| **D1 — Model scope** | Hierarchical geo-channel MMM with partial pooling; observational likelihood + CalibrationSignal evidence terms |
| **D2 — ABI preservation** | No new production contracts; adapter to existing DecisionSurface |
| **D3 — Evidence** | CalibrationSignal sole ingress; H2/H2b registry mandatory before fit |
| **D4 — Posterior role** | Diagnostic and TrustReport; not optimizer input |
| **D5 — Sandbox authorization** | Bayes-H3 **research** implementation allowed after this ADR |
| **D6 — Production** | **Not authorized** — Promotion Gate + Bayes-H4/H5 |

---

## Consequences

- **Bayes-H3 research sandbox** may begin (PyMC prototype, labeled not decision-grade).  
- **Bayes-H3 production promotion** remains **blocked**.  
- **Bayes-H4** must define recovery worlds for \(\mu_c, \tau_c, \beta_{g,c}\) and Δμ before prod candidacy discussion.  
- [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md) row for Bayes-H2d → **Accepted**; next row → Bayes-H3 sandbox.  
- [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) Bayes-H2d section should reference this ADR as **Accepted**.

---

## Acceptance checklist

| # | Criterion | Met |
|---|-----------|-----|
| AC-1 | Model purpose and non-goals stated | ✅ |
| AC-2 | Hierarchy and parameter groups defined | ✅ |
| AC-3 | Observation model conceptual only (no code) | ✅ |
| AC-4 | CalibrationSignal-only integration | ✅ |
| AC-5 | Pooling semantics align with Bayes-H2b | ✅ |
| AC-6 | TrustReport mapping; posterior diagnostic only | ✅ |
| AC-7 | DecisionSurface compatibility | ✅ |
| AC-8 | Release-gate implications | ✅ |
| AC-9 | Validation obligations before implementation | ✅ |
| AC-10 | Research sandbox allowed; production not authorized | ✅ |
| AC-11 | Anti-patterns and open questions | ✅ |
| AC-12 | Roadmap alignment gate applied | ✅ |

**This ADR does not authorize PyMC implementation, production Bayesian decisioning, or Bayes-H3 production promotion.**
