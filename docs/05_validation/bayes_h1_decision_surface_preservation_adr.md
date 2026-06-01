# Bayes-H1 — DecisionSurface Preservation ADR

**ADR ID:** `bayes_h1_decision_surface_preservation_v1`  
**Status:** **Accepted** (architecture only — does not authorize PyMC or sampling code)  
**Date:** 2026-05-29  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Supersedes:** Informal Bayes-H1 notes in [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) (hierarchy-only scope)  
**Related:** [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) · [trust_report_semantics.md](trust_report_semantics.md) · [reliability_threshold_governance.md](reliability_threshold_governance.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) Decision 10

**Explicit scope:** This ADR binds **platform contract behavior** for all future Bayesian MMM work. It does **not** authorize implementation of priors, likelihoods, samplers, or PyMC models.

---

## Context

The Marketing Intelligence Platform completed a major risk-reduction arc:

- v1.0.0 freeze and contract architecture  
- Reliability Program Phases 5C–5F (recovery, TrustReport semantics, Monte Carlo framework)  
- [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) (hierarchy and calibration *direction*)

The remaining failure mode for Bayesian Geo MMM is **contract fragmentation**: a capable sampler that defines its own decision API, optimization path, or approval workflow. This ADR prevents that outcome before any Bayesian code is written.

---

## Glossary (binding)

| Term | Definition |
|------|------------|
| **DecisionSurface** | The canonical production API for counterfactual simulation and budget optimization on **full-panel** geometry. Point-valued outputs (e.g. Δμ per scenario) derived from declared estimands and fitted generative state. |
| **Estimator** | Algorithm that fits parameters (Ridge BO, future Bayesian hierarchical). Internal to training. |
| **Posterior** | Distribution \(p(\theta \mid \text{data})\) over model parameters. **Research/diagnostic** unless explicitly promoted via gates. |
| **TrustReport** | Readiness, warnings, uncertainty disclosure, convergence — **not** the optimization input. |
| **Recommendation** | Human- or system-facing narrative (“shift budget toward search”). May cite TrustReport; **not** a substitute for `simulate()`. |
| **Release gate** | `approved_for_prod`, promotion, `PolicyError` — same for all estimators. |

---

## Decision 1 — DecisionSurface supremacy

### Decision

**DecisionSurface** remains the **only** production decision object for counterfactuals and optimization. It is defined by:

- **Full-panel Δμ** (and related simulate outputs) on training panel geometry  
- Declared **Estimand** masks and scenario definitions  
- Existing `mmm decide` / planning `simulate` / `optimize_budget_via_simulation` entry points  

This holds for:

- Ridge BO (current production estimator)  
- Bayesian hierarchical Geo MMM (future)  
- Any future estimator (state-space, robust optimization, experimentation adapters)  

**No estimator may define its own decision surface**, alternate Δμ schema, or parallel “decide” API.

### Rationale

- Platform moat is **contract-centric** orchestration, not a single model family.  
- Experimentation, replay, and release governance are wired to DecisionSurface — not to PyMC objects.  
- INV-056 showed **decision recovery ≠ coefficient recovery**; the decision object must stay stable while estimators change.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| `BayesianDecisionSurface` with credible intervals as primary output | Fragments prod path; intervals belong in TrustReport |
| Per-estimator simulate() signatures | Breaks experimentation adapters and certification |
| Geo-posterior maps as budget output | Conflates attribution with decision |

### Consequences

- Bayesian training must emit a **DecisionSurface adapter** compatible with existing planning code.  
- Certification VAL-004 / VAL-005 run on **DecisionSurface** outputs, not raw posteriors.  
- New estimators require ADR amendment to touch DecisionSurface semantics — not silent PR changes.

---

## Decision 2 — Posterior separation

### Decision

**Posterior** objects and posterior summaries are:

| Allowed use | Prohibited use |
|-------------|----------------|
| Diagnostic tables in artifacts | Budget plans |
| TrustReport uncertainty fields | Release gate inputs alone |
| Research exports (`prod_decisioning_allowed: false`) | Customer-facing “recommended allocation” without simulate |
| Coverage / PPC / convergence checks | Skipping TrustReport because “Bayesian is probabilistic” |

Posteriors are **not** recommendations, **not** budget plans, and **not** implicit approval to optimize.

### Rationale

- Users confuse “posterior mean contribution” with **incrementality**; platform separates **estimation** from **decision**.  
- Release gates require deterministic, auditable readiness — not draw-dependent approval.

### Consequences

- Artifact schema may add `posterior_summary` blocks **sibling** to DecisionSurface — never replacing it.  
- TrustReport must surface posterior quality; DecisionSurface remains point-valued for prod (default).

---

## Decision 3 — Optimization contract

### Decision

**Optimizer input:** **DecisionSurface only** (via existing simulation optimizer path).

The optimizer must **never** consume directly:

- Raw **coefficients** \(\beta_{g,c}\) as allocation weights  
- **Posterior draws** or MCMC samples  
- **Posterior channel contributions** or attribution decompositions  
- **Bayesian-only** objective files or alternate optimizers wired only to PyMC  

Point generative state used inside DecisionSurface may be derived from posterior **mean** (or governance-approved point policy), but optimization calls the **same** `simulate` stack as Ridge.

### Rationale

- VAL-005 and optimizer certification assume simulation-on-panel semantics.  
- Draw-based optimization is a research policy requiring separate Bayes-H5 ADR if ever allowed.

### Consequences

- `optimize_budget_via_simulation` unchanged at API level.  
- Research notebooks may explore draw-based optimization **outside** prod gates — not in `run_environment: prod`.

---

## Decision 4 — TrustReport role

### Decision

**Posterior uncertainty** and Bayesian diagnostics belong in **TrustReport** (and research extensions), including but not limited to:

- Posterior interval width on \(\beta_{g,c}\) or \(\Delta\mu\) (diagnostic)  
- Convergence quality (\(\hat{R}\), ESS, divergences)  
- Pooling strength / shrinkage (\(\tau_c\), geo flags)  
- Prior sensitivity and signal conflict flags  

**DecisionSurface remains separate** — point counterfactuals for operational decide paths.

TrustReport continues to drive:

- `decision_safe` / readiness (with Phase 5D metric classes)  
- Trust modifiers (drift, identifiability, freshness)  
- Attribution-unsafe labeling when diagnostic metrics fail  

### Rationale

- Phase 5D–5E separated decision-grade vs diagnostic metrics; Bayesian must not collapse them.  
- Operators need one readiness object — not TrustReport for Ridge and a second for Bayes.

### Consequences

- Bayes-H3 artifacts must populate **existing** TrustReport fields where possible; new fields require schema ADR.  
- Default: `prod_decisioning_allowed: false` until Bayes-H4/H5.

---

## Decision 5 — Release-gate compatibility

### Decision

Bayesian paths must use the **same** platform contracts as Ridge:

| Contract | Requirement |
|----------|-------------|
| **Release gates** | `approved_for_prod`, promotion workflow, `PolicyError` — identical semantics |
| **TrustReport** | Required before prod decide; no `BayesianApproved` flag |
| **CalibrationSignal** | Only path for experiment/replay evidence into fitting |
| **Estimand** | Registry IDs unchanged; Bayesian maps evidence to priors/likelihoods internally |

**No Bayesian-specific approval path** (no `bayesian_certified_for_prod`, no bypass of promotion fingerprint rules).

### Rationale

- DR-06: ReliabilityScorecard is advisory/tiered — not a second gate for Bayes only.  
- Operators cannot learn two release workflows.

### Consequences

- Bayes-H5 production candidacy is a **review** against existing gates — not a new gate type.  
- ReliabilityScorecard may add `framework:bayesian_hierarchical` stratum — same scorecard schema.

---

## Decision 6 — Explicit anti-patterns

The following patterns are **prohibited** in production paths and **discouraged** in research without explicit ADR:

| Anti-pattern | Why prohibited |
|--------------|----------------|
| **Optimize every posterior draw** then pick best | Non-auditable; prod optimizer contract violation |
| **Average allocations** across posterior draws | Hidden risk; not DecisionSurface |
| **Posterior mean contribution as budget recommendation** | Attribution ≠ decision (INV-056) |
| **Bypass `simulate()`** for Δμ | Breaks Estimand and full-panel geometry |
| **Bypass DecisionSurface** for decide | Contract fragmentation |
| **Bayesian-only CLI** that writes allocations to prod | Side door around gates |
| **Coef recovery as prod gate** for Bayesian | Diagnostic only per 5D governance |
| **Attribution-first “optimal” channel mix from posterior** | Marketing narrative without counterfactual path |

Code review and certification must reject PRs exhibiting these patterns.

---

## Validation requirements (future implementation checklist)

Before any Bayesian estimator merges to `run_environment: prod` (Bayes-H5 bar), certify:

| # | Check | Evidence |
|---|-------|----------|
| V1 | Training artifact exposes **DecisionSurface-compatible** point state | Integration test: `simulate` Δμ on WORLD-008 / Bayes-H2 worlds |
| V2 | **TrustReport** populated (convergence, pooling, uncertainty) | Artifact schema test + CERT-4A-012 |
| V3 | **CalibrationSignal** path only for experiment evidence | No direct coef injection; negative test |
| V4 | **Release gate** behavior identical to Ridge | Promotion + `PolicyError` fixtures |
| V5 | **No new platform contract** JSON types for decide | Schema diff review |
| V6 | Optimizer uses **simulation path only** | VAL-005 on Bayesian point state |
| V7 | Posterior blobs marked **research** or diagnostic tier | `prod_decisioning_allowed: false` default |

This checklist is **architecture acceptance** for Bayes-H3+; execution is Bayes-H4 reliability.

---

## Acceptance criteria (contributor FAQ)

| Question | Answer |
|----------|--------|
| **What is optimized?** | Budget allocation (or scenario parameters) via **DecisionSurface** simulation — same as Ridge. |
| **What is estimated?** | Hierarchical model parameters \(\beta_{g,c}, \mu_c, \tau_c, \ldots\) — internal to the **estimator**. |
| **What is uncertainty?** | Posterior distribution — surfaced in **TrustReport**, not the primary decide output. |
| **What is a recommendation?** | Optional narrative / research export — **not** a gate input; must not replace simulate. |
| **What is a decision object?** | **DecisionSurface** — full-panel Δμ and optimization inputs derived from point generative state. |

---

## Supersession and amendments

| Change type | Process |
|-------------|---------|
| Clarify adapter wording | Patch this ADR (minor) |
| Allow draw-based prod optimization | New ADR + Bayes-H5 executive approval |
| New estimand or DecisionSurface field | Platform contract ADR (DR-07 family) — not Bayes-only |
| New TrustReport posterior fields | Schema ADR + Bayes-H3 doc |

---

## Recommended next phase

**Bayes-H2** is complete: [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md).

**Next:** Materialize `WORLD-BAYES-*` validation worlds (Bayes-H2b exit). **Explicitly not next:** PyMC, priors, likelihood code, samplers.

---

## References

- [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md)  
- [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md)  
- [monte_carlo_reliability_program.md](monte_carlo_reliability_program.md)  
- [platform_roadmap.md](platform_roadmap.md) — Track 4
