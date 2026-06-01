# Bayes-H3 — Research Sandbox Inference Backend

**ADR ID:** `bayes_h3_research_sandbox_backend_v1`  
**Title:** Bayes-H3 — PyMC as initial research sandbox backend (NumPyro deferred)  
**Status:** **Accepted** (research sandbox backend only — does **not** authorize production Bayesian decisioning)  
**Date:** 2026-06-01  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisites:** [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md) · Bayes-H3 sandbox guardrails (P0) ✅ · Bayes-H3 sandbox MVP fit ✅ · [MIP_MINI_AUDIT_20260601_BAYES_H3_SANDBOX_MVP.md](../audits/MIP_MINI_AUDIT_20260601_BAYES_H3_SANDBOX_MVP.md)  
**Governance:** [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md) · [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md)  
**Implementation:** `mmm.research.bayes_h3_sandbox.model` · `mmm.research.bayes_h3_sandbox.entrypoint.run_sandbox_fit`  
**Related:** [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) · [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md) · `mmm.models.bayesian.pymc_trainer` (legacy, sandbox-fenced)

---

## Roadmap alignment gate

| Gate question | Answer |
|---------------|--------|
| **Tier** | 1 — Contract / architecture (sandbox implementation detail) |
| **MIP goal** | MMM calibration ecosystem — auditable research diagnostics |
| **Contracts touched** | TrustReport (diagnostic stub only); Release Gates (no Bayes prod pass) |
| **Failure mode reduced** | Backend churn before recovery proof; opaque sampling stack in research |
| **Proof artifact** | This ADR; MVP fit via `run_sandbox_fit` |
| **Gate level** | Architecture Gate (sandbox) |
| **Research allowed?** | **Yes** — PyMC in `mmm.research.bayes_h3_sandbox` only |
| **Production promotion** | **Blocked** — backend choice ≠ prod authorization |

---

## Decision

**Bayes-H3 uses PyMC as the initial research sandbox inference backend.**

| Backend | H3 role | Notes |
|---------|---------|--------|
| **PyMC** | **Initial / binding for H3 sandbox MVP** | Default for `fit_h3_sandbox_hierarchical` and sanctioned `run_sandbox_fit` |
| **NumPyro** | **Reserved — not H3** | Candidate for later scale/performance phases **after** H3/H4 establish correctness and recovery |
| **Stan (`cmdstanpy`)** | **Unchanged** | Existing stub per INV-049; out of H3 scope |

Optional extra `[bayesian]` install: `pymc`, `arviz`. Sampling tests are `@pytest.mark.slow` and optional in CI fast jobs.

---

## Rationale

1. **Transparent, auditable research diagnostics** — PyMC exposes model graphs, deterministic variables, and ArviZ-backed convergence summaries suitable for **research-only** TrustReport-shaped blocks under strict governance (labels, negative tests, sandbox entrypoint).

2. **Alignment with existing repo code** — `mmm.models.bayesian.pymc_trainer` and Bayes-H3 `model.py` already use PyMC; fencing keeps legacy trainer **sandbox-only** while H3 fit flows through `mmm.research.bayes_h3_sandbox`.

3. **Alignment with PyMC-Marketing MMM ecosystem** — Conceptual parity with industry hierarchical MMM practice without importing PyMC-Marketing as a production dependency in v1.

4. **NumPyro deferred deliberately** — JAX/NumPyro may help large geo × channel panels later, but introduces a second stack before **Bayes-H4 recovery worlds** prove the **H2d generative spec** and sandbox fences. Performance optimization must not precede correctness evidence.

---

## Explicit non-goals

| Non-goal | Reason |
|----------|--------|
| Production Bayesian decisioning | [Bayes-H1](bayes_h1_decision_surface_preservation_adr.md) |
| NumPyro in H3 MVP | Defer until post-H4 performance phase |
| Backend choice as release approval | Promotion Gate unchanged |
| Posterior / coefficient → optimizer | Sandbox guardrails + H1 |
| Replacing Ridge prod estimator | Bayes-H3 production row **Blocked** |

---

## Consequences

### Implementation

- **Sanctioned path:** `mmm.research.bayes_h3_sandbox.entrypoint.run_sandbox_fit` → `fit_h3_sandbox_hierarchical` (PyMC).
- **Legacy path:** `mmm.models.bayesian.pymc_trainer.BayesianMMMTrainer` remains research-fenced; not the H3 product entrypoint.
- **Future NumPyro:** Requires a new ADR amendment or Bayes-H5+ backend ADR after H4 recovery criteria; must preserve research-only labels and diagnostic-only outputs.

### Validation

- **H3 (complete for MVP):** sandbox runs, diagnostic artifact shape, guardrails.
- **H4 (next):** recovery on controlled worlds — backend-agnostic success criteria where possible (compare posterior summaries to truth, not PyMC-specific APIs).

### CI

- Fast CI: artifact wrapping and guardrails without requiring PyMC.
- Slow / optional jobs: `mmm[bayesian]` sampling tests.

---

## Backend choice is not production authorization

All Bayesian outputs remain:

- `label: RESEARCH ONLY — NOT DECISION GRADE`
- `approved_for_prod: false`
- `prod_decisioning_allowed: false`
- `outputs_are_diagnostic_only: true`

until **Promotion Gate**, **Bayes-H4+ recovery**, TrustReport production mapping, and explicit ADR amending prod defaults.

**This ADR does not authorize production Bayesian MMM, DecisionSurface emission from Bayes, or optimizer integration.**
