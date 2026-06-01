# MIP Platform Audit Template — Architecture, Statistical Grounding, Research Coverage, and Execution Health

**Status:** **Accepted** as recurring audit template  
**Related:** [ROADMAP_ALIGNMENT_GATE.md](ROADMAP_ALIGNMENT_GATE.md) (policy) · [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md) (status) · [05_validation/platform_roadmap.md](05_validation/platform_roadmap.md)

---

## Purpose

This template is used to **periodically audit** whether the Marketing Intelligence Platform roadmap, implemented artifacts, research work, and production capabilities remain aligned with the north star:

> Build a **best-in-class Marketing Intelligence Platform** for experiment-informed, trust-aware, statistically grounded, explainable marketing decisioning.

The audit should identify:

- what has been implemented
- what is aligned with platform goals
- what is missing
- what is drifting
- what is statistically or conceptually weak
- what is not yet production safe
- what research should be prioritized
- what architectural gaps must be closed
- what recommendations should be executed next

### How this fits the governance stack

| Layer | Role |
|-------|------|
| [ROADMAP_ALIGNMENT_GATE.md](ROADMAP_ALIGNMENT_GATE.md) | **Policy** — what is allowed |
| [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md) | **Status** — where things stand |
| **This template** | **Recurring evaluation** — whether the platform is becoming best-in-class or merely accumulating artifacts |

**Research is not punished for being non-production.** The audit punishes research only when it is **unlabeled**, **untraceable**, or **silently promoted**.

---

## Audit levels (when to run what)

Do **not** run the full template after every small commit. Use three levels:

| Level | When to run | Scope |
|-------|-------------|--------|
| **Mini audit** | After small implementation milestones (validator stub, fixture batch, doc cross-links) | Sections **1**, **4**, **8** (summary, registry check, validation spot-check) + **17** verdict |
| **Phase audit** | Before starting a major new phase (e.g. Bayes-H2d model spec, Bayes-H3 sandbox, new evidence source, new optimizer behavior) | Full template; emphasize **2**, **3**, **6**, **14**, **16** |
| **Promotion audit** | Before anything affects **production** decisions, optimizers, LLM recommendations, or release artifacts | Full template; emphasize **3**, **9**, **10**, **11**, **15** P0; Promotion Gate from alignment gate |

### Suggested cadence (full or phase audit)

Run a **phase** or **promotion** audit:

- after every major phase
- before starting a new model family
- before production promotion
- after major roadmap changes
- after introducing new evidence sources
- after introducing new optimizer or LLM decisioning behavior
- at least once per milestone cycle

**Current note (2026-05-29):** Bayes-H2d ADR is accepted. Run a **phase audit** before **Bayes-H3 research sandbox** implementation — first move from evidence-governance scaffolding into executable model exploration.

---

## Audit inputs

Gather before auditing:

| Input | Location |
|-------|----------|
| Alignment gate | [ROADMAP_ALIGNMENT_GATE.md](ROADMAP_ALIGNMENT_GATE.md) |
| Alignment registry | [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md) |
| Platform roadmap | [05_validation/platform_roadmap.md](05_validation/platform_roadmap.md) |
| Current ADRs | `docs/05_validation/bayes_h*.md`, `synthetic_architecture_decisions.md` |
| Validation worlds | `validation/worlds/`, `docs/BAYES_H2B_VALIDATION_WORLDS_001.md` |
| Fixture bundles | `validation/worlds/WORLD-*` |
| Validator reports | e.g. `hierarchy_evidence_validator`, `VAL-BAYES-H2B-SMOKE` |
| TrustReport specs | [05_validation/trust_report_semantics.md](05_validation/trust_report_semantics.md) |
| Release gate specs | [04_governance/production_readiness.md](04_governance/production_readiness.md), [prod_safety_checklist.md](04_governance/prod_safety_checklist.md) |
| Model specs | [bayes_h2d_hierarchical_model_spec_adr.md](05_validation/bayes_h2d_hierarchical_model_spec_adr.md) |
| Experiment evidence | [02_concepts/experiment_evidence.md](02_concepts/experiment_evidence.md) |
| Open investigations | [06_investigations/open_investigations.md](06_investigations/open_investigations.md) |
| Research notes | ADRs, refinement docs |
| Implementation tests | `tests/` |
| CI / smoke reports | pytest, smoke JSON |

**Audit output:** Save completed audits as `docs/audits/MIP_PLATFORM_AUDIT_YYYYMMDD.md` (optional; not required for every mini audit).

**Completed audits:**

| Audit | Verdict |
|-------|---------|
| [MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md](audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md) | **Yellow** — Bayes-H3 research sandbox authorized with P0 fences |

---

## 1. Executive summary

| Field | Value |
|-------|--------|
| **Audit date** | |
| **Audit level** | Mini / Phase / Promotion |
| **Audited scope** | |
| **Current platform phase** | |
| **Overall health** | Green / Yellow / Red |
| **Main conclusion** | |
| **Top 3 strengths** | 1. · 2. · 3. |
| **Top 3 risks** | 1. · 2. · 3. |
| **Top 3 recommended actions** | 1. · 2. · 3. |

---

## 2. North star alignment

**Question:** Is current work still advancing the **Marketing Intelligence Platform**, rather than becoming an isolated MMM, GeoX, Bayesian, or LLM project?

**Evaluate against:**

- unified experimentation
- MMM calibration ecosystem
- budget planning and optimization
- trust-aware measurement
- explainable recommendation engine
- conversational AI orchestration
- experiment-informed decisioning

| Platform goal | Current support | Evidence | Gap | Recommendation |
|---------------|-----------------|----------|-----|----------------|
| Unified experimentation | | | | |
| MMM calibration ecosystem | | | | |
| Budget planning and optimization | | | | |
| Trust-aware measurement | | | | |
| Explainable recommendation engine | | | | |
| Conversational AI orchestration | | | | |
| Experiment-informed decisioning | | | | |

**Required conclusion:** Aligned / Partially aligned / Drifting / Not aligned

**Rationale:**

---

## 3. Binding ABI preservation

Check whether current work preserves:

- DecisionSurface
- Estimand
- CalibrationSignal
- TrustReport
- Release Gates
- full-panel Δμ decisioning

| Contract | Preserved? | Evidence | Violation risk | Required action |
|----------|:----------:|----------|----------------|-----------------|
| DecisionSurface | | | | |
| Estimand | | | | |
| CalibrationSignal | | | | |
| TrustReport | | | | |
| Release Gates | | | | |
| full-panel Δμ | | | | |

### Blocking findings (check any present)

- [ ] alternate decision surface
- [ ] posterior-to-optimizer path
- [ ] coefficient-to-optimizer path
- [ ] experiment-specific API (bypassing CalibrationSignal)
- [ ] calibration bypass
- [ ] TrustReport bypass
- [ ] release-gate bypass

---

## 4. Roadmap alignment registry check

Review [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md).

| Question | Answer |
|----------|--------|
| Is every current roadmap item registered? | |
| Is every item assigned a tier? | |
| Is every item assigned a gate? | |
| Is the next authorized step explicit? | |
| Are research and production-promotion statuses separated? | |
| Are completed items stale or incorrectly marked? | |
| Are blocked items being bypassed? | |

| Roadmap item | Registry status | Actual status | Mismatch? | Action |
|--------------|-----------------|---------------|:---------:|--------|
| | | | | |

---

## 5. Architecture soundness audit

**Dimensions:** contracts before models · artifacts before orchestration · validators before promotion · clear ownership of decisions / uncertainty / evidence · reproducibility · traceability · fail-closed · no hidden heuristics

| Architecture dimension | Rating | Evidence | Gap | Recommendation |
|------------------------|--------|----------|-----|----------------|
| Contracts before models | | | | |
| Artifacts before orchestration | | | | |
| Validators before promotion | | | | |
| Decision ownership | | | | |
| Uncertainty ownership (TrustReport) | | | | |
| Evidence ownership (CalibrationSignal) | | | | |
| Reproducibility | | | | |
| Traceability | | | | |
| Fail-closed behavior | | | | |
| No hidden heuristics | | | | |

**Ratings:** Strong / Adequate / Weak / Missing

---

## 6. Statistical grounding audit

| Statistical dimension | Current state | Risk | Evidence | Required improvement |
|-----------------------|---------------|------|----------|----------------------|
| Estimand clarity | | | | |
| Identification assumptions | | | | |
| Causal interpretation boundaries | | | | |
| Uncertainty treatment | | | | |
| Variance / SE handling | | | | |
| Calibration validity | | | | |
| Experimental evidence compatibility | | | | |
| Pooling assumptions | | | | |
| Drift handling | | | | |
| Falsification tests | | | | |
| Sensitivity analysis | | | | |
| Robustness checks | | | | |
| Identifiability diagnostics | | | | |
| Extrapolation control | | | | |

**Key questions:**

- Are we measuring the right estimand?
- Are assumptions explicit?
- Are uncertainty claims justified?
- Are causal claims supported?
- Are diagnostics separated from decision-grade outputs?
- Are decision metrics separated from attribution metrics?
- Are failure modes visible in TrustReport?

---

## 7. Research and cutting-edge coverage audit

Ensure the platform does not become stale or overly conservative.

| Research area | Covered? | Current artifact | Gap | Priority | Recommendation |
|---------------|:--------:|------------------|-----|:--------:|----------------|
| Bayesian hierarchical MMM | | | | | |
| Dynamic / state-space MMM | | | | | |
| Causal ML | | | | | |
| Doubly robust calibration | | | | | |
| Heterogeneous treatment effects | | | | | |
| Partial pooling | | | | | |
| Geo-experiment integration | | | | | |
| Incrementality-aware MMM | | | | | |
| Synthetic controls / ASCM / SDID | | | | | |
| Uncertainty-aware optimization | | | | | |
| Decision-theoretic planning | | | | | |
| Response curve validation | | | | | |
| Saturation and carryover robustness | | | | | |
| LLM-assisted analysis and explanation | | | | | |
| Agentic orchestration with governed tools | | | | | |
| Automated literature scouting | | | | | |
| Benchmark vs industry systems | | | | | |

**Priority:** P0 = platform correctness · P1 = best-in-class · P2 = useful extension · P3 = exploratory

---

## 8. Validation and reliability audit

| Validation area | Covered? | Evidence | Missing tests | Recommendation |
|-----------------|:--------:|----------|---------------|----------------|
| Synthetic worlds | | | | |
| Recovery worlds | | | | |
| Replay worlds | | | | |
| Drift worlds | | | | |
| Optimizer worlds | | | | |
| Identifiability worlds | | | | |
| Evidence-routing worlds (`WORLD-BAYES-*`) | | | | |
| Negative fixtures | | | | |
| Smoke tests | | | | |
| Contract tests | | | | |
| CI coverage | | | | |
| Deterministic outputs | | | | |
| Fail-closed behavior | | | | |

**Key checks:**

- [ ] Validating **behavior**, not just file existence
- [ ] Negative cases included
- [ ] Tests prevent silent averaging
- [ ] Tests prevent missing-SE promotion
- [ ] Tests prevent stale evidence misuse
- [ ] Tests prevent posterior decisioning
- [ ] Tests prevent alternate decision surfaces

---

## 9. TrustReport and release gate audit

| Trust / gate dimension | Current state | Gap | Risk | Recommendation |
|------------------------|---------------|-----|------|----------------|
| Included evidence | | | | |
| Excluded evidence | | | | |
| Stale evidence | | | | |
| Missing uncertainty | | | | |
| Conflicting signals | | | | |
| Sensitivity | | | | |
| Drift | | | | |
| Pooling diagnostics | | | | |
| Convergence diagnostics | | | | |
| Identifiability risks | | | | |
| Alignment status | | | | |
| Unsupported claims | | | | |
| decision-ready vs degraded vs blocked vs research-only | | | | |

---

## 10. Decisioning and optimization audit

**Required path:**

```text
Model → simulate() → DecisionSurface → optimizer → recommendation → TrustReport → Release Gates
```

**Never:** coefficients → optimizer · posterior draws → optimizer · curves alone → budget recommendation

| Decisioning component | Current behavior | Contract safe? | Gap | Recommendation |
|-----------------------|------------------|:----------------:|-----|----------------|
| Full-panel Δμ | | | | |
| Optimizer inputs | | | | |
| Recommendation traceability | | | | |
| Uncertainty attached | | | | |
| Diagnostics vs decisions separated | | | | |

---

## 11. LLM / conversational interface readiness audit

| LLM readiness dimension | Current state | Gap | Recommendation |
|-------------------------|---------------|-----|----------------|
| Governed tool interfaces | | | |
| Artifact-grounded answers | | | |
| No unsupported recommendations | | | |
| Explicit confidence language | | | |
| TrustReport-aware responses | | | |
| Release-gate-aware responses | | | |
| Unsupported-question handling | | | |
| Reproducible decision traces | | | |
| No hallucinated evidence | | | |
| No hidden optimizer calls | | | |

---

## 12. Competitive / best-in-class benchmark audit

| Benchmark dimension | Current state | Best-in-class target | Gap | Recommendation |
|---------------------|---------------|---------------------|-----|----------------|
| Causal incrementality integration | | | | |
| Experiment-informed MMM | | | | |
| Trustworthy budget optimization | | | | |
| Transparent uncertainty | | | | |
| Explainable recommendations | | | | |
| Reproducible artifacts | | | | |
| Governance and auditability | | | | |
| Simulation-based planning | | | | |
| Research extensibility | | | | |
| Conversational decision support | | | | |

---

## 13. Missing capabilities inventory

| Missing capability | Category | Severity | Blocks what? | Recommended action |
|--------------------|----------|----------|--------------|-------------------|
| | architecture / validation / statistical / … | Critical–Low | | |

**Categories:** architecture · contracts · validation · statistical methods · causal evidence · optimization · TrustReport · release gates · LLM interface · documentation · tests · CI/CD · research · production hardening

---

## 14. Drift and anti-pattern check

| Anti-pattern | Present? | Evidence | Action |
|--------------|:--------:|----------|--------|
| Model-first development | | | |
| Docs without enforcement | | | |
| Tests without behavioral assertions | | | |
| Research artifact used as production truth | | | |
| Coefficient worship | | | |
| Curve worship | | | |
| Posterior worship | | | |
| Optimizer bypass | | | |
| TrustReport bypass | | | |
| Release-gate bypass | | | |
| LLM answer without artifact grounding | | | |
| Expanding scope without reducing a failure mode | | | |
| Adding algorithms without assumptions or validation | | | |

---

## 15. Recommendations

### Immediate P0 (before next phase)

| Recommendation | Why | Required artifact | Owner / track | Blocks |
|----------------|-----|-------------------|---------------|--------|
| | | | | |

### Near-term P1 (strong platform quality)

| Recommendation | Why | Required artifact | Owner / track | Blocks |
|----------------|-----|-------------------|---------------|--------|
| | | | | |

### Research P2 (best-in-class evolution)

| Recommendation | Why | Required artifact | Owner / track | Blocks |
|----------------|-----|-------------------|---------------|--------|
| | | | | |

---

## 16. Next authorized work

| Category | Items |
|----------|--------|
| **Authorized** | |
| **Blocked** | |
| **Research allowed** | |
| **Production promotion blocked until** | |

**Example (illustrative — verify against registry):**

- **Authorized:** Bayes-H3 research sandbox (PyMC prototype, `RESEARCH ONLY — NOT DECISION GRADE`)
- **Blocked:** production Bayesian model implementation; posterior decisioning; optimizer on posterior/coefs
- **Research allowed:** sandbox exploration per [bayes_h2d_hierarchical_model_spec_adr.md](05_validation/bayes_h2d_hierarchical_model_spec_adr.md)
- **Production promotion blocked until:** Bayes-H4 worlds; Promotion Gate; TrustReport + release gates pass; reproducible DecisionSurface

---

## 17. Audit verdict

| Verdict | Meaning |
|---------|---------|
| **Green** | Proceed |
| **Yellow** | Proceed with required fixes |
| **Red** | Stop and repair foundations |

| Field | Value |
|-------|--------|
| **Verdict** | Green / Yellow / Red |
| **Rationale** | |
| **Required fixes** | |
| **Next audit trigger** | |

---

## Appendix — Quick mini-audit checklist

For **mini audits** only, complete:

1. Executive summary (§1) — health + top 3 actions  
2. Registry spot-check (§4) — any mismatch?  
3. Validation spot-check (§8) — behavior tests still pass?  
4. Verdict (§17)  

If any blocking finding in §3 → escalate to **phase audit**.
