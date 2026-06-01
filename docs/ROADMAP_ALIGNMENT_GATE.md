# Roadmap Alignment Gate — MIP North Star, Risk-Tiered Phase Gates, and Anti-Drift Controls

**Status:** **Accepted** — governance policy for all future MIP / MMM / GeoX / Bayesian / LLM roadmap work  
**Related:** [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md) (living rows) · [MIP_PLATFORM_AUDIT_TEMPLATE.md](MIP_PLATFORM_AUDIT_TEMPLATE.md) (recurring evaluation) · [platform_roadmap.md](05_validation/platform_roadmap.md) · [decision_vs_research.md](02_concepts/decision_vs_research.md) · [production_readiness.md](04_governance/production_readiness.md)

---

## Purpose

This document defines how we keep roadmap execution aligned with the **Marketing Intelligence Platform (MIP) north star** without slowing down harmless maintenance or research exploration.

The goal is **not** to build isolated MMM, GeoX, Bayesian, or LLM components.

The goal is a **governed Marketing Intelligence Platform** that supports:

- unified experimentation
- MMM calibration
- budget planning and optimization
- trust-aware measurement
- explainable recommendations
- conversational AI orchestration
- experiment-informed decisioning

---

## Core principle

**Research is allowed by default. Production promotion is gated by default.**

The roadmap gate is **not a creativity filter**. It is a **promotion filter**.

| Question | Gate answers |
|----------|----------------|
| Can this idea be explored? | **Yes** — in research sandbox or architecture docs when labeled appropriately |
| Can this affect platform contracts, production decisions, optimizers, TrustReports, LLM recommendations, or release artifacts? | **Only** through the tiered gates and proof chain |

**The closer work gets to production decisioning, the stricter the gate.**

- Low-risk maintenance stays lightweight.
- Contract-affecting work needs formal review.
- Production-facing work needs full governance.

**What this policy does not mean:** “Do not explore new algorithms unless they already fit the current ABI perfectly.” That would block meaningful expansion (stronger Bayesian designs, causal ML, HTE, state-space models, better priors, uncertainty propagation, improved calibration). The gate blocks **unsafe promotion**, not research itself.

**What this policy does mean:** An interesting research artifact must not silently become decision-grade system output — used by an optimizer or LLM, looking authoritative, with no TrustReport / release gate / estimand check.

---

## North star rule

No non-trivial **production-promotion** work is allowed unless it **strengthens the platform** and **preserves the binding ABI**. Research and architecture exploration may probe beyond today’s ABI **inside the sandbox** when labeled **not decision-grade**.

### Binding ABI

| Contract | Role |
|----------|------|
| **DecisionSurface** | Canonical production decision interface |
| **Estimand** | Declared quantity under estimation |
| **CalibrationSignal** | Structured experiment / replay evidence ingress |
| **TrustReport** | Trust rollup for recommendations |
| **Release Gates** | Promotion and prod-readiness semantics |
| **full-panel Δμ decisioning** | Canonical counterfactual on training panel geometry |

### Prohibited in production-facing work only

These prohibitions apply to **production decisioning and promotion**, not to labeled research sandbox work (see [Research Freedom and Promotion Path](#research-freedom-and-promotion-path)).

No **production-facing** work may introduce:

- alternate decision surfaces
- coefficient-to-optimizer paths
- posterior-to-optimizer paths
- experiment-specific model APIs
- Bayesian-only decision paths
- calibration bypasses
- TrustReport bypasses
- release-gate bypasses

---

## Risk-tiered gate policy

Every task must be assigned **one of four risk tiers**.

### Tier 0 — Lightweight maintenance

**Applies to:**

- typo fixes
- README link updates
- formatting cleanup
- doc cross-link updates
- small fixture metadata edits
- package export cleanup
- non-semantic refactors

**Required check:**

- Does not alter platform contracts
- Does not alter decision behavior
- Does not alter evidence semantics
- Does not alter release behavior
- Does not authorize new work accidentally

**Allowed proof:**

- short commit note
- simple diff review
- existing tests still pass

**Blocked if:**

- it changes contract meaning
- it changes expected validation behavior
- it changes TrustReport or Release Gate semantics
- it changes optimizer / model behavior

### Tier 1 — Contract / architecture work

**Applies to:**

- ADRs
- roadmap updates
- validation world specs
- schema contracts
- artifact contracts
- TrustReport semantics
- Release Gate semantics
- CalibrationSignal semantics
- Estimand definitions
- DecisionSurface definitions

**Requires:** **Architecture Gate**

### Tier 2 — Implementation work

**Applies to:**

- validators
- adapters
- fixtures
- contract tests
- CI smoke tests
- artifact readers / writers
- decision service plumbing
- simulation wrappers
- governed LLM tool interfaces

**Requires:** **Implementation Gate**

### Tier 3 — Production / decisioning work

**Applies to:**

- model releases
- optimizer changes
- budget recommendations
- production DecisionSurface generation
- production TrustReport generation
- production release-gate evaluation
- LLM-generated recommendations
- executive-facing decision artifacts
- **production** Bayesian model release / decision integration
- GeoX / MMM **production** decision integration

**Requires:** **Promotion Gate**

**Note:** Tier 3 labels **production** paths. Sandbox PyMC prototypes, posterior studies, and algorithm comparisons are **Tier 1 research** or **research sandbox** — not Tier 3 — when labeled not decision-grade.

---

## Research Freedom and Promotion Path

This policy **must not** prevent meaningful algorithmic or statistical research.

The platform should **actively support** research into:

- stronger Bayesian hierarchical models
- better priors and partial pooling strategies
- causal calibration methods
- GeoX / CLS / A/B evidence integration
- uncertainty propagation
- identifiability diagnostics
- response curve validation
- state-space and dynamic effects
- heterogeneous treatment effects
- causal ML
- robustness and falsification methods
- improved optimization under uncertainty
- LLM-assisted measurement workflows

Research may **violate production constraints inside a sandbox** if it is clearly labeled:

```text
RESEARCH ONLY — NOT DECISION GRADE
```

### Research sandbox outputs (allowed)

Research sandbox outputs **may** include:

- posterior draws
- coefficients
- decomposition
- attribution
- experimental model diagnostics
- alternative estimands
- prototype optimizers
- exploratory priors
- non-production simulations

### Research sandbox outputs (not allowed for production)

Research outputs **may not** be used for **production recommendations** unless promoted through the gates. They must not:

- write production DecisionSurface artifacts
- feed production optimizers
- emit production recommendations
- bypass CalibrationSignal, TrustReport, or Release Gates in prod paths
- silently become production code
- claim production readiness without Promotion Gate evidence

### Promotion from research to production requires

1. Accepted architecture decision  
2. Contract mapping to the binding ABI  
3. Estimand compatibility  
4. CalibrationSignal compatibility if evidence is used  
5. TrustReport semantics  
6. Release Gate semantics  
7. Validation worlds or equivalent tests  
8. Reproducible DecisionSurface compatibility  
9. Clear distinction between diagnostic artifacts and decision-grade artifacts  

**The gate blocks unsafe promotion, not research itself.**

### Activity boundary (quick reference)

| Activity | Gate allows? | Condition |
|----------|:------------:|-----------|
| Explore new Bayesian hierarchy | Yes | Marked research / architecture-only |
| Try better priors | Yes | Not production decisioning |
| Compare model families | Yes | Research scorecard |
| Prototype PyMC model | Yes, eventually | Research sandbox only |
| Study posterior uncertainty | Yes | Diagnostic only |
| Improve statistical correctness | Yes | Strongly encouraged |
| Add new algorithm to **production** optimizer | Gate it | Must preserve DecisionSurface |
| Let posterior draws drive budget allocation | **Block** (prod) | Violates decision path |
| Let coefficients drive optimizer | **Block** (prod) | Violates Δμ decisioning |
| Add new calibration ingress without ABI | **Block** (prod) | Violates CalibrationSignal |
| Use research results in LLM recommendations | Gate it | Must be grounded and labeled |

The policy **encourages algorithmic expansion** and requires a **clean bridge** before production.

### Terminology: “blocked” in research contexts

Prefer precise wording:

| Avoid (ambiguous) | Prefer (precise) |
|-------------------|------------------|
| “Bayes-H3 is blocked” | “Bayes-H3 **production** implementation is blocked; sandbox exploration allowed if labeled not decision-grade” |
| “PyMC blocked” | “PyMC **for production decisioning** blocked; research sandbox PyMC allowed after architecture” |
| “Model work blocked” | “**Promotion** to prod blocked; research and architecture spec allowed” |

Vague “blocked” can discourage useful thinking. Use **production promotion blocked** vs **research allowed**.

---

## Roadmap alignment gate

Every **Tier 1, Tier 2, or Tier 3** roadmap item must answer the following questions.

### 1. Platform goal advanced

Which MIP goal does this advance?

**Allowed goals:**

- unified experimentation platform
- MMM calibration ecosystem
- budget planning and optimization
- trust-aware measurement
- explainable recommendation engine
- conversational AI interface
- experiment-informed decisioning platform
- reliability / governance / release safety

If the work does not clearly advance one of these, **production promotion** is **blocked** or the item is **re-scoped** — unless it is explicitly **research-only** (see [Research Freedom and Promotion Path](#research-freedom-and-promotion-path)).

### 2. Binding contracts preserved

Which platform contracts does this touch?

| Check | Required |
|-------|----------|
| DecisionSurface preserved? | ✓ |
| Estimand preserved? | ✓ |
| CalibrationSignal preserved? | ✓ |
| TrustReport preserved? | ✓ |
| Release Gates preserved? | ✓ |
| full-panel Δμ preserved? | ✓ |

If any contract is weakened, bypassed, duplicated, or redefined, the work is **blocked**.

### 3. Explicit non-goals

The proposal must state what it **does not** authorize.

**Examples (scope to tier):**

- does not authorize **production** model release (research sandbox may still be allowed)
- does not authorize PyMC **for production decisioning** (sandbox PyMC may be allowed later)
- does not authorize posterior **as DecisionSurface**
- does not authorize **production** optimizer changes
- does not authorize coefficient-based **production** planning
- does not authorize new **production** experiment-specific APIs
- does not authorize production release

If non-goals are missing for Tier 1–3 work, the work is **incomplete**.

### 4. Failure mode reduced

The proposal must reduce at least one **known failure mode**.

**Known failure modes:**

- wrong estimand
- stale evidence
- missing uncertainty
- conflicting evidence silently averaged
- local evidence treated as national truth
- national evidence overriding local truth
- coefficient recovery mistaken for decision recovery
- posterior uncertainty treated as decision surface
- curves used as optimizer truth
- attribution artifacts used for budget decisions
- unsupported LLM recommendations
- release gate bypass
- calibration bypass
- TrustReport bypass

If the work does not reduce a known failure mode, it is probably **sideways** and must be challenged.

### 5. Validation proof

The proposal must name the proof artifact appropriate to its tier.

**Allowed proof artifacts:**

- ADR section
- validation world
- fixture
- validator
- contract test
- smoke test
- TrustReport field
- release gate
- scorecard
- decision trace
- sandbox report

**Proof standard by tier:**

| Tier | Proof |
|------|--------|
| 0 | simple diff / existing tests |
| 1 | ADR, spec, validation-world plan, acceptance criteria |
| 2 | fixtures, tests, validator output, negative cases |
| 3 | TrustReport, Release Gate, reproducible DecisionSurface, decision trace |

### 6. Drift risk

Each Tier 1–3 item must declare drift risk:

| Level | Meaning |
|-------|---------|
| **Low** | contract / fixture / test / governance only |
| **Medium** | implementation that touches platform contracts |
| **High** | model, optimizer, Bayesian, LLM, or production decision path work |

High-risk work requires **accepted architecture** and **validation proof** before implementation.

### 7. Next authorized step

Each item must state **exactly** what it authorizes next.

**Allowed:**

- authorizes docs update only
- authorizes fixture implementation only
- authorizes validator stub only
- authorizes research sandbox only
- authorizes architecture-only model spec only
- authorizes production release only after gates pass

**Not allowed:**

- vague “continue implementation”
- vague “build model”
- vague “productionize”
- vague “improve optimizer”

---

## Three levels of gates

### Level 1 — Architecture Gate

**Purpose:** Prevent concept drift before work begins.

**Applies to:**

- Tier 1 work
- model specs
- roadmap phase changes
- new platform contracts
- new evidence types
- new decision paths
- new LLM orchestration flows
- any Tier 2 or Tier 3 work without prior accepted architecture

**Entry requirements:**

- platform goal identified
- binding ABI impact declared
- non-goals stated
- failure mode reduced
- validation proof planned
- prohibited behaviors listed
- next authorized step defined

**Exit requirements:**

- ADR or design doc **accepted**
- scope boundaries clear
- validation worlds or tests specified where needed
- TrustReport implications defined where applicable
- Release Gate implications defined where applicable
- no unauthorized implementation approved

**Blocking conditions:**

- introduces alternate decision surface
- bypasses CalibrationSignal
- bypasses TrustReport
- bypasses Release Gates
- uses posterior or coefficients for production decisioning
- lacks validation plan for contract-affecting work
- lacks explicit non-goals

### Level 2 — Implementation Gate

**Purpose:** Prevent code from violating architecture.

**Applies to:**

- Tier 2 work
- validators
- fixtures
- adapters
- contract tests
- decision service plumbing
- simulation wrappers
- governed LLM tools

**Entry requirements:**

- accepted architecture doc, ADR, or contract
- accepted fixture or schema contract where applicable
- test plan exists
- expected pass / fail behavior defined
- no unauthorized dependency required

**Exit requirements:**

- deterministic tests pass
- negative tests pass where applicable
- failure reasons are explicit
- outputs are traceable
- no forbidden APIs added
- no unauthorized model behavior added
- no decision-surface bypass added
- no hidden heuristics added

**Blocking conditions:**

- code creates a second decision path
- code infers estimands silently
- code silently averages conflicting evidence
- code promotes missing-SE evidence to decision-grade
- code treats stale evidence as fresh
- code uses posterior artifacts for decisions
- code allows optimizer to consume coefficients directly
- code lacks tests for failure behavior

### Level 3 — Promotion Gate

**Purpose:** Prevent research or internal artifacts from becoming production decisioning without governance.

**Applies to:**

- Tier 3 work
- production recommendations
- budget plans
- optimizer outputs
- model releases
- experiment-informed decisions
- LLM-generated recommendations
- executive-facing decision artifacts

**Entry requirements:**

- DecisionSurface produced through full-panel Δμ simulation
- Estimand declared and compatible
- CalibrationSignal lineage available where evidence is used
- TrustReport generated
- Release Gates evaluated
- uncertainty and conflicts exposed
- freshness status exposed
- unsupported claims listed

**Exit requirements:**

- release gates pass or explicitly degrade confidence
- TrustReport is decision-attached
- recommendation has traceable evidence
- decision surface is reproducible
- decision does not depend on coefficients, curves, or posterior draws
- user-facing explanation states confidence and limitations
- blocked / degraded decisions are not presented as clean recommendations

**Blocking conditions:**

- no TrustReport
- no release-gate result
- unclear estimand
- unsupported evidence scope
- conflicting evidence hidden
- stale evidence hidden
- missing uncertainty hidden
- coefficient / curve / posterior used as production truth
- LLM recommendation not grounded in governed artifacts

---

## Block vs warning rules

Not all issues block all work.

### Blocking issues

- alternate production decision surface
- optimizer consumes coefficients directly
- optimizer consumes posterior draws directly
- missing TrustReport for production decision
- missing Release Gate for production decision
- CalibrationSignal bypass for experiment evidence
- estimand unknown for decision-grade use
- conflicting evidence silently averaged
- research artifact used as production truth

### Warning / confidence-degrading issues

- stale but non-critical diagnostic evidence
- missing SE in diagnostic-only evidence
- sparse local evidence with clear pooling caveat
- incomplete documentation cross-link
- research result without production claim
- low-severity fixture metadata gap

Warnings must be visible in **TrustReport**, **scorecards**, or **validation reports** when decision-relevant.

---

## Roadmap traceability table

Every **Tier 1–3** roadmap item must maintain this table:

| Roadmap Item | Tier | MIP Goal | Contract Touched | Failure Mode Reduced | Proof Artifact | Gate Level | Status | Next Authorized Step |
|---|:---:|---|---|---|---|---|---|---|
| *(fill per item)* | | | | | | | | |

### Example rows

| Roadmap Item | Tier | MIP Goal | Contract Touched | Failure Mode Reduced | Proof Artifact | Gate Level | Status | Next Authorized Step |
|---|:---:|---|---|---|---|---|---|---|
| *(see registry)* | | | | | | | | |

Full rows: [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md).
| Bayes-H3 PyMC (production) | 3 | MMM calibration ecosystem | DecisionSurface, CalibrationSignal, TrustReport | posterior-as-decision | Bayes-H4 worlds + Promotion Gate | Promotion Gate | **Prod promotion blocked** | not until H2d + H4 gates |
| Bayes-H3 PyMC (research sandbox) | 1 | MMM calibration ecosystem | (sandbox) | algorithm exploration | sandbox report, not decision-grade | Architecture + sandbox | **Allowed** when labeled RESEARCH ONLY | after Bayes-H2d spec accepted |

---

## Sideways work detector

Work is **sideways** if it:

- adds modeling sophistication **for production** without improving decision safety (research sandbox excluded)
- adds a new method without a TrustReport implication
- adds a new output without release-gate semantics
- adds a new artifact that cannot be validated
- improves attribution but not decisioning
- optimizes coefficients instead of DecisionSurface
- creates LLM convenience without grounding
- expands scope without reducing a known failure mode
- produces documentation that cannot be enforced
- produces tests that only check existence, not behavior

**If two or more are true, stop and re-scope.**

---

## Phase authorization rules

A phase may **start** only if:

- previous phase has accepted artifact
- next step is explicitly authorized
- validation proof exists or is being created
- forbidden behaviors are listed
- no blocked dependency is being bypassed

A phase may **not** start if:

- it relies on “we will validate later”
- it introduces model behavior before contract behavior
- it introduces production behavior before release gates
- it introduces LLM behavior before artifact grounding
- it introduces optimizer behavior before DecisionSurface compatibility

---

## Current Bayes-H2b application

### Current authorized sequence

1. Bayes-H2b ADR ✅  
2. Bayes-H2b validation worlds ✅  
3. Bayes-H2b validation runner contract ✅  
4. Seven fixture bundles ✅  
5. `hierarchy_evidence_validator` stub ✅  
6. `VAL-BAYES-H2B-SMOKE` ✅  
7. **Bayes-H2d model spec ADR** — architecture only ✅  
8. **Research sandbox implementation** (e.g. PyMC prototype) — labeled not decision-grade (**next**)  
9. **Promotion evaluation** — only after gates 1–8  

This sequence **allows** meaningful model expansion; it grounds expansion in evidence routing, estimand discipline, TrustReport visibility, and release-gate semantics before production.

### Production promotion blocked (research still allowed when labeled)

| Item | Production | Research sandbox |
|------|------------|------------------|
| Bayes-H3 PyMC / samplers | Blocked until H2d + promotion path | Allowed after H2d architecture spec |
| Posterior-driven budget allocation | Blocked | Diagnostic only |
| Bayesian optimizer integration (prod) | Blocked | Prototype optimizer OK if not decision-grade |
| Coefficient-based production planning | Blocked | Coef tables OK in sandbox |
| Production Bayesian release | Blocked | N/A |

---

## Final policy

The preferred platform advancement chain is:

```text
ADR
  → validation world
  → fixture
  → validator
  → CI / smoke test
  → TrustReport
  → Release Gate
  → DecisionSurface
  → recommendation
```

- **Tier 0** maintenance may bypass the full chain when it does not alter semantics.  
- **Research** may bypass the **production** chain inside the research sandbox when explicitly labeled **not decision-grade**.  
- Any **production-facing** work that skips TrustReport, Release Gates, or DecisionSurface governance is **not** platform production work. It is either **research**, **maintenance**, or **drift**.

| Outcome | Policy |
|---------|--------|
| Research | **Allowed** by default |
| Maintenance | **Allowed** (Tier 0) |
| Production promotion | **Gated** by default |
| Unsafe promotion (research → prod without proof) | **Blocked** (= drift) |

**Research is allowed. Maintenance is allowed. Drift is blocked.**
