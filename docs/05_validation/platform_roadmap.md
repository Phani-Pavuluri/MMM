# Marketing Intelligence Platform — contract-driven roadmap

**Status:** Active planning document (v1.x → v2 foundation)  
**Supersedes (narrative only):** feature-centric “MMM package roadmap” as the primary planning frame  
**Preserves:** All historical synthetic validation phases — see [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md)

---

## Positioning

The **MMM Python package** (`mmm`) is one **implementation surface** within a broader **Marketing Intelligence Platform**. Long-term stability is defined by **platform contracts** (semantic ABI), not by any single model family or optimizer implementation.

| Layer | Role |
|-------|------|
| **Platform contracts** | DecisionSurface, Estimand, CalibrationSignal, TrustReport, release-gate semantics, full-panel Δμ |
| **MMM implementation** | Ridge BO, materialization, governance modules, `mmm decide`, certifications today |
| **Future surfaces** | Experimentation systems, conversational orchestration, recommendation engines — must **consume** the same contracts |

---

## Roadmap evolution philosophy

### Before

- **Feature-centric MMM roadmap** — new transforms, optimizers, and model families listed as primary progress.
- Validation treated as a test harness adjacent to product features.

### Now

- **Contract-centric, reliability-first platform roadmap** — progress measured by preserved semantics and **proved** behavior under controlled worlds.
- Modeling expansion is **downstream** of reliability infrastructure.

### Strategic question (current)

| Former framing | Current framing |
|----------------|-----------------|
| “Can we build more advanced MMM methods?” | “Can we **prove** advanced methods are **trustworthy** under controlled worlds while **preserving decision semantics**?” |

### Current production moat (explicit)

The defensible moat is **not** sheer model complexity. It is:

| Moat pillar | Meaning |
|-------------|---------|
| **Estimand discipline** | Declared estimands; geo-time ATT and full-panel Δμ semantics |
| **Replay governance** | Evidence registry, prod replay gates, refit policy |
| **Decision semantics** | Single canonical production decision surface |
| **Reliability proving** | GroundTruthWorld → certification → scorecard (in flight) |
| **Contract preservation** | No parallel production truth paths |

---

## Top-level roadmap structure

```text
1. Platform Contract Preservation           ← DecisionSurface, Estimand, release gates (MMM v1)
2. Reliability & Validation Program         ← GroundTruthWorld, worlds, certs, scorecard
3. Core Production Decisioning              ← replay, Δμ, optimizer, release governance
4. Research Sandbox                         ← Bayes-H*, modeling expansions
5. Conversational / Orchestration           ← future; consumes contracts only
```

**Note:** GeoX / panel-experimentation **Track B** (contract identity) and **Track D** (estimator robustness) belong to the **unified MIP / GeoX project**, not this MMM repository. See [ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md](../ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md).

---

## Platform Contract Preservation

All MMM and platform development must preserve compatibility with:

| Contract | Production role |
|----------|-----------------|
| **DecisionSurface** | Canonical interface for production budget/simulation decisions |
| **Estimand** | Declared quantity being estimated (e.g. geo-time ATT, full-panel counterfactual Δμ) |
| **CalibrationSignal** | Structured experiment/replay evidence entering calibration — not ad hoc coef targets |
| **TrustReport** | Rollup of readiness, certifications, warnings — required for production recommendations |
| **Release-gate semantics** | `approved_for_prod`, promotion, fingerprint, PolicyError fail-closed rules |
| **Full-panel Δμ** | Canonical production counterfactual on training panel geometry — not curve extrapolation alone |

**Implementation mapping (v1):** [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), [production_readiness.md](../04_governance/production_readiness.md), [calibration.md](../02_concepts/calibration.md), [artifact_schema.md](../04_governance/artifact_schema.md).

### Preservation rules (binding for prod paths)

| Rule | Rationale |
|------|-----------|
| **No alternate production decision surface** | Prevents forked budget APIs with incompatible semantics |
| **No optimizer-specific estimands** | Optimizer must score the same Estimand as simulate/decide |
| **No calibration path bypassing CalibrationSignal** | Replay/evidence must flow through declared signal contracts |
| **No production recommendation without TrustReport** | Operators and downstream systems need explicit trust rollup |
| **No orchestration layer bypassing release gates** | Agents and copilots cannot skip `approved_for_prod` / promotion |
| **No decomposition or curve surface promoted to decision-grade** | Curves remain diagnostic; Δμ path is canonical |
| **No experimentation integration bypassing Estimand semantics** | Lift units must declare estimand, scale, geo scope, windows |
| **No new production model without contract compatibility review** | ADR + reliability evidence before prod gate changes |

### Roadmap governance rule (modeling)

> **No major modeling capability enters production unless it demonstrates measurable improvement against reliability-program worlds and certifications.**

See [synthetic_validation_roadmap.md §10](synthetic_validation_roadmap.md#governing-rule-release-policy) for execution criteria (VAL registry, Phase 4A runners, Phase 4B ReliabilityScorecard).

### Roadmap gate — modeling expansion (Phase 4B/5B evidence)

> **No major production modeling expansion may proceed until:**
>
> 1. **Exact recovery investigation** (Phase 5C) is complete — [exact_recovery_investigation.md](exact_recovery_investigation.md), [INV-056](../06_investigations/open_investigations.md#inv-056--exact-recovery-failure-analysis-phase-5c)  
> 2. **Reliability threshold governance** (Phase 5D) is complete — [reliability_threshold_governance.md](reliability_threshold_governance.md), DR-04 draft, metric-class scorecard  
> 3. **Drift validation** (Phase 5E) is complete — [drift_detection.md](drift_detection.md), [trust_report_semantics.md](trust_report_semantics.md)  

**Evidence (WORLD-008–012 + behavioral lattice):** structural reliability is high; **behavioral recovery is the limiting factor**; exact-recovery worlds expose coefficient/transform weaknesses; **platform contracts remain stable**. The next objective is **proving and improving recovery under known truth**, not adding model sophistication.

Future **experimentation** and **orchestration** systems must consume **DecisionSurface**, **Estimand**, **CalibrationSignal**, and **TrustReport** — not invent parallel JSON shapes or gate names.

### MMM–MIP handoff reconciliation (2026-07-13)

`MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001` found the export lane **blocked by
ownership mixing**. `MMM_MIP_HANDOFF_V1_PRODUCER_BOUNDARY_CLEANUP_001` removed
the MIP-owned consumer parser, input loading, user-intent/conversational
answerability, refusal wording, and LLM-routing policy from MMM. MMM retains
only producer artifacts and technical evidence: schemas, serialization,
structural validation, diagnostics, calibration lineage, promotion evidence,
allowed/blocked technical claims, artifact availability, and range restrictions.

`MMM_MIP_HANDOFF_V1_TYPED_FAILURE_PACKET_001` implements the MMM-owned,
versioned technical failure packet and producer outcome wrapper. The subsequent
`MMM_MIP_HANDOFF_V1_TYPED_RUN_MANIFEST_001` adds the strict, versioned producer
run manifest and additive export-boundary linkage. Neither task reintroduces MIP
parsing or conversational policy. R9 and R10 are implemented with contract and
boundary tests; R16 MIP consumer readiness remains blocked, MIP ingestion and
recommendation authority remain unauthorized, and the producer interface is not
frozen. The next narrow producer task must be selected from the remaining audit
gaps using current evidence; typed calibration-treatment lineage (R6) is the
current candidate. `MMM_MIP_HANDOFF_V1_CALIBRATION_TREATMENT_LINEAGE_001`
implements that producer evidence contract without changing calibration math,
Ridge behavior, Bayes promotion, or MIP policy. R9 and R10 remain implemented;
R16 remains blocked and interface freeze remains unauthorized. The next task
must be selected from the remaining audited producer gaps based on evidence.
See [the reconciliation audit](MMM_MIP_HANDOFF_V1_RECONCILIATION_AUDIT_001.md).

Typed diagnostics and limitations (R7) are now producer-owned evidence. R9,
R10, and calibration-treatment lineage remain implemented; R16 remains blocked
and interface freeze remains unauthorized. The next task must come from the
remaining audited producer gaps based on current evidence.

The remaining-gap selection audit names
`MMM_MIP_HANDOFF_V1_PRODUCER_GOLDEN_FIXTURES_001` as the next narrow producer
task. R16 remains blocked and interface freeze remains unauthorized.

That fixture suite now closes R13 without authorizing consumer readiness,
recommendations, optimization, simulation, response surfaces, or interface freeze.

Post-R13 audit selection: `MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001`
(R15) precedes safe consumer binding; R11/R12 remain deferred planning evidence gaps.

R15 is now implemented as a producer-owned compatibility/deprecation policy
with a deterministic registry of the supported public contract and fixture-set
versions. It records the observed permissive export-model versus strict typed
contract parser behavior without changing runtime parsing or golden fixtures.
This policy does not negotiate schemas, freeze the producer interface, or
authorize MIP consumer readiness. R6, R7, R9, R10, calibration-treatment
lineage, and R13 remain implemented; R11/R12 remain partial; R16 remains
blocked. The next task must be selected from post-R15 evidence and all
downstream authorization flags remain false.

Post-compatibility evidence selection: R11 public simulation export and R12
response-surface evidence remain partial. Their next shared producer
prerequisite is `MMM_MIP_HANDOFF_V1_SUPPORTED_RANGE_EVIDENCE_001`: a positive,
versioned supported-range record, not a simulation/response-surface export.
R16 remains blocked, interface freeze remains unauthorized, and all downstream
authorization flags remain false.

Supported-range evidence is now implemented as the MMM-owned, versioned
`MMMSupportedRangeEvidence` contract with typed bounds, scope, availability,
extrapolation, restrictions, and additive manifest/export references. It records
only existing observed/training-domain and explicit restriction evidence; it
does not derive new numerical support. R11 and R12 remain partial, R16 remains
blocked, interface freeze remains unauthorized, and all downstream authorization
flags remain false. Based on post-range evidence, the next narrow producer task
is `MMM_MIP_HANDOFF_V1_PUBLIC_SIMULATION_EXPORT_001`; it is not authorized by
this status update.

---

## Track 1 — Platform Contract Layer

**Goal:** Freeze and evolve the semantic ABI that outlives any single model implementation.

| Workstream | Status | Documents |
|------------|--------|-----------|
| GroundTruthWorld / world schema | ✅ Phase 0–1B frozen | [groundtruth_contract.md](groundtruth_contract.md), [world_schema.md](world_schema.md) |
| Decision artifact + planning assumptions | ✅ v1 shipped | [artifact_schema.md](../04_governance/artifact_schema.md), [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md) |
| Validation registry (VAL-001–014) | ✅ frozen; thresholds TBD_v1 | [validation_registry.md](validation_registry.md) |
| Contract vocabulary formalization (DecisionSurface, TrustReport as named types) | Planned | ADR amendment; align code exports over time |
| Cross-surface contract conformance tests | Planned | Phase 4A certification runners |

**Not in scope here:** New model families — those enter via Track 4 after Track 2 evidence.

---

## Track 2 — Reliability & Validation Program

**Goal:** Prove implementation behavior against declared truth at scale. **Reliability proving is the primary moat and validation mechanism for future modeling expansion.**

After Phases **4B** (recovery worlds) and **5B** (behavioral lattice), the program can **locate failures** (structural vs behavioral). Priority shifts from expanding validation coverage to **understanding and improving behavioral recovery** under known truth.

**Detailed phased plan (historical phases preserved):** [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md)

### Track 2 milestones (Phase 5 — post-lattice)

| Phase | Status | Focus |
|-------|--------|--------|
| **5A** | ✅ | Structural lattice sweep — sweep machinery |
| **5B** | ✅ | Behavioral lattice — recovery by axis |
| **5C** | ✅ | [Exact recovery investigation](exact_recovery_investigation_report.md) — INV-056 |
| **5D** | ✅ | [Reliability threshold governance](reliability_threshold_governance.md) — metric classes, DR-04 |
| **5E** | ✅ | Drift detection runner (VAL-012) + TrustReport semantics |
| **5F** | ✅ | [Monte Carlo program](monte_carlo_reliability_program.md) — tier-0 characterization + threshold recommendations |
| **Track 4** | **In progress** | Seven `WORLD-BAYES-*` fixtures ✅ — `hierarchy_evidence_validator` + `VAL-BAYES-H2B-SMOKE` **next** |

### Aligned capabilities

| Capability | Roadmap phase | Doc / code |
|------------|---------------|------------|
| GroundTruthWorld contract | Phase 0 ✅ | [groundtruth_contract.md](groundtruth_contract.md) |
| World bundle materialization | Phase 1A–2B ✅ | [world_materialization.md](world_materialization.md) |
| Bundle validator L1–L3 | Phase 2A ✅ | [world_validator_spec.md](world_validator_spec.md) |
| Replay unit materialization | Phase 2B ✅ | `replay_units.py` |
| Deterministic archetype generators | Phase 3A ✅ | `generators.py` |
| ScenarioBuilder | Phase 3B ✅ | [scenario_builder.md](scenario_builder.md) |
| Validation worlds (smoke) | Ongoing | `validation/worlds/WORLD-001`–`007` |
| Minimal DGP archetype library | Phase 2 (parallel) | §5 synthetic roadmap |
| Certification runners | **Phase 4A (next)** | §7 synthetic roadmap |
| ReliabilityScorecard | Phase 4B | §8 synthetic roadmap |
| Large-scale world sweeps | Phase 5 | §8 scale tiers |
| Replay validation (VAL-006) | Phase 4A+ | [calibration.md](../02_concepts/calibration.md) |
| Optimizer certification | Phase 4A+ | [optimizer_certification.md](../04_governance/optimizer_certification.md) |
| Calibration robustness | Phase 4A+ / VAL-007 | [validation_registry.md](validation_registry.md) |
| Drift worlds | Phase 2 / 3B axes | `drift_truth` in ScenarioBuilder |
| Identifiability / collinearity worlds | Phase 3B+ | WORLD-006 scenario |

### North star (unchanged)

> Validated across **N** synthetic worlds with reported recovery error, replay consistency, regret, and artifact failure rates — **not** causal incrementality claims.

---

## Track 3 — Core Production Decisioning

**Goal:** Evolve production paths **conservatively** — trustworthy decision-making over rapid model expansion.

**Philosophy:** This track hardens what ships in `run_environment: prod`. New capabilities default to Track 4 until reliability evidence exists.

| Focus area | v1 evidence | Hardening / investigations |
|------------|-------------|----------------------------|
| **Replay integrity** | Replay prod gate, fold-aligned refit policy | INV-005, INV-007, INV-011 |
| **Δμ correctness** | Full-panel simulate; synthetic cert | INV-008, Phase 4A |
| **Optimizer reliability** | Optimizer cert (synthetic surfaces) | INV-001 |
| **Calibration reliability** | Weighted replay, freshness | INV-007, INV-041 |
| **Artifact integrity** | Fingerprint v2, decision bundle tier | INV-004, INV-014 |
| **Release governance** | Production readiness, promotion | INV-002, INV-028 |
| **TrustReport correctness** | `production_readiness_report` rollup | Maps to TrustReport semantics |
| **Decision reproducibility** | Repro cert, decision trace | INV-013 |

**Canonical prod stack (v1):** Ridge BO, `semi_log`, geometric adstock, Hill, full-panel Δμ — [v1_release_notes.md](../04_governance/v1_release_notes.md).

---

## Track 4 — Research Sandbox

**Goal:** Isolate experimental modeling and optimization **without** weakening production gates.

**Rule:** Research capabilities **cannot** become production-grade until validated against:

- Reliability-program worlds (Track 2)
- Certification runners (Phase 4A)
- Recovery thresholds (`TBD_v1` → versioned)
- Replay validation (VAL-006)
- TrustReport semantics
- Release-gate compatibility

### Track 4 — Research Sandbox inventory (no prod until Track 2 gate)

All items below are **Track 4 only** until Phase **5C–5E** complete. They must not introduce a new production decision surface.

| Capability | Track 4 notes |
|------------|----------------|
| **Bayesian hierarchical geo-level MMM** | Strategic future direction — see below; Bayes-H1–H5 |
| **Dynamic priors** | Research only |
| **State-space / time-varying coefficients MMM** | No prod contract |
| **Robust optimization** | Research extension only |
| **Nevergrad / additional search optimizers** | After VAL-005 evidence; not prod |
| **Additional transforms** (Weibull, logistic, log adstock in prod) | Canonical stack first |
| **Additional optimizers** (beyond simulation path) | Research Sandbox |
| **Online adaptation / adaptive calibration** | Not v1 scope |
| **Auto-retrain** | Operations research — not autonomous prod |
| Bayesian production decisioning | Prod decide blocked (INV-032) |
| Uncertainty-aware allocation | Research / proxy scores |
| Autonomous orchestration | Track 5 |

**Research docs:** [bayesian.md](../02_concepts/bayesian.md), [hierarchical_borrowing.md](../02_concepts/hierarchical_borrowing.md), [robust_optimization_research.md](../02_concepts/robust_optimization_research.md), [ridge_uncertainty_research.md](../04_governance/ridge_uncertainty_research.md).

Historical synthetic “non-goals” remain in [synthetic_validation_roadmap.md §12–13](synthetic_validation_roadmap.md#12-deferred-capability-expansion-after-reliability-program).

### Research Sandbox item — Bayesian Hierarchical Geo-Level MMM

**Status:** Research-track planning only — **not production-ready**; does **not** change current Bayesian modules or prod gates.

**Full phased spec:** [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md)

#### Purpose

Future Bayesian path modeling **geo-level outcomes** and **geo-specific media coefficients** with **national/channel hyper-parameters**, **partial pooling** (local \(\beta_{g,c}\) shrink toward \(\mu_c\); large geos deviate when supported; small/noisy geos borrow strength), and **experiment-informed priors/likelihoods**. Posterior uncertainty feeds **TrustReport** by default — not direct prod decisions.

#### Platform contract preservation (binding)

| Contract | Must preserve |
|----------|----------------|
| DecisionSurface | Same prod simulate/optimize semantics |
| Estimand | Declared quantities unchanged in meaning |
| CalibrationSignal | Structured experiment/replay evidence only |
| TrustReport | Readiness, warnings, posterior diagnostics |
| Release-gate semantics | `approved_for_prod`, promotion, PolicyError |
| Full-panel Δμ | Canonical production counterfactual surface |

> **Rule:** The Bayesian model may change the **estimator**; it may **not** change the **production decision contract**.

#### Required validation before production (future)

Hierarchical synthetic worlds; geo heterogeneity recovery; partial-pooling / small-geo shrinkage validation; experiment-prior recovery; posterior calibration and PPC; convergence diagnostics; Δμ recovery; optimizer recovery; TrustReport and release-gate compatibility; comparison against **Ridge baseline** on the same reliability worlds.

#### Research phases (Bayes-H*)

| Phase | Scope |
|-------|--------|
| **Bayes-H1** | ✅ [DecisionSurface Preservation ADR](bayes_h1_decision_surface_preservation_adr.md) + [architecture refinement](bayesian_hierarchical_geo_mmm_refinement.md) — **no code** |
| **Bayes-H2** | ✅ [CalibrationSignal Mapping ADR](bayes_h2_calibration_signal_mapping_adr.md) — **no code** |
| **Bayes-H2b** | ✅ [Hierarchical Experiment-Prior Scope Rules ADR](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) — **no code** |
| **Bayes-H2c** | Materialize `WORLD-BAYES-*` bundles + runner (blocked: H2c model spec ADR) |
| **Bayes-H3** | Research-only PyMC: sampling, convergence, PPC, TrustReport-compatible artifact (`prod_decisioning_allowed: false`) |
| **Bayes-H4** | Reliability certification: shrinkage, heterogeneity, coverage, experiment-prior, Δμ, Ridge comparison |
| **Bayes-H5** | Production candidacy review — **only if H1–H4 pass**; no prod decisioning until release gates approve |

#### Strengths vs trade-offs (summary)

| Strengths | Trade-offs |
|-----------|------------|
| Regional heterogeneity + pooling stability | Geo panels, compute cost, diagnostics burden |
| Local experiment calibration | Identifiability, convergence risk |
| Posterior uncertainty for TrustReport | Not automatically causal |
| GeoX / experiment-informed alignment | **Not prod-ready** without Bayes-H4 + gates |

#### Explicit non-goal

**Do not** build Bayesian **production decisioning** before reliability-world validation (Bayes-H2–H4) and release-gate approval (Bayes-H5). Existing `INV-032` prod block remains until then.

**Strategic note:** Bayesian hierarchical geo MMM aligns naturally with **GeoX**, **CalibrationSignal** contracts, **hierarchical experiment priors**, and **regional heterogeneity** — but must consume the same platform contracts (DecisionSurface, Estimand, CalibrationSignal, TrustReport, release gates) and **cannot** introduce a new decision surface. Blocked until Phase **5C** exact-recovery investigation completes.

**Investigations:** INV-063–069 (design, pooling validation, posterior calibration, local experiment priors, compute, TrustReport, DecisionSurface compatibility).

---

## Track 5 — Conversational / Orchestration Layer

**Goal (future):** User-facing planning and recommendation experiences that **orchestrate** existing platform capabilities.

| Future capability | Constraint |
|-------------------|------------|
| Conversational planning assistant | Must call DecisionSurface APIs, not reimplement Δμ |
| Recommendation systems | Must attach TrustReport + Estimand context |
| Experiment-planning copilots | Must emit CalibrationSignal-compatible evidence specs |
| Budget-planning copilots | Must respect release gates and promotion |
| Strategy orchestration | No bypass of `approved_for_prod` |
| Explainable recommendations | Must cite contract-backed artifacts, not raw coef tables |

**Explicit:** Orchestration must **consume** platform contracts — not invent parallel semantics, gate names, or “shadow” budget optimizers.

**MMM package-side agents (deferred):** Optional interpretive layers around deterministic MMM outputs — not the modeling engine. See [mmm_package_side_agents_roadmap.md](mmm_package_side_agents_roadmap.md) for prerequisites, boundaries, and five deferred agent roles. Implementation blocked until typed run manifests and failure packets exist.

**Status:** Not implemented in v1.x. Listed for architectural alignment only.

---

## Historical synthetic validation phases (index)

All phases below are **preserved** in [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) under **Track 2**:

| Phase | Status |
|-------|--------|
| 0 — Validation contracts | ✅ Frozen |
| 1A — Materialization architecture | ✅ Frozen |
| 1B — Schema & validator spec | ✅ Frozen |
| 2A — Materializer | ✅ Smoke |
| 2B — Replay materialization | ✅ Smoke |
| 3A — Generators | ✅ Smoke |
| 3B — ScenarioBuilder | ✅ Smoke |
| 2 — Minimal DGP library | Planned |
| 4A — Certification runners | **Next** |
| 4B — ReliabilityScorecard | Planned |
| 5A — Structural lattice | ✅ |
| 5B — Behavioral lattice | ✅ |
| 5C — Exact recovery investigation | **Next** |
| 5D — Threshold governance | ✅ |
| 5E — Drift & TrustReport | ✅ |
| 5F — Monte Carlo | Planned |
| 5F — Monte Carlo reliability | Planned |
| 6 — External benchmarks | Planned |

---

## Unresolved roadmap questions

| ID | Topic | Track |
|----|-------|-------|
| DR-03 | Negative world catalog shape | 2 |
| DR-04 | Threshold ownership (`TBD_v1`) | 2 |
| DR-05 | Runtime vs CI certification split | 2 |
| DR-06 | ReliabilityScorecard release role | 2 |
| — | Formal DecisionSurface / TrustReport type exports vs docs-only names | 1 |
| — | Orchestration MCP/API contract sketch | 5 |

Tracked in [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) and [open_investigations.md](../06_investigations/open_investigations.md).

---

## Recommended next implementation phase

**Track 2 (Reliability Program):** Phase **5F** complete — [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md). **Tier-1** batch (N=100) is operational follow-up, not a new phase.

**Track 4 (Research Sandbox):** Bayes-H1–H2d ADRs ✅; `VAL-BAYES-H2B-SMOKE` ✅. **Next:** Bayes-H3 **research sandbox** (PyMC prototype, `RESEARCH ONLY`). **Production** Bayesian decisioning blocked.

---

## Related documentation

| Document | Role |
|----------|------|
| [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) | Track 2 detailed phases (preserved) |
| [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) | ADRs DR-01–DR-06 |
| [open_investigations.md](../06_investigations/open_investigations.md) | Cross-track investigation backlog |
| [investigation_index.md](../06_investigations/investigation_index.md) | Grouped by platform concern |
| [v1_release_notes.md](../04_governance/v1_release_notes.md) | Shipped prod boundary |
