# Synthetic validation — architecture decision record (ADR)

**ADR bundle ID:** `mmm_synthetic_validation_adr_v1`  
**Status:** Accepted (Phase 0 freeze)  
**Date:** 2026-05-22

**Related contracts:** [groundtruth_contract.md](groundtruth_contract.md) · [validation_registry.md](validation_registry.md) · [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) · [platform_roadmap.md](platform_roadmap.md)

This document records binding decisions for the synthetic validation framework and their relationship to **platform contract preservation**. Implementation phases must not contradict these decisions without a superseding ADR.

**Platform framing:** MMM is one implementation surface within a **Marketing Intelligence Platform**. Long-term ABI: DecisionSurface, Estimand, CalibrationSignal, TrustReport, release-gate semantics, full-panel Δμ — see [platform_roadmap.md](platform_roadmap.md).

---

## Decision 1 — Synthetic worlds are truth systems, not datasets

### Decision

Validation scenarios are defined by **`GroundTruthWorld` documents** (declarative truth). Panel CSVs, Parquet files, and replay JSON files are **materializations**, not the authoritative definition of correctness.

### Rationale

- MMM platform failures are often **contract and gate** failures (fingerprints, readiness, Δμ path), not “wrong R² on a CSV.”
- Datasets without declared lift, optimum, and expected gates cannot support pass/fail certification.
- Competing truth in test code caused v1 duplication (runtime synthetic cert vs inline test DGP); a single truth system prevents regression.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **Curated CSV benchmark suite** | Truth implicit; no `true_regret` or `expected_failures` |
| **Property-based random panels only** | No stable `world_id`; hard to reproduce scorecards |
| **Inline pytest fixtures as truth** | Fragile; untracked across CI vs runtime |

### Consequences

- Phase 1+ must implement materialization from worlds, not world-from-CSV inference.
- CI artifacts may cache materializations but must label them with `world_id` + hash.
- Documentation and operators reference **world IDs**, not file names.

---

## Decision 2 — GroundTruthWorld is the only source of truth

### Decision

All synthetic validation metrics, pass/fail expectations, and certification levels must trace to fields in **`groundtruth_contract.md`**. No parallel truth structs (`DGPConfig`, `FixtureTruth`, duplicated coef arrays in tests).

### Rationale

- Prevents five competing definitions once generators and certifications land.
- Enables ReliabilityScorecard aggregation across worlds with consistent semantics.
- Registry `truth_object` column maps directly to contract dot-paths.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **Truth in certification modules only** | Cert and test drift |
| **Truth in generator code** | Opaque; hard to review |
| **Dual truth: “analytic” + “empirical”** | Ambiguous pass criteria |

### Consequences

- New checks require `validation_id` + world field, not new magic numbers in tests.
- Migration of `CHECK_REGISTRY` to `world_id` is mandatory in Phase 1.
- Code review rejects PRs that assert numeric truth without `world_id` reference.

---

## Decision 3 — Scenario composition preferred over many fixed datasets

### Decision

Use a **`ScenarioBuilder`** (Phase 3) that composes archetype worlds from axes (signal, noise, correlation, sample size, drift, privacy, experiment quality) rather than maintaining hundreds of static datasets.

### Rationale

- Reliability program needs hundreds–thousands of worlds; static files do not scale.
- Cartesian sampling with tagged `scenario_tags` supports stratified scorecards.
- Archetypes (`baseline_world`, `adstock_world`, …) stay small and reviewable.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **One dataset per test file** | Unmaintainable; truth duplication |
| **Fully random worlds without tags** | Non-reproducible debugging |
| **Only seven static worlds forever** | Insufficient for robustness claims |

### Consequences

- Phase 2 delivers archetypes, not 100 CSVs.
- Phase 3 delivers composition policy (full factorial vs sampled lattice) — see Design review.
- Existing exact checks become **named points** on the lattice.

---

## Decision 4 — Certifications consume truth objects, not generated files

### Decision

Certification modules (`SyntheticCertification`, `DecisionCertification`, etc.) take **`GroundTruthWorld` + platform outputs** and compare via **`validation_registry.md`**. They do not read panel CSV columns as ground truth.

### Rationale

- Aligns with Decision 1 and 2.
- Allows inject-only tests (ridge summary injection) where full train is unnecessary.
- File format changes do not redefine correctness.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **Cert from “golden” extension_report JSON** | Golden files hide truth; brittle |
| **Cert from re-fit on materialized panel only** | Conflates estimation error with platform error |
| **Separate truth per certification type** | Registry becomes meaningless |

### Consequences

- Cert reports include `world_id`, `validation_id`, and metric values.
- Runtime extension reports remain **outputs under test**, not truth (except reproducibility reference runs with declared tolerances).

---

## Decision 5 — External synthetic datasets are sanity checks only

### Decision

Google, Robyn, PyMC, and similar public examples may be used for **interop and regression sanity** (Phase 6). They **must not** set `approved_for_prod`, certification levels, or registry thresholds.

### Rationale

- Schema and estimand mismatch with MMM geo-week + decide contracts.
- Upstream changes are not under platform control.
- Passing Robyn does not prove `mmm decide` artifact integrity.

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **Robyn parity as release gate** | Wrong contract surface |
| **Import Google synthetic as GroundTruthWorld** | Undocumented truth |
| **No external checks** | Misses dependency breakage |

### Consequences

- External jobs are advisory or separate workflow badge.
- ADR required to promote any external set to registry-backed truth.

---

## Decision 6 — Synthetic validation proves robustness, not causality

### Decision

Framework outputs (including `ReliabilityScorecard`) measure **engineering robustness under declared generative truth**. They must **never** be marketed or documented as proof of **causal incrementality** on real business data.

### Rationale

- Observational MMM + replay is not randomized lift.
- Operators may over-interpret high synthetic pass rates.
- Real experiments remain at top of evidence hierarchy (roadmap §2).

### Alternatives rejected

| Alternative | Why rejected |
|-------------|--------------|
| **“Synthetic causal validity” wording** | Misleading |
| **Skip disclaimer to simplify docs** | Compliance and trust risk |
| **Use synthetic only for ML metrics (AUC, etc.)** | Ignores decide/governance surface |

### Consequences

- All scorecards and cert reports include non-causal disclaimer boilerplate (Phase 4+).
- Product docs cross-link to [decision_vs_research.md](../02_concepts/decision_vs_research.md).

---

## Decision 7 — World bundle directory materialization (DR-01)

### Decision

Materialize each synthetic world as **Option C: world bundle directory** under `validation/worlds/<world_id>/` with immutable `world_truth.json`, derived Parquet/JSON artifacts, `metadata.json`, and `checksums.json`. See [world_materialization.md](world_materialization.md).

### Rationale

Separates truth from large derived files; supports lineage checksums; aligns with MMM artifact governance and train→decide fixture layout.

### Alternatives rejected

Option A (single `world.json`) and Option B (flat split without bundle manifest) — see comparison table in [world_materialization.md](world_materialization.md).

### Consequences

- Phase 1B defines JSON shapes; Phase 2 implements materializer.
- Certifications must verify `checksums.json` before run.
- No certification-grade hand-edited panels.

---

## Decision 8 — Three independent version dimensions (DR-02)

### Decision

Record **`world_contract_version`**, **`world_generator_version`**, and **`materialization_version`** independently in bundle metadata and certification reports, with PATCH/MINOR/MAJOR bump rules per [truth_versioning.md](truth_versioning.md). Catalog **`world_version`** remains the per-instance identifier.

### Rationale

Schema, DGP authoring, and rendering change on different cadences; ReliabilityScorecard comparability requires explicit filters.

### Alternatives rejected

Single `world_version` string; git SHA as sole version; package version only.

### Consequences

- Phase 0 `world_version` in truth metadata is superseded by the triple + instance version in catalog.
- Major contract bumps invalidate cross-release scorecard comparison without relabeling.

---

## Design review — unresolved questions

### DR-01 — World serialization format ✅ Resolved

**Decision:** Option C — world bundle directory. [world_materialization.md](world_materialization.md)

### DR-02 — Truth versioning ✅ Resolved

**Decision:** Three-version policy. [truth_versioning.md](truth_versioning.md)

### DR-03 — Negative world representation ✅ Resolved (Phase 5F)

**Decision:**

1. **`metadata.negative_world: true`** on catalog entries expected to fail gates.
2. **`artifact_truth.expected_failures`** — enumerated `{gate_id, expected_outcome, optional_error_class}` per world (not ad hoc PolicyError-only templates).
3. **Shared archetypes** — negative worlds use the same bundle layout, materializer, and certification runner as positive worlds.
4. **Scorecard policy** — negative worlds count toward **gate-contract** certification; excluded from recovery pass-rate numerators unless tagged `negative:gate_test`.

**Rejected:** Separate materialization pipeline; unlisted failures without registry rows.

### DR-04 — Threshold ownership process (draft resolution)

**Status:** Draft resolution — Phase 5D ([reliability_threshold_governance.md](reliability_threshold_governance.md), INV-057).

**Decision (draft):**

1. **Ownership:** Platform lead + governance committee sign-off to promote any row from `TBD_v1_runtime` → `approved`.
2. **Stratification:** Thresholds are **per `validation_id`** with optional **per-`archetype_id` / scenario tag** overlays documented in the threshold version record — not silent per-world tweaks.
3. **Metric classes:** Thresholds are versioned **by metric class** (`decision_grade`, `diagnostic_attribution`, `trust_modifier`, `structural`). Decision-grade bounds are tighter; diagnostic attribution bounds are looser and **non-blocking by default**.
4. **Provisional runtime:** `TBD_v1_runtime` values in code **do not** gate production until promoted.
5. **Evidence traceability:** Each approved threshold cites `required_evidence` from [validation_registry.md](validation_registry.md) §14 (world families, lattice stratum, Monte Carlo tier).
6. **INV-056 policy:** Coefficient and transform recovery thresholds remain **diagnostic** unless a product adopts an explicit attribution certification profile.

**Rationale:** Exact recovery investigation showed Δμ can pass while coef fails; treating coef thresholds as release blockers misstates Ridge BO decision reliability.

**Still open:** Numeric promotion of VAL-012 severity bands to `approved` pending tier-1 Monte Carlo.

### DR-05 — Runtime vs CI certification ownership

**Open:** Which certifications run on every train extension report vs CI-only batch; whether ReliabilityScorecard is nightly-only or release-tag gate. v1 today: synthetic cert on train + CI parity — future split must not duplicate truth.

### DR-06 — ReliabilityScorecard release role ✅ Resolved (Phase 5F)

**Decision:**

| Tier | N worlds | Scorecard role |
|------|----------|----------------|
| tier_0_pilot | &lt;100 | **Advisory** — publish JSON; no semver block |
| tier_1_calibration | ≥100 | **CI regression** on `decision_reliability_score` + structural; diagnostic failures warn only |
| tier_2_release_review | ≥1000 | **Required** for DR-04 threshold approval evidence |
| Production | — | Scorecard does **not** replace promotion workflow or live TrustReport |

**Regression:** Block merge if `decision_reliability_score` drops &gt;0.05 vs baseline tier snapshot while unit tests pass.

**Rationale:** [monte_carlo_reliability_program.md](monte_carlo_reliability_program.md), [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md).

---

## Decision 9 — Platform contracts are the long-term ABI (DR-07)

### Decision

Planning and release policy transition from a **feature-centric MMM package roadmap** to a **contract-centric Marketing Intelligence Platform roadmap** ([platform_roadmap.md](platform_roadmap.md)). All production paths must preserve:

- **DecisionSurface** — canonical production decision API semantics  
- **Estimand** — declared estimand for replay and Δμ  
- **CalibrationSignal** — structured calibration evidence (replay units, registry)  
- **TrustReport** — readiness / certification rollup before recommendations  
- **Release-gate semantics** — `approved_for_prod`, promotion, fingerprints  
- **Full-panel Δμ** — canonical counterfactual on panel geometry  

### Rationale

- Prevents orchestration and future model families from forking budget semantics.  
- Reliability proving (GroundTruthWorld + certification runners) becomes the gate for modeling expansion, not feature count.  

### Consequences

- New production modeling requires contract compatibility review + reliability-program evidence.  
- Research capabilities remain in **Research Sandbox** until Phase 4A+ validation.  
- Synthetic validation phases 0–3B are **Track 2** deliverables, not discarded.  

---

## Decision 10 — Bayes-H1 DecisionSurface preservation (Track 4)

### Decision

All future **Bayesian hierarchical Geo MMM** work (and any Bayesian estimator) must preserve **DecisionSurface supremacy**: full-panel Δμ via existing simulate/optimize paths; posteriors in **TrustReport** only; **CalibrationSignal** + **Estimand** + **release gates** unchanged; no Bayesian-specific approval path.

Binding detail: [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md).

### Rationale

- Largest Track 4 risk is contract fragmentation, not sampler correctness.  
- Reliability Program 5C–5F established decision vs attribution separation; Bayesian must not regress it.

### Consequences

- **No PyMC / sampler / prior code** until `WORLD-BAYES-*` worlds materialize (Bayes-H2b exit).  
- Bayes-H3+ must pass ADR validation checklist V1–V7 before prod candidacy.  
- Superseding draw-based prod optimization requires new ADR.

---

## Decision 11 — Bayes-H2 CalibrationSignal mapping (Track 4)

### Decision

All GeoX, CLS, A/B, holdout, and replay evidence enters Bayesian hierarchical Geo MMM **only** through **CalibrationSignal**, mapped by a signal registry to prior / likelihood / pseudo-observation / penalty / exclusion per scope, estimand, uncertainty, freshness, and conflict rules. TrustReport must expose evidence quality; posteriors must not bypass replay or estimand gates.

Binding detail: [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md).

### Rationale

- Experiment ingestion is the highest-risk Bayesian design surface after DecisionSurface fragmentation.  
- Ridge replay, freshness, and VAL-006/007 assume a single evidence ABI.

### Consequences

- Seven validation worlds (Bayes-H2 checklist) required before Bayes-H3 sampler code.  
- **Bayes-H2b** refines hierarchy-level scope propagation.  
- Conflicting evidence must never silently average.

---

## Decision 12 — Bayes-H2b Hierarchical experiment-prior scope rules (Track 4)

### Decision

After Bayes-H2 maps a CalibrationSignal to a mechanism, **hierarchy propagation** is governed by a contract-governed **scope graph** (not a naive tree): upward gates (coverage, representativeness, freshness, conflict), downward caveats (`borrowed_strength_from_parent`), **no lateral calibration** except pooling-only, deterministic conflict precedence for sparse geos, explicit **claim levels**, and TrustReport **`hierarchy_evidence`** fields. Seven `WORLD-BAYES-*` worlds must materialize before Bayes-H3.

Binding detail: [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md).

### Rationale

- Bayes-H2 without propagation rules allows local GeoX to be misread as national truth.  
- Reliability strategy requires auditable scope behavior on controlled worlds before samplers.

### Consequences

- No PyMC until `WORLD-BAYES-*` bundles exist in catalog.  
- Borrowed-strength estimates must never be labeled as direct experiment results.  
- National CLS must not imply DMA causal lift.

---

## Decision 13 — Bayes-H2d Hierarchical model specification (Track 4)

### Decision

The **hierarchical Bayesian Geo MMM** generative spec is defined in architecture only: geo-channel partial pooling (\(\beta_{g,c} \sim \mathcal{N}(\mu_c, \tau_c^2)\)), Gaussian observation model concept, **CalibrationSignal-only** evidence terms, TrustReport-owned uncertainty, **posterior diagnostic only**, and **DecisionSurface** compatibility via point-policy adapter. **Bayes-H3 research sandbox** is authorized after `VAL-BAYES-H2B-SMOKE`; **production** Bayesian decisioning is not.

Binding detail: [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md).

### Rationale

- Evidence routing (H2b) must precede model shape so fitters cannot invent parallel ingress.  
- Model expansion is encouraged in sandbox; promotion remains gated ([ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md)).

### Consequences

- PyMC/sampler code is **Tier 2 sandbox** work — labeled `RESEARCH ONLY — NOT DECISION GRADE`.  
- Bayes-H4 recovery worlds required before prod candidacy.  
- No alternate DecisionSurface or optimizer-on-posterior paths.

---

## Decision 14 — Bayes-H3 research sandbox inference backend (Track 4)

### Decision

**Bayes-H3 uses PyMC as the initial research sandbox backend.** NumPyro is reserved as a candidate backend for later scale/performance phases after H3/H4 establish model correctness and recovery behavior.

Binding detail: [bayes_h3_research_sandbox_backend_adr.md](bayes_h3_research_sandbox_backend_adr.md).

### Rationale

- PyMC best matches the current goal: transparent, auditable, research-only hierarchical MMM diagnostics under strict governance.  
- Aligns with existing repo code (`pymc_trainer`, `bayes_h3_sandbox.model`) and the PyMC-Marketing MMM ecosystem conceptually.  
- NumPyro deferred until recovery evidence exists — performance must not outrun correctness.

### Consequences

- Sanctioned fit path remains `run_sandbox_fit` (PyMC in `mmm.research.bayes_h3_sandbox`).  
- Backend choice is **not** production authorization; all Bayesian outputs stay research-only until promotion gates pass.  
- NumPyro requires a future ADR amendment after Bayes-H4.

---

## Decision 15 — Bayes-H4 recovery worlds (Track 4)

### Decision

**Bayes-H4** adds deterministic generative recovery worlds (`WORLD-BAYES-H4-*`) to validate whether the Bayes-H3 sandbox recovers known \(\mu_c\), \(\tau_c\), and \(\beta_{g,c}\) under research-only fences. Promotion remains blocked.

Binding detail: [bayes_h4_recovery_worlds_adr.md](bayes_h4_recovery_worlds_adr.md).

### Rationale

- H3 proved safe execution; H4 proves scientific behavior on controlled truth.  
- Recovery metrics are diagnostic report-only until pilot thresholds are calibrated.

### Consequences

- `run_h4_recovery_world` must not emit production DecisionSurface or optimizer inputs.  
- Pass/fail promotion thresholds are **TBD** (see INV-071).  
- Production Bayesian decisioning remains blocked.

---

## Supersession

| ADR ID | Supersedes | Status |
|--------|------------|--------|
| `mmm_synthetic_validation_adr_v1` | Informal test-only DGP truth | Active |
| `bayes_h1_decision_surface_preservation_v1` | Informal Bayes prod path experiments | Active (Track 4) |
| `bayes_h2_calibration_signal_mapping_v1` | Informal §3 mapping in refinement doc | Active (Track 4) |
| `bayes_h2b_hierarchical_experiment_prior_scope_rules_v1` | Informal §3.3 propagation in refinement doc | Active (Track 4) |
| `bayes_h2d_hierarchical_model_spec_v1` | Informal §2 hierarchy in refinement doc | Active (Track 4) |
| `bayes_h3_research_sandbox_backend_v1` | Informal backend choice in sandbox code | Active (Track 4) |
| `bayes_h4_recovery_worlds_v1` | Informal recovery validation | Active (Track 4) |
| DR-01 | Phase 1A | Option C bundle — [world_materialization.md](world_materialization.md) |
| DR-02 | Phase 1A | Three-version policy — [truth_versioning.md](truth_versioning.md) |
| DR-07 | Feature-centric MMM-only roadmap narrative | [platform_roadmap.md](platform_roadmap.md) |

To supersede: new ADR section with migration plan for registry and frozen contract version.

---

## Compliance checklist (for PRs touching validation)

- [ ] No new truth outside `GroundTruthWorld` contract fields  
- [ ] New measurement has `validation_id` in registry  
- [ ] Pass criteria not invented in code comments — use `TBD_v1` until calibrated  
- [ ] No external dataset wired to `approved_for_prod`  
- [ ] User-facing text does not claim causal incrementality from synthetic passes  
