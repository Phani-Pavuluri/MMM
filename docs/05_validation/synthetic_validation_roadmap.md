# Synthetic validation framework — roadmap

**Role in platform planning:** This document is **Track 2 — Reliability & Validation Program** of the [Marketing Intelligence Platform roadmap](platform_roadmap.md). It preserves the full historical phased execution plan (Phases 0–6). It does not replace completed work — it reclassifies it under contract-driven, reliability-first platform strategy.

**Status:** Phases 0–1B frozen; Phases **2A–3B**, **4A**, **4B-1** through **4B-5**, **4C**, **5A–5F** landed. Core reliability architecture complete; **tier-1 Monte Carlo execution** is operational follow-up.

| Platform track | Document |
|----------------|----------|
| **Master roadmap (5 tracks)** | [platform_roadmap.md](platform_roadmap.md) |
| **Track 2 (this doc)** | Phases 0–6 below |
| **Track 3 (prod decisioning)** | [v1_release_notes.md](../04_governance/v1_release_notes.md), [production_readiness.md](../04_governance/production_readiness.md) |

| Phase | Document |
|-------|----------|
| Phase 0 — contracts | [groundtruth_contract.md](groundtruth_contract.md), [validation_registry.md](validation_registry.md), [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) |
| Phase 1A — materialization | [world_materialization.md](world_materialization.md), [truth_versioning.md](truth_versioning.md), [world_catalog.md](world_catalog.md) |
| Phase 1B — schema & validator spec | [world_schema.md](world_schema.md), [world_bundle_schema.md](world_bundle_schema.md), [world_validator_spec.md](world_validator_spec.md) |

**Audience:** Platform engineers, governance owners, and release gate owners proving MMM behavior under declared `GroundTruthWorld` truth.

**MMM positioning:** The `mmm` package is one implementation surface. Platform contracts (DecisionSurface, Estimand, CalibrationSignal, TrustReport, release gates, full-panel Δμ) are the long-term ABI — see [platform_roadmap.md § Platform Contract Preservation](platform_roadmap.md#platform-contract-preservation).

---

## 1. Executive goal

### What the package already validates (v1.0.0)

The current MMM release already proves **local correctness** and **contract adherence** through a mix of unit tests, controlled DGP checks, governance modules, and nightly certification:

| Area | Current evidence |
|------|------------------|
| **Unit behavior** | Transform math, design matrix construction, config validators, artifact serializers |
| **DGP recovery** | Noiseless / lightly structured synthetic panels (`test_synthetic_dgp_recovery`, exact certification registry) |
| **Replay consistency** | Replay ETL, fold-aligned refit, prod replay gates, generalization gap reporting |
| **Governance contracts** | Production readiness, promotion workflow, model release state, calibration freshness |
| **Optimizer behavior** | Grid-derived optima, directional fallback modes, repeatability scenarios |
| **Production readiness** | Rollup of synthetic, optimizer, repro, and contract completeness |
| **Artifact integrity** | Decision bundle tier, fingerprint matching, semantic contracts on prod decide |
| **Operational hooks** | Nightly CI, performance certification (optional), decision trace persistence |

These checks are valuable and remain the **floor** for any future framework. They are implemented as **fixed scenarios** and **single-world** certifications, not as a composable ground-truth validation platform.

### Remaining gap

Current validation demonstrates that **individual components behave as designed** under known, narrow conditions. It does **not** yet support defensible statements about:

| Gap | Why it matters |
|-----|----------------|
| **Behavior across diverse worlds** | Signal strength, collinearity, and sample size interact; one DGP per check hides interaction failures |
| **Decision reliability** | Δμ and optimize paths may be locally exact but unstable under drift, noise, or misspecified calibration |
| **Replay robustness** | Replay error distributions under weak vs strong experiment evidence are not characterized |
| **Calibration robustness** | Weighted replay and fold-aligned modes need stress across experiment quality and freshness |
| **Operational resilience** | Failure modes under prod gates (fingerprints, readiness, promotion) are tested pointwise, not systematically |
| **Failure behavior under production conditions** | Expected rejections (PolicyError, blocked readiness) are not catalogued per world |

### North star

Eventually the platform should support operator- and release-facing statements of the form:

> **Validated across N synthetic worlds** spanning low/medium/high signal, collinearity regimes, experiment evidence quality, calibration drift, privacy perturbation, and decision scenarios — with reported bias, variance, coverage, replay error, decision regret, and artifact failure rates.

**N** is staged (100 → 1,000 → 10,000) via the **Validation & Reliability Program** (Section 10). Reports are **ReliabilityScorecards**, not causal claims.

### What this does not prove

**This framework does not prove causal incrementality.** Synthetic worlds encode **known generative truth** for engineering validation. Real randomized experiments remain the primary evidence layer for causal lift. Public benchmark datasets are **secondary sanity checks** (Section 9).

---

## 2. Validation philosophy

### Evidence hierarchy

```text
Real experiments (randomized lift, geo tests, incrementality studies)
        ↓
        Primary evidence for causal estimands in production use
Synthetic validation framework (this roadmap)
        ↓
        Ground-truth engineering proof across controlled worlds
Public benchmark datasets (Google, Robyn, PyMC examples, etc.)
        ↓
        External sanity checks and interoperability — not prod certification
```

**Why public synthetic datasets are secondary**

| Reason | Implication |
|--------|-------------|
| Truth is implicit or undocumented | Hard to attach pass/fail criteria to “expected regret” or “expected gate” |
| Schema mismatch | MMM geo-week contracts, replay units, and extension reports do not map 1:1 |
| Version drift | Upstream example changes break CI without clarifying what regressed |
| Wrong failure mode | Passing a Robyn-style fit does not certify `mmm decide` artifact contracts |

Benchmarks belong in **Phase 6** as **external sanity checks**, not as substitutes for `GroundTruthWorld`-driven certification.

### Ground truth is mandatory

Avoid validation designs that only emit **(X → y)** tables. Those support ML benchmarking but not MMM platform certification.

**Require `GroundTruthWorld` objects** that declare:

- Generative media and experiment truth
- Decision truth (Δμ, optimal budget, regret)
- Shift truth (drift, changepoints, policy changes)
- Artifact truth (which gates should pass or fail)

Panel rows are **derived views** of a world, not the source of truth.

### Validation-first, narrow scope

Prove **existing** contracts (semi_log, geometric adstock, Hill, full-panel Δμ, decide paths, governance modules) before expanding modeling surface. New modeling capabilities are **deferred** until the reliability program demonstrates measurable improvement (Section 10 governing rule; Section 12 deferred expansion).

---

## 3. GroundTruthWorld contract

**Frozen in Phase 0:** [groundtruth_contract.md](groundtruth_contract.md). Summary below; normative field definitions live in that document.

```text
GroundTruthWorld
├── world_id: str
├── schema_version: str
├── time_structure
├── geo_structure
├── media_truth
├── experiment_truth
├── decision_truth
├── shift_truth
└── artifact_truth
```

### `time_structure`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `granularity` | e.g. weekly | Panel builder, CV split axis |
| `n_periods` | Horizon length | Adstock burn-in, replay window alignment |
| `calendar_start` | Optional anchor | Seasonality extensions (future); replay unit dates |
| `seasonality_truth` | Optional explicit seasonal component | Decomposition diagnostics (non-decision) |

### `geo_structure`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `n_geos` | Cross-section size | Partial pooling, identifiability tests |
| `geo_ids` | Stable keys | Fingerprinting, replay geo masks |
| `hierarchy` | Optional parent/child | Hierarchy extensions (research) |

### `media_truth`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `coefficients` | True β on modeling scale | Recovery metrics, calibration targets |
| `adstock` | True decay / carryover params | Adstock recovery, design matrix checks |
| `saturation` | True Hill (or canonical) params | Saturation recovery |
| `interactions` | Optional cross-channel terms | Separability / identifiability scenarios |
| `spend_process` | How media series is generated | Collinearity and noise scenarios |

### `experiment_truth`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `lift` | True incremental lift per unit | Replay target construction |
| `treatment_effects` | Geo-time ATT or equivalent | Evidence registry validation |
| `variance` | Observation noise on lift | Calibration robustness |
| `freshness` | Age / stale flags | Calibration freshness gates |

### `decision_truth`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `true_delta_mu` | Counterfactual Δμ for reference scenarios | Δμ recovery, simulate certification |
| `true_optimal_budget` | Constrained optimum under true surface | Optimizer certification |
| `true_regret` | Suboptimality of observed allocator | Decision quality, reliability program |

### `shift_truth`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `changepoints` | Structural breaks in media or coef | Drift detection, readiness warnings |
| `coefficient_drift` | Smooth or step drift in β | Calibration drift, stress reports |
| `policy_changes` | Transform or budget rule changes | Governance / transform_policy gates |
| `privacy_changes` | Noise or aggregation perturbation | Fingerprint and panel QA behavior |

### `artifact_truth`

| Field (proposed) | Purpose | Downstream consumers |
|------------------|---------|----------------------|
| `expected_gates` | Which governance checks must pass | GovernanceCertification |
| `expected_failures` | Expected PolicyError or blocked readiness | Negative testing, prod workflow validation |
| `expected_certification_levels` | e.g. synthetic `exact`, optimizer mode | Production readiness rollups |

**Contract rules (design)**

- One world → one authoritative truth document; panel CSV/Parquet is a **materialization**.
- Truth definitions must not be duplicated in test files; tests reference world IDs and registry entries.
- `schema_version` enables backward-compatible evolution without silent semantic drift.

---

## 4. Validation target registry

**Frozen in Phase 0:** [validation_registry.md](validation_registry.md) (`VAL-001`–`VAL-014`, pass criteria `TBD_v1`). Summary table below.

| Capability | Truth object | Metric (examples) | Pass criteria | Priority |
|------------|--------------|-------------------|---------------|----------|
| Coefficient recovery | `media_truth.coefficients` | Bias, RMSE, sign error rate | TBD during execution | P0 |
| Adstock recovery | `media_truth.adstock` | Param error, carryover correlation | TBD during execution | P0 |
| Hill recovery | `media_truth.saturation` | Param error, monotone feature error | TBD during execution | P0 |
| Δμ recovery | `decision_truth.true_delta_mu` | Absolute / relative Δμ error | TBD during execution | P0 |
| Optimizer recovery | `decision_truth.true_optimal_budget` | L1 distance, regret vs truth | TBD during execution | P0 |
| Replay consistency | `experiment_truth` + implied y | Replay train/holdout loss, unit match rate | TBD during execution | P0 |
| Calibration robustness | `experiment_truth` (quality tiers) | Calibration weight stability, false attach rate | TBD during execution | P1 |
| Decision safety | `artifact_truth.expected_gates` | `decision_safe`, unsupported-question rate | TBD during execution | P0 |
| Drift detection | `shift_truth` | Time-to-detect, false alarm rate | TBD during execution | P1 |
| Governance behavior | `artifact_truth` | Expected pass/fail vs actual | TBD during execution | P0 |
| Certification behavior | `artifact_truth.expected_certification_levels` | Level match (exact vs incomplete) | TBD during execution | P0 |
| Artifact integrity | `artifact_truth` | Bundle tier, fingerprint match, schema validity | TBD during execution | P0 |
| Reproducibility | Run snapshot + world seed | Bitwise or tolerance-based output match | TBD during execution | P1 |
| Promotion workflow | `artifact_truth` + promotion record | Promotion validation pass/fail | TBD during execution | P1 |

**Registry consumers:** ScenarioBuilder, certification modules (Section 7), reliability program (Section 8), CI/nightly orchestration.

**Anti-pattern:** Defining pass thresholds in roadmap prose. Thresholds are calibrated on pilot worlds in Phase **5** (n=100 pilot) and recorded in versioned registry YAML/JSON after Phase **4A** runners exist.

---

## 5. Synthetic DGP library roadmap

Recommend a **minimal** library of world **archetypes** — not dozens of generators. Each archetype implements the `GroundTruthWorld` contract for a focused slice of truth.

### `baseline_world`

| | |
|--|--|
| **Purpose** | Linear-ish semi_log surface, low interaction, baseline recovery |
| **Ground truth produced** | `media_truth.coefficients`, `decision_truth.true_delta_mu` |
| **Capabilities validated** | Coefficient recovery, Δμ recovery, decision safety |
| **Dependencies** | Phase 1 contract, Phase 0 registry |

### `adstock_world`

| | |
|--|--|
| **Purpose** | Known geometric adstock with impulse and sustained spend |
| **Ground truth produced** | `media_truth.adstock`, carryover states |
| **Capabilities validated** | Adstock recovery, design matrix carryover, Δμ under dynamics |
| **Dependencies** | `baseline_world` time/geo scaffolding |

### `saturation_world`

| | |
|--|--|
| **Purpose** | Hill saturation with monotone feature path |
| **Ground truth produced** | `media_truth.saturation` |
| **Capabilities validated** | Hill recovery, optimizer behavior on curved surfaces |
| **Dependencies** | `adstock_world` (canonical stack) |

### `geo_world`

| | |
|--|--|
| **Purpose** | Multi-geo panels, partial pooling stress, identifiability |
| **Ground truth produced** | `geo_structure`, optional hierarchy flags |
| **Capabilities validated** | Coefficient recovery by geo, identifiability signals |
| **Dependencies** | `baseline_world` |

### `experiment_world`

| | |
|--|--|
| **Purpose** | Replay units with known lift and noise |
| **Ground truth produced** | `experiment_truth` (lift, variance, freshness) |
| **Capabilities validated** | Replay consistency, calibration robustness |
| **Dependencies** | `geo_world`, replay ETL contracts |

### `optimizer_world`

| | |
|--|--|
| **Purpose** | Two-channel and saturated surfaces with grid-computable optima |
| **Ground truth produced** | `decision_truth.true_optimal_budget`, `true_regret` |
| **Capabilities validated** | Optimizer recovery, certification_mode alignment |
| **Dependencies** | `saturation_world`, existing optimizer certification semantics |

### `drift_world`

| | |
|--|--|
| **Purpose** | Changepoints and coefficient drift |
| **Ground truth produced** | `shift_truth` |
| **Capabilities validated** | Drift detection, calibration freshness, readiness warnings |
| **Dependencies** | `experiment_world`, governance modules |

**Explicitly not in minimal library:** Weibull/logistic transforms, Bayesian generative models, state-space coefficients (non-goals, Section 11).

---

## 6. Scenario composition roadmap

**Execution:** Reliability Program **Phase 3B** (Section 10). Phase 3A provides deterministic archetype generators only.

### `ScenarioBuilder` (planned — Phase 3B)

Composable layer that **parameterizes** archetype worlds instead of hardcoding one-off datasets.

**Example axes:**

| Axis | Levels | Maps to |
|------|--------|---------|
| Signal | low / medium / high | Coef scale vs noise |
| Noise | low / medium / high | Observation and lift variance |
| Correlation | low / medium / severe | Cross-channel collinearity |
| Sample size | small / medium / large | `n_geos`, `n_periods` |
| Drift | on / off | `shift_truth` |
| Privacy loss | on / off | `privacy_changes` in `shift_truth` |
| Experiment quality | weak / medium / high | `experiment_truth` SNR and match quality |

**Output:** A fully specified `GroundTruthWorld` (plus materialized panel and replay fixtures).

### Why composition over hardcoded datasets

| Composition | Hardcoded datasets |
|-------------|-------------------|
| Truth declared once per scenario ID | Truth duplicated in test code |
| Cartesian products for reliability program | Explosion of unmaintainable files |
| Registry-driven pass criteria | Fragile per-file assertions |
| Reproducible `world_id` + seed | Ad hoc seeds in each test |

**Relationship to today’s `CHECK_REGISTRY`:** Existing exact checks become **smoke instances** of specific `world_id`s, not parallel truth definitions (aligns with v1 single-source synthetic certification).

---

## 7. Certification framework roadmap

**Execution:** Reliability Program **Phase 4A** (runners) and **Phase 4B** (ReliabilityScorecard). Planned certification types extend today’s `*_certification_report` pattern. Each consumes `GroundTruthWorld` + run artifacts; emits versioned report JSON.

### `SyntheticCertification`

| | |
|--|--|
| **Purpose** | Mathematical correctness of transforms, design matrix, Δμ path |
| **Inputs** | World materialization, config, `run_synthetic_certification_suite` registry |
| **Outputs** | `certification_level`, per-check status, `world_id` |
| **Pass/fail logic** | All P0 registry targets for world class; thresholds TBD during execution |
| **Dependencies** | Phase 2 `baseline_world`, `adstock_world`, `saturation_world`; existing `CHECK_REGISTRY` migration |

### `DecisionCertification`

| | |
|--|--|
| **Purpose** | `mmm decide simulate` / optimize vs `decision_truth` |
| **Inputs** | Trained or injected ridge summary, world panel, scenario specs |
| **Outputs** | Δμ error, regret, `decision_safe` alignment |
| **Pass/fail logic** | Registry targets for decision safety + Δμ; TBD during execution |
| **Dependencies** | Decision service, `optimizer_world` |

### `ReplayCertification`

| | |
|--|--|
| **Purpose** | Replay ETL, refit modes, generalization gap vs `experiment_truth` |
| **Inputs** | Replay units derived from world, train config |
| **Outputs** | Replay loss delta, unit coverage, prod gate expectation match |
| **Pass/fail logic** | TBD during execution |
| **Dependencies** | `experiment_world`, replay prod gate |

### `OptimizerCertification`

| | |
|--|--|
| **Purpose** | Extend current optimizer cert with world-linked regret and mode |
| **Inputs** | `decision_truth.true_optimal_budget`, optimizer result |
| **Outputs** | `certification_mode`, L1 distance, regret |
| **Pass/fail logic** | Align with analytic vs directional semantics; TBD during execution |
| **Dependencies** | Existing `optimizer_certification.py` contract, `optimizer_world` |

### `GovernanceCertification`

| | |
|--|--|
| **Purpose** | `artifact_truth.expected_gates` vs actual governance JSON |
| **Inputs** | Extension report, config, world expectations |
| **Outputs** | Per-gate pass/fail, unexpected pass/fail list |
| **Pass/fail logic** | Exact match on expected pass/fail set; TBD during execution |
| **Dependencies** | Production readiness, promotion, model release modules |

### `ArtifactCertification`

| | |
|--|--|
| **Purpose** | Decision bundle tier, fingerprints, semantic contracts |
| **Inputs** | Decide `--out` payload, train extension report, world |
| **Outputs** | Schema validation, fingerprint match, tier |
| **Pass/fail logic** | TBD during execution |
| **Dependencies** | Artifact factory, prod decide path |

### `ReliabilityCertification`

| | |
|--|--|
| **Purpose** | Aggregate scorecard over many worlds (Section 8) |
| **Inputs** | Batch of world results |
| **Outputs** | `ReliabilityScorecard` |
| **Pass/fail logic** | Release policy on scorecard percentiles; TBD during execution |
| **Dependencies** | Phase 5 program (Section 10) |

---

## 8. Reliability program roadmap

**Execution:** Reliability Program **Phase 5** (large-scale sweeps) with scorecard rollup in **Phase 4B**. Later-stage validation over composed scenarios — measures **robustness**, not causality.

### Scale tiers

| Tier | Worlds (n) | Use |
|------|------------|-----|
| Pilot | 100 | Threshold calibration, smoke for scorecard schema |
| Standard | 1,000 | Pre-release gate, nightly extended (optional) |
| Deep | 10,000 | Major version / infra change; offline batch |

### Metrics (aggregated per capability)

- Bias and variance of coefficient / transform recovery
- Coverage of confidence or stability heuristics (where defined)
- Replay error distribution
- Decision regret vs `decision_truth`
- Decision stability (allocation L1 under perturbation)
- Artifact failure rate (unexpected PolicyError, bundle reject)
- Optimizer stability (regret variance, mode fraction)
- Drift detection performance (latency, false positives)

### Output: `ReliabilityScorecard`

Versioned document (JSON) with:

- `n_worlds`, scenario lattice definition, seed policy
- Per-capability percentile summaries
- Regression deltas vs baseline scorecard
- Explicit disclaimer: **robustness under declared generative truth, not causal incrementality**

---

## 9. External benchmark roadmap

| Source | Role | Production certification? |
|--------|------|-------------------------|
| **Google / Meridian-style examples** | Schema and API sanity, optional parity checks | **No** |
| **Robyn examples** | External fit behavior comparison (research) | **No** |
| **PyMC / Bayesian examples** | Posterior plumbing regression (research path) | **No** |

**Purpose:** Detect gross interoperability or dependency regressions.

**Not purpose:** Substitute for `GroundTruthWorld` certification or prod readiness approval.

**Phase 6 deliverable:** Thin adapter scripts + documented skip reasons when contracts do not align.

---

## 10. Validation & Reliability Program (v1.x → v2 foundation)

This section **extends** the phased execution plan (Section 11) without replacing completed phases. It makes **reliability proving** a first-class objective: the platform must be able to show measured behavior against declared `GroundTruthWorld` truth across many worlds, not only pass fixed smoke checks.

**Relationship to earlier sections:**

| Earlier roadmap section | Reliability program phase |
|-------------------------|---------------------------|
| §5 Minimal DGP library (`baseline_world`, …) | Informs archetypes; generative equations land with Phase 2 library work **after** truth composition (3B) stabilizes |
| §6 Scenario composition | **Phase 3B** — ScenarioBuilder |
| §7 Certification framework | **Phase 4A** — runners; **Phase 4B** — ReliabilityScorecard |
| §8 Reliability program (Monte Carlo) | **Phase 5** — large-scale sweeps |

**Investigation backlog:** [open_investigations.md](../06_investigations/open_investigations.md) tracks open gaps (thresholds `TBD_v1`, DR-03–DR-06, cert runner wiring).

### Governing rule (release policy)

> **No new modeling capability becomes production-grade unless it demonstrates measurable improvement against reliability-program worlds and certifications.**

“Measurable improvement” means: defined `validation_id`(s) from [validation_registry.md](validation_registry.md), executed via Phase **4A** runners on Phase **3B**+ worlds, summarized on Phase **4B** ReliabilityScorecard with versioned thresholds (post–pilot calibration). Ad-hoc unit tests or external benchmarks alone are insufficient.

### Phase 3A — Minimal GroundTruthWorld generators ✅ (smoke)

| | |
|--|--|
| **Goal** | Move from hand-authored worlds to deterministic generated worlds |
| **Scope** | Deterministic baseline generator; deterministic replay generator; seed reproducibility; generator writes **only** `world_truth.json`; materializer remains owner of all derived artifacts |
| **Deliverables** | `mmm/validation/synthetic/generators.py`; `validation/worlds/WORLD-003-generated-baseline/`; `validation/worlds/WORLD-004-generated-replay/`; determinism tests; validator + materializer compatibility |
| **Success criteria** | Same seed → identical world truth; different seeds → controlled variation; generated worlds pass L1–L3 validation; replay worlds load via `load_calibration_units_from_json` |
| **Dependencies** | Phase 2B |
| **Status** | ✅ Smoke complete — see [world_materialization.md](world_materialization.md) §9 |

### Phase 3B — ScenarioBuilder MVP ✅ (smoke)

| | |
|--|--|
| **Goal** | Reusable world composition rather than manual world authoring |
| **Scope** | `ScenarioBuilder` + `ScenarioSpec`; deterministic truth only; uses `compose_archetype_truth` + existing materializer |
| **Capabilities** | `n_geos`, `n_periods`, `channels`, `noise_level`, `correlation_level`, `seasonality`, `drift`, `experiment_quality`, `privacy_loss`, `missingness` |
| **Deliverables** | `scenario_builder.py`, [scenario_builder.md](scenario_builder.md), `WORLD-005`–`WORLD-007` scenario smoke worlds, tests |
| **Success criteria** | Same spec → identical truth; materialized bundles pass L1–L3; severe collinearity → `artifact_truth.expected_warnings`; drift → `drift_truth` entries |
| **Dependencies** | Phase 3A |
| **Not in MVP** | Monte Carlo sweeps, certification runners, complex DGP-in-panel simulation |
| **Expected risk** | Combinatorial explosion at scale — requires sampling policy before Phase 5 |

### Phase 4A — Structural certification runners ✅ (smoke)

| | |
|--|--|
| **Goal** | Executable structural/contract certification on materialized world bundles (no train/decide) |
| **Certification areas** | Bundle integrity; checksums; replay loader; transform/metadata/governance truth; decision_truth structure; DecisionSurface / Estimand / CalibrationSignal / TrustReport / release-gate compatibility |
| **Deliverables** | `certification_runner.py`, `certification_registry.py`, `synthetic_world_certification_report.json` per run; [certification_runner.md](certification_runner.md) |
| **Deferred (explicit skip)** | VAL-001–014 behavioral rows (`requires_rich_dgp_worlds`, `requires_train_decide_execution`, `requires_thresholds`) — never fake-pass |
| **Success criteria** | WORLD-001/002 and ScenarioBuilder worlds certify; tamper/contract negative tests fail honestly |
| **Dependencies** | Phase 3B |
| **Not in scope** | Coefficient/optimizer/Δμ recovery, Monte Carlo, ReliabilityScorecard |

### Phase 4B-1 — Rich DGP materialization ✅ (smoke)

| | |
|--|--|
| **Goal** | Deterministic KPI panels generated from truth via semi_log + geometric adstock + Hill |
| **Deliverables** | `dgp_materializer.py`, `WORLD-008-exact-recovery`, `dgp_diagnostics.parquet`, [dgp_materialization.md](dgp_materialization.md) |
| **Success criteria** | Formula tests pass; L1–L3 validator; stable checksums; `world_truth.json` never mutated |
| **Not in scope** | Train/decide runners, VAL recovery, Monte Carlo, ScenarioBuilder expansion |

### Phase 4B-2 — Train/decide recovery certification ✅ (smoke)

| | |
|--|--|
| **Goal** | Train Ridge on WORLD-008 DGP panel; recover coef / Δμ under TBD_v1_runtime tolerances |
| **Deliverables** | `recovery_certification.py`, runner integration, `train_config.yaml`, tests |
| **Executed** | Coef recovery, transform consistency, analytic Δμ, fingerprint, decision artifact |
| **Skipped** | Optimizer (`requires_optimizer_truth_thresholds`); replay (no units on WORLD-008) |
| **Dependencies** | Phase 4B-1 |

### Phase 4B-3 — Optimizer recovery world ✅ (smoke)

| | |
|--|--|
| **Goal** | Non-placeholder `decision_truth` + known optimal budget; VAL-005 execution |
| **Deliverables** | `WORLD-009-optimizer-recovery`, `optimizer_truth.py`, REC-4B3-* checks, `tests/test_synthetic_optimizer_recovery.py` |
| **Executed** | Grid-recorded `true_optimal_budget`; production `optimize_budget_via_simulation`; allocation L1, objective gap, budget conservation |
| **Dependencies** | Phase 4B-2 |

### Phase 4B-4 — Replay calibration recovery world ✅ (smoke)

| | |
|--|--|
| **Goal** | Known experiment lift in `experiment_truth`; replay units from truth; VAL-006 execution |
| **Deliverables** | `WORLD-010-replay-recovery`, `replay_truth.py`, REC-4B4-* checks, `tests/test_synthetic_replay_recovery.py` |
| **Executed** | Full-panel transform + estimand mask; pre-window adstock; fitted vs true replay-implied lift |
| **Dependencies** | Phase 4B-3 |

### Phase 4B-5 — Drift / identifiability recovery ✅ (smoke)

| | |
|--|--|
| **Goal** | Prove governance/certification surfaces degraded reliability (not perfect recovery) |
| **Deliverables** | `WORLD-011-drift-recovery`, `WORLD-012-identifiability-recovery`, `reliability_truth.py`, REC-4B5-* |
| **Executed** | Drift: pre-train + post MAE blow-up; identifiability: VIF warning + readiness downgrade |
| **Skipped honestly** | Coef recovery across drift; VAL-012 full runner (INV-055) |
| **Dependencies** | Phase 4B-4 |

### Phase 4C — ReliabilityScorecard ✅ (MVP)

| | |
|--|--|
| **Goal** | Aggregate WORLD-008–012 certification into capability-level reliability view |
| **Deliverables** | `reliability_scorecard.py`, `validation/synthetic_reliability_scorecard.json`, [reliability_scorecard.md](reliability_scorecard.md) |
| **Executed** | pass/partial/fail/skip scoring; expected-skip exclusion; conservative readiness interpretation |
| **Not in scope** | Production gate; Monte Carlo; new worlds |
| **Dependencies** | Phase 4B-5 |

### Phase 5 — Large-scale world sweeps

| | |
|--|--|
| **Goal** | Stress-test the package at scale under composed scenarios |
| **World variations** | Signal-to-noise; number of geos; number of channels; drift levels; experiment quality; privacy loss; collinearity; sparse media; missing data |
| **Measures** | Failure rate; instability; false confidence; calibration degradation; optimizer failures |
| **Deliverables** | Reliability trend reports; failure taxonomy; release comparison reports (delta vs baseline scorecard) |
| **Scale tiers** | Pilot 100 → standard 1,000 → deep 10,000 (see §8 scale table) |
| **Success criteria** | Stable scorecard schema; regression policy when unit tests pass but scorecard degrades |
| **Dependencies** | Phase 4B; Phase 3B lattice |
| **Expected risk** | CI cost and flakiness at deep tier — offline batch, manifest-hosted bundles |

### Phase 6 — External benchmarks (unchanged)

See [§9 External benchmark roadmap](#9-external-benchmark-roadmap). Benchmarks remain **non-blocking** sanity checks; they cannot satisfy the governing rule above.

---

## 11. Phased execution plan (summary & status)

### Phase 0 — Validation contracts ✅ (frozen)

| | |
|--|--|
| **Deliverables** | [groundtruth_contract.md](groundtruth_contract.md), [validation_registry.md](validation_registry.md), [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) |
| **Dependencies** | v1.0.0 governance and certification modules (read-only reference) |
| **Expected risk** | Schema churn if fields are overloaded — mitigated by freeze + ADR change control |
| **Validation requirements** | Stakeholder sign-off on registry; no duplicate truth in tests |

### Phase 1A — Materialization architecture ✅ (frozen)

| | |
|--|--|
| **Deliverables** | [world_materialization.md](world_materialization.md) (DR-01: Option C bundle), [truth_versioning.md](truth_versioning.md) (DR-02: three versions), [world_catalog.md](world_catalog.md) |
| **Dependencies** | Phase 0 |
| **Expected risk** | Large bundle storage in git — mitigated by manifest + CI cache policy (deferred) |
| **Validation requirements** | ADR DR-01/DR-02 accepted; no Python materializer |

### Phase 1B — GroundTruthWorld schema & validator spec ✅ (frozen)

| | |
|--|--|
| **Deliverables** | [world_schema.md](world_schema.md), [world_bundle_schema.md](world_bundle_schema.md), [world_validator_spec.md](world_validator_spec.md) |
| **Dependencies** | Phase 1A |
| **Expected risk** | Schema overfitting to Ridge-only paths — mitigated by family minimums + v1 prod constraints |
| **Validation requirements** | Spec review complete; no JSON Schema files or code (by design) |

### Phase 2A — Materializer ✅ (smoke)

| | |
|--|--|
| **Deliverables** | `mmm/validation/synthetic/materializer.py`, `validator.py`, `validation/worlds/WORLD-001-baseline/`, tests |
| **Dependencies** | Phase 1B |
| **Expected risk** | Accidental truth write-back — mitigated: materializer never writes `world_truth.json` |
| **Validation requirements** | INV-004 reproducible checksums; L1–L3 on WORLD-001 |

### Phase 2B — Replay materialization ✅ (smoke)

| | |
|--|--|
| **Deliverables** | `WORLD-002-replay`, `replay_units.py`, L3 replay validator checks, loader test |
| **Dependencies** | Phase 2A |
| **Expected risk** | Replay JSON shape drift vs `units_io` — mitigated by `ReplayEstimandSpec.from_dict` test |
| **Validation requirements** | `load_calibration_units_from_json` + stable `replay_sha256` |

### Phase 3A — Minimal GroundTruthWorld generators ✅ (smoke)

See [§10 Phase 3A](#phase-3a--minimal-groundtruthworld-generators--smoke) for goal, scope, and success criteria.

### Phase 3B — ScenarioBuilder MVP ✅ (smoke)

See [§10 Phase 3B](#phase-3b--scenariobuilder-mvp--smoke) and [scenario_builder.md](scenario_builder.md).

### Phase 2 — Minimal DGP library (parallel / after 3B stabilizes)

| | |
|--|--|
| **Deliverables** | Seven archetypes (Section 5), equation-backed panel generation where needed |
| **Dependencies** | Phase 1B; Phase 3B scenario axes for collinearity, noise, drift |
| **Expected risk** | Generators accidentally become “training data product” |
| **Validation requirements** | Each archetype maps to ≥1 P0 registry row; thresholds filled during pilot (Phase 5) |

### Phase 4A — Structural certification runners ✅

See [§10 Phase 4A](#phase-4a--structural-certification-runners--smoke) and [certification_runner.md](certification_runner.md).

### Phase 4B-1 — Rich DGP materialization ✅

See [§10 Phase 4B-1](#phase-4b-1--rich-dgp-materialization--smoke) and [dgp_materialization.md](dgp_materialization.md).

### Phase 4B-2 — Train/decide recovery certification ✅

See [§10 Phase 4B-2](#phase-4b-2--traindecide-recovery-certification--smoke).

### Phase 4B-3 — Optimizer recovery ✅

See [§10 Phase 4B-3](#phase-4b-3--optimizer-recovery-world--smoke).

### Phase 4B-4 — Replay calibration recovery ✅

See [§10 Phase 4B-4](#phase-4b-4--replay-calibration-recovery-world--smoke).

### Phase 4B-5 — Drift / identifiability recovery ✅

See [§10 Phase 4B-5](#phase-4b-5--drift--identifiability-recovery--smoke).

### Phase 4C — ReliabilityScorecard ✅

See [reliability_scorecard.md](reliability_scorecard.md).

### Phase 5A — Small lattice sweep ✅

See [lattice_sweep.md](lattice_sweep.md). Fixed 12-world ScenarioBuilder grid; materialize + structural certification + scorecard by axis; report at `validation/reports/lattice_sweep_mvp_report.json`. Still no large Monte Carlo.

### Phase 5B — Rich behavioral lattice sweep ✅

See [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md). Fixed 10-world rich DGP grid (exact / optimizer / replay / drift / identifiability); recovery-enabled certification; behavioral vs structural scores; report at `validation/reports/behavioral_lattice_sweep_mvp_report.json`.

#### Phase 4B/5B evidence — roadmap pivot

| Finding | Implication |
|---------|-------------|
| Structural reliability **high** (~0.89 on behavioral lattice) | Contracts, bundles, CERT-4A path are sound |
| Behavioral reliability **materially lower** (~0.57) | Scientific confidence limited by recovery, not structure |
| Exact-recovery worlds fail **coef + transform** recovery | WORLD-008 / L5B `exact_recovery` — REC-4B2-001–003 |
| Platform contracts **stable** | DecisionSurface / Estimand / TrustReport not the bottleneck |
| Package can **tell us where it is wrong** | Next work = **understand failures**, not more features |

> **Strategic shift:** From expanding validation coverage → **proving and improving behavioral recovery** under known truth.

### Phase 5C — Exact Recovery Investigation Program ✅

| | |
|--|--|
| **Goal** | Explain WORLD-008 exact-recovery failures before fixing or expanding the model family |
| **Investigation** | [INV-056](../06_investigations/open_investigations.md#inv-056--exact-recovery-failure-analysis-phase-5c) (critical) |
| **Spec** | [exact_recovery_investigation.md](exact_recovery_investigation.md) |
| **Questions** | Ridge shrinkage; hyperparameter search; identifiability; transform compensation; CV vs recovery; tolerances; truth-pinned transforms; sensitivity to noise/geos/periods/channels; theoretical recovery ceiling |
| **Deliverables** | Investigation report; failure taxonomy; sensitivity analysis; threshold recommendations; Bayesian world recommendations |
| **Non-goals** | No new MMM methods, Bayesian, state-space, transforms, optimizers, orchestration |
| **Success** | Explain observed failures **before** improving them |

### Phase 5D — Reliability threshold governance and metric semantics ✅

| | |
|--|--|
| **Goal** | Encode INV-056 findings: separate decision-grade vs attribution-diagnostic metrics |
| **Deliverables** | [reliability_threshold_governance.md](reliability_threshold_governance.md); registry §14; scorecard v1.1 metric-class scores; DR-04 draft |
| **Non-goals** | New models, Bayesian, Nevergrad, prod decisioning changes |
| **Investigations** | INV-056 closed; INV-057/059 addressed; INV-058/060 open |

### Phase 5E — Drift detection and TrustReport semantics ✅

| | |
|--|--|
| **Goal** | Complete VAL-012 with dedicated drift runner; formalize TrustReport interpretation |
| **Deliverables** | [drift_detection.md](drift_detection.md), [trust_report_semantics.md](trust_report_semantics.md), `drift_detection_runner.py`, scorecard v1.2 |
| **Investigations** | INV-055 closed |

### Phase 5F — Monte Carlo reliability program ✅

| | |
|--|--|
| **Goal** | Statistically characterize reliability; recommend thresholds (not auto-approve) |
| **Deliverables** | [monte_carlo_reliability_program.md](monte_carlo_reliability_program.md), [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md), `monte_carlo_reliability.py`, pilot JSON |
| **Resolved** | DR-03 negative worlds; DR-06 scorecard role |
| **Follow-up** | Tier-1 N=100 batch runner (operational, not new phase) |

### Phase 4C — ReliabilityScorecard

See [§10 Phase 4C](#phase-4c--reliabilityscorecard).

### Phase 5 — Lattice + investigation program (5A–5F)

| Sub-phase | Doc |
|-----------|-----|
| 5A ✅ | [lattice_sweep.md](lattice_sweep.md) |
| 5B ✅ | [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md) |
| 5C ✅ | [exact_recovery_investigation.md](exact_recovery_investigation.md) |
| 5D ✅ | [reliability_threshold_governance.md](reliability_threshold_governance.md) |
| 5E ✅ | [drift_detection.md](drift_detection.md), [trust_report_semantics.md](trust_report_semantics.md) |
| 5F ✅ | [monte_carlo_reliability_program.md](monte_carlo_reliability_program.md) |

See also [§10 Phase 5](#phase-5--large-scale-world-sweeps) (historical scale-tier narrative; **5F** is the executable Monte Carlo phase).

### Phase 6 — External benchmarks

| | |
|--|--|
| **Deliverables** | Optional adapters, documented non-goals, separate CI job (non-blocking or advisory) |
| **Dependencies** | Phases 0–2 (contracts stable) |
| **Expected risk** | False confidence if benchmarks promoted to prod gates |
| **Validation requirements** | Benchmarks cannot set `approved_for_prod` |

---

## 12. Deferred capability expansion (Research Sandbox — Track 4)

The following capabilities belong to **[platform_roadmap.md Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)**. They are **explicitly deferred** from production until the [Reliability Program](#10-validation--reliability-program-v1x--v2-foundation) (Phases 4A–5) and the [governing rule](#governing-rule-release-policy) are satisfied.

| Deferred capability | Track |
|---------------------|-------|
| Bayesian hierarchical geo-level MMM | Research Sandbox — [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) (Bayes-H1–H5); blocked until **5C** |
| Dynamic priors | Research Sandbox |
| State-space / time-varying coefficients MMM | Research Sandbox |
| Robust optimization | Research Sandbox |
| Nevergrad / additional search optimizers | Research Sandbox |
| New transforms (Weibull, logistic, log adstock in prod) | Research Sandbox |
| Additional optimizers (prod) | Research Sandbox (after VAL-005 on worlds) |
| Online adaptation / adaptive calibration | Research Sandbox |
| Auto-retrain | Research Sandbox / operations |
| Production Bayesian decisioning | Research Sandbox |
| Uncertainty-aware allocation | Research Sandbox |
| Autonomous orchestration | Track 5 — Conversational / Orchestration |

**Platform gate:** No major production modeling expansion until Phase **5C** (exact recovery), **5D** (threshold governance), and **5E** (drift validation) complete — [platform_roadmap.md](platform_roadmap.md).

**Still in scope for the reliability program (not deferred):** explicit operator **promotion workflow** (governance); promotion **validation** is VAL-014. Autonomous promotion is out of scope.

### Bayesian Hierarchical Geo-Level MMM (Track 4 — parallel planning)

Research-track only; see [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md). Synthetic validation tie-in:

| Bayes phase | Reliability program link |
|-------------|--------------------------|
| **Bayes-H2b** ✅ | [Scope rules ADR](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) + [validation worlds catalog](../BAYES_H2B_VALIDATION_WORLDS_001.md) |
| **Bayes-H2c** | Materialize `WORLD-BAYES-*` + VAL-BAYES runner (no PyMC) |
| **Bayes-H2d** | Hierarchical model spec ADR — blocked until H2c CI smoke |
| **Bayes-H4** | Certification + scorecard stratum: shrinkage, heterogeneity, coverage, Δμ, optimizer, Ridge comparison |
| **Bayes-H5** | Production candidacy — **not** automatic; requires release-gate ADR |

**Non-goal:** No Bayesian production decisioning before Bayes-H2–H4 evidence and release-gate approval. Does not change current Bayesian code paths.

---

## 13. Explicit non-goals (framework boundary)

The synthetic validation framework **does not** claim or substitute for:

| Non-goal | Rationale |
|----------|-----------|
| Causal incrementality from observational MMM | Requires real experiments (Section 2) |
| Public benchmarks as prod certification | Phase 6 adapters only; cannot set `approved_for_prod` |
| Duplicate truth in tests | Tests reference `world_id`, not parallel coef arrays |
| Bayesian production decisioning before reliability worlds | Bayes-H2–H4 + release gates required; see [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) |

Modeling expansions listed in Section 12 follow the **governing rule** in Section 10.

---

## 14. Success criteria (roadmap → implementation)

Implementation should not start until:

| Criterion | Evidence |
|-----------|----------|
| **GroundTruthWorld contract stable** | Versioned schema approved; no open semantic disputes on `decision_truth` / `artifact_truth` |
| **Validation registry agreed** | P0/P1 rows signed; owners for each capability |
| **DGP scope approved** | Seven archetypes only; no scope creep to new transforms |
| **Certification scope approved** | Seven cert types mapped to existing reports; migration from v1 `CHECK_REGISTRY` planned |
| **No duplicated truth definitions** | Written rule: tests reference `world_id`, not inline coef arrays |
| **Threshold process defined** | How TBD becomes versioned thresholds (pilot 100 worlds) |
| **Non-goals acknowledged** | Product/governance sign-off on Sections 12–13 |
| **Governing rule adopted** | No prod-grade modeling without reliability-program evidence (Section 10) |

---

## Related documentation

| Document | Relationship |
|----------|--------------|
| [platform_roadmap.md](platform_roadmap.md) | Master contract-driven platform roadmap (5 tracks) |
| [synthetic_certification.md](../04_governance/synthetic_certification.md) | Current exact-check floor; migrate into SyntheticCertification |
| [optimizer_certification.md](../04_governance/optimizer_certification.md) | Optimizer cert semantics |
| [statistical_validation.md](../02_concepts/statistical_validation.md) | What tests do / do not prove today |
| [v1_release_notes.md](../04_governance/v1_release_notes.md) | Production scope boundary |
| [production_readiness.md](../04_governance/production_readiness.md) | Readiness rollup consumer |
| [open_investigations.md](../06_investigations/open_investigations.md) | Living backlog (INV-*, DR-03–DR-06) |
| [investigation_index.md](../06_investigations/investigation_index.md) | Grouped investigation index |
| [exact_recovery_investigation.md](exact_recovery_investigation.md) | Phase 5C — exact recovery investigation (next) |
| [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md) | Research Sandbox — Bayesian hierarchical geo MMM (Bayes-H1–H5) |

---

## Unresolved design questions

Tracked in [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) § Design review. **DR-01** and **DR-02** resolved in Phase 1A. **DR-07** platform contracts — [platform_roadmap.md](platform_roadmap.md). Open: DR-03–DR-06.

---

## Recommended next executable phase

**Track 4** — [BAYES_H2B_VALIDATION_WORLDS_001.md](../BAYES_H2B_VALIDATION_WORLDS_001.md) accepted. **Next:** bundle materialization + runner 002. Bayes-H2d/H3 blocked. Tier-1 MC (N=100) remains recommended before DR-04 threshold approval.
