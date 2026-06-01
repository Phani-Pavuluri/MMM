# Open investigations backlog

**Purpose:** Single living register of unresolved gaps, design debt, conceptual risks, audit findings, and deferred work that can materially affect building a world-class **Marketing Intelligence Platform** (MMM is one implementation surface).

**Planning frame:** [platform_roadmap.md](../05_validation/platform_roadmap.md) — contract-centric, reliability-first. Investigations map to platform tracks below.

**Scope:** Evidence-backed items only. Previously fixed issues are omitted unless later docs re-open them.

**Maintenance:** Add or close investigations via PR; link evidence paths. Do not implement fixes in this file.

### Platform investigation categories

Use these categories when adding or reclassifying items (in addition to technical categories on each record):

| Category | Concern |
|----------|---------|
| **contract compatibility** | DecisionSurface, Estimand, CalibrationSignal, TrustReport, release gates |
| **semantic drift** | Docs vs code vs artifacts diverging on estimand or gate meaning |
| **decision-surface fragmentation risk** | Alternate prod budget/simulate paths, curve-only optimizers |
| **orchestration compatibility** | Future agents/copilots bypassing gates or contracts |
| **replay semantics** | Refit modes, evidence registry, geo/aggregate misuse |
| **TrustReport consistency** | Readiness rollup vs decide warnings vs promotion |
| **certification reliability gaps** | CHECK_REGISTRY vs worlds, thresholds TBD_v1, L4 missing |

### Current production moat (investigations lens)

Moat is **estimand discipline**, **replay governance**, **decision semantics**, **reliability proving**, and **contract preservation** — not sheer model complexity. See [platform_roadmap.md](../05_validation/platform_roadmap.md#roadmap-evolution-philosophy).

### Investigation → platform track map

| Track | Example INV IDs |
|-------|-----------------|
| **1 — Platform Contract Layer** | INV-014, INV-024, INV-054 |
| **2 — Reliability & Validation** | INV-008, INV-017, INV-020–023, INV-027, INV-058, INV-060, INV-070 |
| **3 — Core Production Decisioning** | INV-001–007, INV-009–011, INV-028, INV-042 |
| **4 — Research Sandbox** | INV-032–037, INV-040, INV-049, INV-052, INV-063–069 (Bayesian hierarchical geo MMM) |
| **5 — Orchestration** | INV-039 (autonomous agents deferred) |

**Review cadence (recommended):**

| Severity / status | Cadence |
|-----------------|---------|
| `critical`, `high` (open) | Every release |
| `medium` (open) | Every major roadmap phase |
| `research`, `postponed`, `accepted limitation`, `revisit later` | Quarterly |

---

## Investigation records

### INV-001 — Optimizer certification on synthetic surfaces, not real-panel geometry

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-001 |
| **Title** | Optimizer certification uses toy synthetic surfaces; directional_fallback is not exact recovery |
| **Category** | optimization |
| **Severity** | high |
| **Status** | open |
| **First identified in** | v1.0.0 governance docs |
| **Evidence sources** | [optimizer_certification.md](../04_governance/optimizer_certification.md), [v1_release_notes.md](../04_governance/v1_release_notes.md) §6, [synthetic_certification.md](../04_governance/synthetic_certification.md) |
| **Problem statement** | `optimizer_certification_report` certifies allocation on deterministic synthetic Δμ surfaces with grid reference optima. Corner-dominant scenarios pass under `directional_fallback` (channel dominance only). |
| **Why it matters** | Operators may treat optimizer cert pass as proof of budget optimality on real panels with collinearity, constraints, and non-smooth objectives. |
| **Risk type** | financial decision |
| **Production impact** | Prod can show `optimizer_certification` pass while real-panel objective geometry differs materially. |
| **Current behavior** | Grid search on same simulate path; `directional_fallback` when ≥95% budget on one channel. Readiness warns on directional mode but does not block by default. |
| **Desired end state** | Certification tied to GroundTruthWorld panels (VAL-005) with declared pass thresholds; clear disclosure when mode is directional only. |
| **Blocking dependencies** | Phase 2 DGP library; VAL-005 thresholds (`TBD_v1`); certification runner on worlds |
| **Suggested validation** | VAL-005 on `decision_truth` worlds; compare cert mode to `artifact_truth.expected_certification_levels` |
| **Suggested owner area** | `mmm/optimization/optimizer_certification.py`, governance |
| **Recommended phase** | Synthetic validation Phase 4 (certification framework) |
| **Related investigations** | INV-008, INV-017, INV-023 |
| **Notes** | Not a bug — scope limitation. |

---

### INV-002 — Production readiness default allows decide with severe warning only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-002 |
| **Title** | `approved_for_prod=false` blocks decide only when `require_production_certification: true` |
| **Category** | governance |
| **Severity** | high |
| **Status** | open |
| **First identified in** | v1.0.0 production readiness |
| **Evidence sources** | [production_readiness.md](../04_governance/production_readiness.md), [v1_release_notes.md](../04_governance/v1_release_notes.md) §5 |
| **Problem statement** | Default prod decide emits severe warning when readiness fails but continues unless strict gate enabled. |
| **Why it matters** | Organizations may assume v1 “prod” always fail-closed on certification rollup failure. |
| **Risk type** | operational |
| **Production impact** | Budget decisions possible with failed synthetic cert, severe replay gap, or missing optimizer cert (depending on config). |
| **Current behavior** | `production_readiness_decide_surface` severe warning; `PolicyError` only if `governance.require_production_certification: true`. |
| **Desired end state** | Explicit org policy documented; optional default strict mode for regulated deployments. |
| **Blocking dependencies** | Product policy decision (not code-only) |
| **Suggested validation** | Contract tests on prod decide with/without strict gate; operator runbook scenarios |
| **Suggested owner area** | `mmm/governance/production_readiness.py`, decision service |
| **Recommended phase** | Governance hardening (post-v1 policy) |
| **Related investigations** | INV-001, INV-013 |
| **Notes** | Intentional v1 default; investigate org adoption risk. |

---

### INV-003 — Observational MMM and replay do not establish causal incrementality

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-003 |
| **Title** | Platform does not claim randomized incremental lift from observational fit + replay |
| **Category** | causal validity |
| **Severity** | critical |
| **Status** | accepted limitation |
| **First identified in** | Phase 0 synthetic ADR; v1 release notes |
| **Evidence sources** | [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) ADR causal disclaimer, [v1_release_notes.md](../04_governance/v1_release_notes.md) §6, [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md) |
| **Problem statement** | Replay calibration and MMM attribution improve internal discipline but do not replace designed experiments for causal estimands. |
| **Why it matters** | Misinterpretation is the primary reputational and financial risk for an MMM platform. |
| **Risk type** | causal |
| **Production impact** | Prod gates enforce contracts, not field lift truth. |
| **Current behavior** | Disclaimers on cert reports, model cards, synthetic ADR; `decision_safe` is contract safety not causal proof. |
| **Desired end state** | Consistent non-causal language on all customer-facing outputs; synthetic scorecard cannot set `approved_for_prod` from external benchmarks alone. |
| **Blocking dependencies** | N/A (communication + process) |
| **Suggested validation** | Doc audit; VAL-014 governance checks on negative worlds when implemented |
| **Suggested owner area** | Governance, documentation |
| **Recommended phase** | Ongoing |
| **Related investigations** | INV-011, INV-012, INV-040 |
| **Notes** | Accepted platform boundary — must stay visible. |

---

### INV-004 — Fingerprint train/decide alignment uses legacy fallback path

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-004 |
| **Title** | Pre-v2 artifacts compare `sha256_panel_keycols_sorted_csv` when `sha256_combined` absent |
| **Category** | artifact integrity |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Documentation truth audit (fingerprint v2 migration) |
| **Evidence sources** | [documentation_truth_audit.md](../documentation_truth_audit.md), [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), [config_yaml.md](../01_getting_started/config_yaml.md) |
| **Problem statement** | Legacy fingerprint comparison omits config, transforms, seeds, and planning assumptions included in v2 `sha256_combined`. |
| **Why it matters** | Train↔decide mismatch on controls, transforms, or seeds may pass legacy checks. |
| **Risk type** | reproducibility |
| **Production impact** | Prod decide may proceed with warning on legacy artifacts; waiver path available. |
| **Current behavior** | Prefer `sha256_combined`; fallback to panel-only hash with warning; waiver via signed JSON. |
| **Desired end state** | All prod bundles emit v2; legacy path sunset with migration timeline. |
| **Blocking dependencies** | Operator migration of stored runs |
| **Suggested validation** | Tests in `test_decision_artifact_hardening.py`; drift monitoring parity |
| **Suggested owner area** | `mmm/data/fingerprint.py`, `mmm/governance/decision_fingerprint.py` |
| **Recommended phase** | Artifact lineage migration |
| **Related investigations** | INV-005, INV-028 |
| **Notes** | Documented legacy-compatible behavior. |

---

### INV-005 — Full-panel replay refit in prod requires explicit waiver

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-005 |
| **Title** | `replay_refit_mode=full_panel_refit` with replay calibration risks leakage without prod waiver |
| **Category** | calibration |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Decision artifact contract; prod safety checklist |
| **Evidence sources** | [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), [prod_safety_checklist.md](../04_governance/prod_safety_checklist.md), [config_yaml.md](../01_getting_started/config_yaml.md) |
| **Problem statement** | Full-panel replay refit uses experiment windows on the same panel used for fit unless fold-aligned or holdout diagnostic modes are used. |
| **Why it matters** | Misconfiguration can inflate replay fit and readiness scores. |
| **Risk type** | scientific validity |
| **Production impact** | Prod training blocked unless `fold_aligned`, `holdout_only_diagnostic`, or `full_panel_replay_refit_prod_waiver_path`. |
| **Current behavior** | `replay_refit_prod_policy` enforces waiver; surfaced on `replay_refit_prod_governance`. |
| **Desired end state** | Default prod templates use `fold_aligned`; waivers rare and audited. |
| **Blocking dependencies** | Operator template adoption |
| **Suggested validation** | `tests/test_replay_refit_mode.py`; prod policy regression suite |
| **Suggested owner area** | `mmm/governance/replay_refit_prod_policy.py` |
| **Recommended phase** | Operator runbook / template enforcement |
| **Related investigations** | INV-007, INV-043 |
| **Notes** | Policy enforced in code; operational adherence is the gap. |

---

### INV-006 — Decision stress is train-time only, not recomputed at decide

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-006 |
| **Title** | `decision_stress_report` does not run at `mmm decide` time |
| **Category** | governance |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | v1.0.0 release notes; decision_stress doc |
| **Evidence sources** | [decision_stress.md](../04_governance/decision_stress.md), [v1_release_notes.md](../04_governance/v1_release_notes.md) §6 |
| **Problem statement** | Stress probes allocation stability at train time; decide-time panel/config may differ. |
| **Why it matters** | Operators may assume stress covers live decide scenarios. |
| **Risk type** | operational |
| **Production impact** | None at decide unless train report is stale. |
| **Current behavior** | `stress_scope`: `train_time` \| `train_time_signal_only`; `auto_budget_change` always false. |
| **Desired end state** | Clear operator docs; optional future decide-time stress (post-v1). |
| **Blocking dependencies** | Product scope for decide-time probes |
| **Suggested validation** | Extension report field presence on train; absence on decide JSON |
| **Suggested owner area** | `mmm/governance/decision_stress.py` |
| **Recommended phase** | Post-v1 governance |
| **Related investigations** | INV-002 |
| **Notes** | v1 intentional scope. |

---

### INV-007 — Severe replay generalization gap warns by default, blocks only if configured

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-007 |
| **Title** | `block_on_severe_replay_gap` defaults false; severe gap may still allow readiness approval paths |
| **Category** | calibration |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Calibration docs; production readiness |
| **Evidence sources** | [calibration.md](../02_concepts/calibration.md), [production_readiness.md](../04_governance/production_readiness.md) |
| **Problem statement** | Replay train/holdout gap severity is advisory unless org enables blocking. |
| **Why it matters** | Overfit replay objective may pass other readiness checks. |
| **Risk type** | scientific validity |
| **Production impact** | Severe gap blocks `approved_for_prod` per readiness rules; configurable block on calibration path. |
| **Current behavior** | Gap in `calibration_summary`; readiness uses severity; `block_on_severe_replay_gap` optional. |
| **Desired end state** | Org policy on default block; GroundTruthWorld negative scenarios (DR-03). |
| **Blocking dependencies** | VAL-006 thresholds; operator policy |
| **Suggested validation** | VAL-006 replay consistency; readiness integration tests |
| **Suggested owner area** | Calibration, production readiness |
| **Recommended phase** | Governance policy + synthetic VAL-006 |
| **Related investigations** | INV-005, INV-017 |
| **Notes** | |

---

### INV-008 — Synthetic certification CHECK_REGISTRY not yet driven by GroundTruthWorld bundles

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-008 |
| **Title** | Runtime synthetic cert uses inline DGP checks, not `world_id` / VAL registry execution |
| **Category** | DGP/synthetic |
| **Severity** | high |
| **Status** | postponed |
| **First identified in** | Synthetic validation roadmap Phase 4–5 |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md), [synthetic_certification.md](../04_governance/synthetic_certification.md), [validation_registry.md](../05_validation/validation_registry.md) |
| **Problem statement** | `CHECK_REGISTRY` proves implementation consistency; VAL-001–014 on authored worlds not yet automated. |
| **Why it matters** | Two parallel validation stories until migration completes. |
| **Risk type** | reproducibility |
| **Production impact** | Prod readiness still depends on CHECK_REGISTRY exact tier. |
| **Current behavior** | `run_synthetic_certification_suite(mode="exact")` on train; CI calls same registry. |
| **Desired end state** | Certification runner reads `world_truth.json` + materialized bundles; maps to VAL IDs. |
| **Blocking dependencies** | Phase 2 DGP library; Phase 4 certification framework |
| **Suggested validation** | Roadmap Phase 4 deliverables; no duplicated coef arrays in tests |
| **Suggested owner area** | `mmm/governance/synthetic_certification.py`, `mmm/validation/synthetic/` |
| **Recommended phase** | Synthetic validation Phase 4 |
| **Related investigations** | INV-001, INV-016, INV-023 |
| **Notes** | Partial mitigation: single CHECK_REGISTRY for runtime + CI (v1 blocker fix). |

---

### INV-009 — Identifiability and optimization approval can be waived in prod

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-009 |
| **Title** | Identifiability limits and `override_unsafe` require explicit waiver artifacts in prod |
| **Category** | statistical validity |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Prod safety checklist; statistical_validation baseline beat waiver |
| **Evidence sources** | [prod_safety_checklist.md](../04_governance/prod_safety_checklist.md), [statistical_validation.md](../02_concepts/statistical_validation.md), `tests/test_override_unsafe_prod.py` |
| **Problem statement** | High VIF / low identifiability score blocks `approved_for_optimization` in prod unless waivers or `require_beats_baselines_for_approval: false`. |
| **Why it matters** | Waivers can re-enable optimization on collinear media splits. |
| **Risk type** | financial decision |
| **Production impact** | `approved_for_optimization` false when identifiability fails; waivers allow override paths. |
| **Current behavior** | `governance_service` + scorecard; `override_unsafe_waiver_path` for unsafe APIs. |
| **Desired end state** | Waiver audit trail in decision trace; separability-driven experiment scheduler used before waive. |
| **Blocking dependencies** | Operator discipline |
| **Suggested validation** | Prod policy regression; feature separability extension outputs |
| **Suggested owner area** | `mmm/governance/policy.py`, extensions identifiability |
| **Recommended phase** | Governance operations |
| **Related investigations** | INV-042, INV-010 |
| **Notes** | Waivers are intentional escape hatches. |

---

### INV-010 — Panel QA prod block can be waived via `prod_block_waiver`

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-010 |
| **Title** | `extensions.panel_qa.prod_block_severity` block bypass requires `prod_block_waiver: true` |
| **Category** | governance |
| **Severity** | medium |
| **Status** | open |
| **First identified in** | Prod safety checklist |
| **Evidence sources** | [prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) |
| **Problem statement** | Data quality failures can be overridden for prod optimization path. |
| **Why it matters** | Bad panels may still reach budget optimization with waiver. |
| **Risk type** | operational |
| **Production impact** | Conditional block before optimization gates. |
| **Current behavior** | Panel QA extension; waiver flag documented in checklist. |
| **Desired end state** | Waiver tied to decision trace and promotion record. |
| **Blocking dependencies** | Trace schema extensions |
| **Suggested validation** | Extension smoke tests; prod checklist review |
| **Suggested owner area** | Panel QA extension |
| **Recommended phase** | Governance operations |
| **Related investigations** | INV-009, INV-028 |
| **Notes** | |

---

### INV-011 — Aggregate-only experiment evidence on geo panels

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-011 |
| **Title** | National/aggregate evidence must not support subgeo causal claims (`supports_subgeo_claims=false`) |
| **Category** | estimand |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Experiment evidence PR docs |
| **Evidence sources** | [experiment_evidence.md](../02_concepts/experiment_evidence.md), [calibration.md](../02_concepts/calibration.md) |
| **Problem statement** | Weighted replay can include aggregate-only units; compatibility resolver excludes incompatible uses but operator error remains possible. |
| **Why it matters** | DMA-level budgets informed by national lift is a common failure mode. |
| **Risk type** | causal |
| **Production impact** | Evidence registry gate in prod Ridge replay mode. |
| **Current behavior** | Compatibility flags; aggregate-only excluded from subgeo claims in docs. |
| **Desired end state** | Validator checks on replay unit geo scope vs evidence tier in worlds (VAL-006). |
| **Blocking dependencies** | GroundTruthWorld experiment_quality axes |
| **Suggested validation** | VAL-006; negative worlds for misuse |
| **Suggested owner area** | `mmm/calibration/` evidence registry |
| **Recommended phase** | Synthetic validation + operator training |
| **Related investigations** | INV-003, INV-012 |
| **Notes** | |

---

### INV-012 — Allocated shocks are computational bridges only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-012 |
| **Title** | Allocated national shocks must declare `allocation_role=computational_bridge_only` |
| **Category** | estimand |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Experiment evidence docs |
| **Evidence sources** | [experiment_evidence.md](../02_concepts/experiment_evidence.md) |
| **Problem statement** | Allocated shocks enable replay objective terms but are not experimental DMA truth. |
| **Why it matters** | Mislabeling undermines trust in geo-level decisions. |
| **Risk type** | causal |
| **Production impact** | Included in weighted objective when allowed by config. |
| **Current behavior** | Schema + docs; prod gates on evidence quality. |
| **Desired end state** | Automated disclosure on calibration summary and model card. |
| **Blocking dependencies** | None |
| **Suggested validation** | Registry validation tests |
| **Suggested owner area** | Calibration evidence |
| **Recommended phase** | Ongoing |
| **Related investigations** | INV-011 |
| **Notes** | |

---

### INV-013 — Reproducibility self-certification is not independent evidence

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-013 |
| **Title** | `reproducibility_certification` without `reference_run_path` cannot satisfy strict readiness |
| **Category** | reproducibility |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Reproducibility certification doc |
| **Evidence sources** | [reproducibility_certification.md](../04_governance/reproducibility_certification.md), [production_readiness.md](../04_governance/production_readiness.md) |
| **Problem statement** | Self-cert sets `identical_output=null` and `reproducibility_evidence=false`. |
| **Why it matters** | Teams may enable extension without independent run comparison. |
| **Risk type** | reproducibility |
| **Production impact** | Strict readiness requires independent comparison when extension enabled. |
| **Current behavior** | Snapshot compare when `reference_run_path` provided. |
| **Desired end state** | CI stores reference runs; promotion workflow links reference run id. |
| **Blocking dependencies** | Artifact store policy |
| **Suggested validation** | VAL-010 when wired to worlds |
| **Suggested owner area** | `mmm/governance/reproducibility_certification.py` |
| **Recommended phase** | Governance operations |
| **Related investigations** | INV-002, INV-038 |
| **Notes** | |

---

### INV-014 — Nested train `decision_bundle` is research tier, not prod decide output

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-014 |
| **Title** | `extension_report.decision_bundle` must not substitute for `mmm decide` decision JSON |
| **Category** | artifact integrity |
| **Severity** | medium |
| **Status** | open |
| **First identified in** | Decision artifact contract |
| **Evidence sources** | [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), [artifact_schema.md](../04_governance/artifact_schema.md) |
| **Problem statement** | Training emits nested bundle at research/diagnostic tier; prod CLI decide emits decision-tier JSON. |
| **Why it matters** | Downstream systems may wire wrong artifact to budget workflows. |
| **Risk type** | operational |
| **Production impact** | Prod semantic validation applies to CLI decide outputs only. |
| **Current behavior** | `artifact_tier_disclosure` on extension report. |
| **Desired end state** | Integration guides enforce CLI decide path for prod. |
| **Blocking dependencies** | Customer integration docs |
| **Suggested validation** | Doc link check; artifact tier tests |
| **Suggested owner area** | Decision service, artifact schema |
| **Recommended phase** | Documentation / integrations |
| **Related investigations** | INV-004 |
| **Notes** | |

---

### INV-015 — Curve-local and unsafe optimizers blocked unless explicitly enabled

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-015 |
| **Title** | Legacy curve-only optimizers require `allow_unsafe_decision_apis` (non-prod default) |
| **Category** | optimization |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Decision vs research doc |
| **Evidence sources** | [decision_vs_research.md](../02_concepts/decision_vs_research.md), [prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) |
| **Problem statement** | Full-panel simulate path is canonical; curve extrapolation alone is not decision-grade. |
| **Why it matters** | Historical integrations may still call legacy APIs. |
| **Risk type** | financial decision |
| **Production impact** | Prod forbids unsafe APIs by default. |
| **Current behavior** | `optimize_budget_via_simulation` for prod; diagnostics for curves. |
| **Desired end state** | Deprecation timeline for legacy optimizers in integrations. |
| **Blocking dependencies** | Customer migration |
| **Suggested validation** | Prod policy regression tests |
| **Suggested owner area** | Planning / decision modules |
| **Recommended phase** | Post-v1 deprecation |
| **Related investigations** | INV-001, INV-003 |
| **Notes** | |

---

### INV-016 — World materializer renders constant panel, not generative DGP

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-016 |
| **Title** | `build_panel_dataframe` uses constant spend/KPI from truth, not stochastic DGP |
| **Category** | DGP/synthetic |
| **Severity** | high |
| **Status** | open |
| **First identified in** | Phase 2A materializer; world_materialization doc |
| **Evidence sources** | [world_materialization.md](../05_validation/world_materialization.md), `mmm/validation/synthetic/materializer.py` |
| **Problem statement** | Materialized panels do not inject noise, adstock dynamics, or lift from generative equations. |
| **Why it matters** | VAL recovery tests on bundles understate integration complexity until DGP library exists. |
| **Risk type** | scientific validity |
| **Production impact** | None directly — affects validation fidelity. |
| **Current behavior** | Constant columns from `media_truth` / `outcome_truth`; INV-005 in validator prevents β in derived files. |
| **Desired end state** | Phase 2 DGP renders panels from equations; materializer optionally defers to DGP output. |
| **Blocking dependencies** | Synthetic validation Phase 2 minimal DGP library |
| **Suggested validation** | VAL-001–003 on DGP-generated panels |
| **Suggested owner area** | `mmm/validation/synthetic/materializer.py` |
| **Recommended phase** | Synthetic validation Phase 2 |
| **Related investigations** | INV-008, INV-020 |
| **Notes** | Documented smoke limitation. |

---

### INV-017 — Validation registry thresholds remain TBD_v1

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-017 |
| **Title** | VAL-001–VAL-014 pass criteria unset pending pilot calibration |
| **Category** | DGP/synthetic |
| **Severity** | critical |
| **Status** | blocked |
| **First identified in** | Phase 0 validation registry freeze |
| **Evidence sources** | [validation_registry.md](../05_validation/validation_registry.md), [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) DR-04 |
| **Problem statement** | All numeric thresholds are placeholder `TBD_v1`; no signed threshold proposal. |
| **Why it matters** | Cannot run ReliabilityScorecard or release gates on synthetic program without calibrated bounds. |
| **Risk type** | operational |
| **Production impact** | Indirect — blocks synthetic program maturity. |
| **Current behavior** | Registry defines metrics and dependencies only. |
| **Desired end state** | Versioned threshold YAML per validation_id (and optionally stratum). |
| **Blocking dependencies** | DR-04 ownership; Phase 2 pilot worlds (~100) |
| **Suggested validation** | Pilot scorecard percentiles documented in registry |
| **Suggested owner area** | Validation program lead |
| **Recommended phase** | Synthetic validation Phase 2–5 |
| **Related investigations** | INV-027, INV-034 |
| **Notes** | Explicit freeze per ADR — not a code bug. |

---

### INV-018 — World validator Level 4 (certification compatibility) not implemented

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-018 |
| **Title** | L4 checks (`VAL-*` vs `artifact_truth`) specified but not coded |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | postponed |
| **First identified in** | world_validator_spec Phase 1B |
| **Evidence sources** | [world_validator_spec.md](../05_validation/world_validator_spec.md), `mmm/validation/synthetic/validator.py` |
| **Problem statement** | Bundles pass L1–L3 but cannot assert expected cert levels vs actual reports. |
| **Why it matters** | Negative worlds and cert regression testing incomplete. |
| **Risk type** | reproducibility |
| **Production impact** | None until cert runner exists. |
| **Current behavior** | `validate_bundle(..., max_level=3)` only. |
| **Desired end state** | L4 validates `artifact_truth.expected_gates` / cert levels against runner output. |
| **Blocking dependencies** | Phase 4 certification framework |
| **Suggested validation** | L4 spec table L4-001–L4-008 |
| **Suggested owner area** | `mmm/validation/synthetic/validator.py` |
| **Recommended phase** | Synthetic validation Phase 4 |
| **Related investigations** | INV-008, INV-023, INV-025 |
| **Notes** | |

---

### INV-019 — No JSON Schema artifact for world_truth.json

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-019 |
| **Title** | `world_schema.md` is prose-only; no machine-readable JSON Schema |
| **Category** | documentation |
| **Severity** | low |
| **Status** | postponed |
| **First identified in** | world_schema Phase 1B deferred |
| **Evidence sources** | [world_schema.md](../05_validation/world_schema.md), [world_validator_spec.md](../05_validation/world_validator_spec.md) §9 |
| **Problem statement** | L1 validation is Python-coded, not schema-generated. |
| **Why it matters** | Cross-language consumers and CI schema diff harder. |
| **Risk type** | operational |
| **Production impact** | None |
| **Current behavior** | Python L1–L3 validator |
| **Desired end state** | Optional JSON Schema + codegen |
| **Blocking dependencies** | Phase 1B review completion |
| **Suggested validation** | Schema validates WORLD-001–004 bundles |
| **Suggested owner area** | Validation synthetic |
| **Recommended phase** | Post Phase 3A |
| **Related investigations** | INV-018 |
| **Notes** | |

---

### INV-020 — Minimal DGP archetype library not delivered (roadmap Phase 2)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-020 |
| **Title** | Seven archetype generators (baseline, adstock, saturation, etc.) not implemented |
| **Category** | DGP/synthetic |
| **Severity** | high |
| **Status** | postponed |
| **First identified in** | Synthetic validation roadmap Phase 2 |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) § Phase 2, [world_catalog.md](../05_validation/world_catalog.md) |
| **Problem statement** | Phase 3A delivers truth-only archetype templates; generative equations for panels not built. |
| **Why it matters** | P0 registry rows (coef/adstock/Hill recovery) lack world-backed runners. |
| **Risk type** | scientific validity |
| **Production impact** | Indirect — validation coverage gap vs prod stack. |
| **Current behavior** | `generators.py` deterministic truth; constant materializer panel. |
| **Desired end state** | Archetype modules emit truth + optional panel from equations. |
| **Blocking dependencies** | Phase 2 scope approval (seven archetypes only) |
| **Suggested validation** | Each archetype maps to ≥1 VAL row |
| **Suggested owner area** | `mmm/validation/synthetic/` |
| **Recommended phase** | Synthetic validation Phase 2 |
| **Related investigations** | INV-016, INV-008 |
| **Notes** | |

---

### INV-021 — ScenarioBuilder and lattice composition not implemented

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-021 |
| **Title** | No ScenarioBuilder for axis composition (signal, noise, correlation, drift, privacy) |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | revisit later |
| **First identified in** | Synthetic ADR 3; roadmap Phase 3 |
| **Evidence sources** | [scenario_builder.md](../05_validation/scenario_builder.md), [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) |
| **Problem statement** | ScenarioBuilder MVP (Phase 3B ✅) composes single specs; lattice sampling (dozens+ worlds) not yet implemented. |
| **Why it matters** | Reliability program Phase 5 needs strata generation policy. |
| **Risk type** | scalability |
| **Production impact** | None |
| **Current behavior** | `ScenarioSpec` + `WORLD-005`–`007`; no batch lattice runner. |
| **Desired end state** | Documented sampling policy + batch `world_id` factory for pilot n=100. |
| **Blocking dependencies** | Sampling policy ADR; Phase 5 infrastructure |
| **Suggested validation** | Reproducible `world_id` naming; seed contract |
| **Suggested owner area** | Validation synthetic |
| **Recommended phase** | Track 2 — post Phase 4A |
| **Related investigations** | INV-022, INV-027 |
| **Notes** | MVP closed Phase 3B; this INV tracks scale composition only. |

---

### INV-022 — Monte Carlo reliability program and ReliabilityScorecard not built

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-022 |
| **Title** | Phase 5 reliability program (n≈1k–10k worlds) not implemented |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | **partial** — pilot characterization; tier-1 batch open |
| **First identified in** | Synthetic validation roadmap Phase 5 |
| **Evidence sources** | [monte_carlo_reliability_program.md](../05_validation/monte_carlo_reliability_program.md), pilot JSON |
| **Problem statement** | Full N=100–10k batch not yet executed. |
| **Current behavior** | Tier-0 pilot + DR-06 resolved; tier-1 runner not built |
| **Desired end state** | Automated tier-1 sweep + percentile scorecards |
| **Blocking dependencies** | Phase 4 cert framework; DR-06; thresholds INV-017 |
| **Suggested validation** | Roadmap §8 scorecard dimensions |
| **Suggested owner area** | Validation platform |
| **Recommended phase** | Synthetic validation Phase 5 |
| **Related investigations** | INV-033, INV-034 |
| **Notes** | |

---

### INV-023 — Certification runner not wired to world bundles

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-023 |
| **Title** | Train/CI certification not executing VAL registry against `validation/worlds/*` |
| **Category** | DGP/synthetic |
| **Severity** | high |
| **Status** | postponed |
| **First identified in** | Roadmap Phase 4; world_materialization deferred |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md), [world_materialization.md](../05_validation/world_materialization.md) |
| **Problem statement** | Governance certs (synthetic, optimizer) run on built-in checks, not world catalog fixtures. |
| **Why it matters** | Duplication risk and coverage gaps vs declarative truth. |
| **Risk type** | reproducibility |
| **Production impact** | Readiness uses non-world certs today. |
| **Current behavior** | `materialize_world` + `validate_bundle` only for worlds. |
| **Desired end state** | Single runner: world → train/decide fixtures → cert reports → L4 validator. |
| **Blocking dependencies** | Phase 4; fixture templates |
| **Suggested validation** | WORLD-001 passes VAL rows when runner exists |
| **Suggested owner area** | Governance + validation synthetic |
| **Recommended phase** | Synthetic validation Phase 4 |
| **Related investigations** | INV-008, INV-018 |
| **Notes** | |

---

### INV-024 — Documentation drift: replay holdout split described inconsistently

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-024 |
| **Title** | `calibration.md` states holdout split “not implemented” but code and `statistical_validation.md` document it |
| **Category** | documentation |
| **Severity** | medium |
| **Status** | open |
| **First identified in** | Documentation truth audit follow-up |
| **Evidence sources** | [calibration.md](../02_concepts/calibration.md) § not implemented, [statistical_validation.md](../02_concepts/statistical_validation.md), `mmm/config/schema.py`, `tests/test_replay_holdout_split.py` |
| **Problem statement** | Operators may believe holdout split unavailable when `use_replay_holdout_split` exists. |
| **Why it matters** | Misconfiguration of replay overfit controls. |
| **Risk type** | operational |
| **Production impact** | Feature exists but under-documented in primary calibration doc. |
| **Current behavior** | `use_replay_holdout_split` implemented; separate path fields may differ from doc list. |
| **Desired end state** | Single canonical calibration doc aligned with code. |
| **Blocking dependencies** | Doc PR only (no product change requested here) |
| **Suggested validation** | `validate_docs.py`; test_replay_holdout_split |
| **Suggested owner area** | Documentation |
| **Recommended phase** | Doc maintenance |
| **Related investigations** | INV-007, INV-043 |
| **Notes** | Evidence of doc lag, not feature absence. |

---

### INV-025 — Negative world catalog representation unresolved (DR-03)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-025 |
| **Title** | DR-03 open: structure of `artifact_truth.expected_failures` for negative worlds |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | blocked |
| **First identified in** | Synthetic architecture decisions design review |
| **Evidence sources** | [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) DR-03, [world_validator_spec.md](../05_validation/world_validator_spec.md) |
| **Problem statement** | No normative pattern for negative worlds (enumerated failures vs templates vs tag-only). |
| **Why it matters** | L4 and VAL-007/008/014 need negative fixtures. |
| **Risk type** | operational |
| **Production impact** | None until negative worlds published. |
| **Current behavior** | `negative_world` metadata flag exists; catalog examples only in prose. |
| **Desired end state** | DR-03 decision + example WORLD-00N-negative bundles. |
| **Blocking dependencies** | Architecture review |
| **Suggested validation** | L4-008 expected failures match PolicyError classes |
| **Suggested owner area** | Validation architecture |
| **Recommended phase** | Synthetic Phase 1B+ / Phase 4 |
| **Related investigations** | INV-018 |
| **Notes** | |

---

### INV-026 — Large world bundle storage and CI fetch policy deferred

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-026 |
| **Title** | Git vs manifest-hosted bundles for large panels unset |
| **Category** | operational |
| **Severity** | low |
| **Status** | postponed |
| **First identified in** | world_materialization Phase 1A |
| **Evidence sources** | [world_materialization.md](../05_validation/world_materialization.md), [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) |
| **Problem statement** | Policy for gitignored panels + manifest hash CI fetch not decided. |
| **Why it matters** | Phase 5 scale may bloat repository or CI. |
| **Risk type** | scalability |
| **Production impact** | CI cost and clone size. |
| **Current behavior** | Small smoke bundles committed in-repo. |
| **Desired end state** | Documented artifact hosting + cache keys on `materialization_version`. |
| **Blocking dependencies** | Infra decision |
| **Suggested validation** | CI job pulls bundle by manifest |
| **Suggested owner area** | DevOps / validation |
| **Recommended phase** | Synthetic Phase 2 infra |
| **Related investigations** | INV-038 |
| **Notes** | |

---

### INV-027 — Pilot threshold calibration (~100 worlds) not executed

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-027 |
| **Title** | Roadmap success criterion “TBD → versioned thresholds” awaiting pilot |
| **Category** | DGP/synthetic |
| **Severity** | high |
| **Status** | blocked |
| **First identified in** | Roadmap §12 success criteria |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) §12, [validation_registry.md](../05_validation/validation_registry.md) |
| **Problem statement** | Threshold process defined in prose only; no pilot scorecard run. |
| **Why it matters** | Blocks honest synthetic release gates. |
| **Risk type** | operational |
| **Production impact** | Indirect |
| **Current behavior** | `TBD_v1` placeholders |
| **Desired end state** | Signed threshold doc per VAL id |
| **Blocking dependencies** | INV-020, INV-021; DR-04 |
| **Suggested validation** | Pilot report with percentiles |
| **Suggested owner area** | Validation program |
| **Recommended phase** | Phase 2–5 |
| **Related investigations** | INV-017, INV-034 |
| **Notes** | |

---

### INV-028 — Fingerprint mismatch and unsafe API waivers depend on signed JSON discipline

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-028 |
| **Title** | Governance waivers (`decision_fingerprint_mismatch`, `override_unsafe`, replay refit) require operational controls |
| **Category** | governance |
| **Severity** | high |
| **Status** | open |
| **First identified in** | v1 release notes; prod safety checklist |
| **Evidence sources** | [v1_release_notes.md](../04_governance/v1_release_notes.md), [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), [prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) |
| **Problem statement** | Technical waiver hooks exist; org process for signing, expiry, and audit is outside repo. |
| **Why it matters** | Waivers undermine fail-closed posture if treated as routine. |
| **Risk type** | operational |
| **Production impact** | Prod can proceed with severe warnings when waivers configured. |
| **Current behavior** | Signed JSON paths validated in code tests. |
| **Desired end state** | Waiver registry linked to decision trace and promotion records. |
| **Blocking dependencies** | Operational governance |
| **Suggested validation** | Periodic waiver inventory review |
| **Suggested owner area** | Governance + security/compliance |
| **Recommended phase** | Enterprise operations |
| **Related investigations** | INV-004, INV-005, INV-009 |
| **Notes** | |

---

### INV-029 — Runtime vs CI certification split undefined (DR-05)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-029 |
| **Title** | DR-05 open: which certifications run on every train vs CI-only batch |
| **Category** | architecture |
| **Severity** | medium |
| **Status** | blocked |
| **First identified in** | Synthetic architecture decisions |
| **Evidence sources** | [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) DR-05, [synthetic_certification.md](../04_governance/synthetic_certification.md) |
| **Problem statement** | v1 runs synthetic cert on train extension report and CI nightly; future split may duplicate truth if not designed. |
| **Why it matters** | Train latency vs coverage tradeoff unresolved. |
| **Risk type** | operational |
| **Production impact** | Train time and report size. |
| **Current behavior** | Synthetic cert on train; nightly workflow for extended suites. |
| **Desired end state** | ADR defining runtime minimal vs CI full cert sets. |
| **Blocking dependencies** | DR-05 decision |
| **Suggested validation** | Performance cert scenarios for train overhead |
| **Suggested owner area** | Architecture |
| **Recommended phase** | Synthetic Phase 4 |
| **Related investigations** | INV-023, INV-032 |
| **Notes** | |

---

### INV-030 — ReliabilityScorecard release role undefined (DR-06)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-030 |
| **Title** | DR-06 open: advisory scorecard vs blocking semver gate |
| **Category** | governance |
| **Severity** | medium |
| **Status** | blocked |
| **First identified in** | Synthetic architecture decisions |
| **Evidence sources** | [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) DR-06, [truth_versioning.md](../05_validation/truth_versioning.md) |
| **Problem statement** | Minimum `n_worlds`, regression policy when unit tests pass but scorecard degrades — unset. |
| **Why it matters** | Defines how synthetic program gates releases. |
| **Risk type** | operational |
| **Production impact** | Future release policy |
| **Current behavior** | No scorecard |
| **Desired end state** | DR-06 decision documented in governance |
| **Blocking dependencies** | INV-022; DR-06 |
| **Suggested validation** | Dry-run scorecard on WORLD-001–004 |
| **Suggested owner area** | Release management |
| **Recommended phase** | Synthetic Phase 5 |
| **Related investigations** | INV-022, INV-027 |
| **Notes** | |

---

### INV-031 — Threshold ownership process unset (DR-04)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-031 |
| **Title** | DR-04 open: who may replace `TBD_v1` (global vs per-archetype thresholds) |
| **Category** | governance |
| **Severity** | medium |
| **Status** | partial — DR-04 **draft resolution** (Phase 5D); formal sign-off pending |
| **First identified in** | Synthetic architecture decisions |
| **Evidence sources** | [reliability_threshold_governance.md](../05_validation/reliability_threshold_governance.md), DR-04 draft in [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md) |
| **Problem statement** | Numeric `TBD_v1` promotion process was undefined; metric classes now defined. |
| **Why it matters** | Blocks INV-017 closure until thresholds are **approved**. |
| **Risk type** | operational |
| **Production impact** | Indirect |
| **Current behavior** | `TBD_v1_runtime` provisional; governance doc defines promotion rules |
| **Desired end state** | Signed committee approval for first `approved` threshold set |
| **Blocking dependencies** | Organizational decision |
| **Suggested validation** | Threshold change ADR template |
| **Suggested owner area** | Governance committee |
| **Recommended phase** | Before Phase 5 |
| **Related investigations** | INV-017, INV-027 |
| **Notes** | |

---

### INV-032 — Bayesian production budget decisioning blocked

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-032 |
| **Title** | Prod `mmm decide optimize-budget` blocked for Bayesian framework |
| **Category** | estimand |
| **Severity** | informational |
| **Status** | accepted limitation |
| **First identified in** | v1.0.0 scope |
| **Evidence sources** | [v1_release_notes.md](../04_governance/v1_release_notes.md), [bayesian.md](../02_concepts/bayesian.md), [documentation_truth_audit.md](../documentation_truth_audit.md) |
| **Problem statement** | v1 prod canonical path is Ridge BO + semi_log + geometric adstock + Hill. |
| **Why it matters** | Sets customer expectations for prod vs research. |
| **Risk type** | operational |
| **Production impact** | PolicyError on prod Bayesian decide. |
| **Current behavior** | `posterior_planning_gate` blocks prod; research paths available. |
| **Desired end state** | If ever enabled, separate ADR + full validation program. |
| **Blocking dependencies** | Product scope expansion |
| **Suggested validation** | Prod policy regression tests |
| **Suggested owner area** | Decision service, Bayesian trainer |
| **Recommended phase** | Post-v1 modeling |
| **Related investigations** | INV-033, INV-034, INV-035 |
| **Notes** | Intentional v1 boundary. |

---

### INV-033 — Bayesian experiment likelihood is research-only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-033 |
| **Title** | `bayesian.use_experiment_likelihood` cannot enable prod decisioning |
| **Category** | calibration |
| **Severity** | informational |
| **Status** | research |
| **First identified in** | Bayesian docs PR 3 |
| **Evidence sources** | [bayesian.md](../02_concepts/bayesian.md), [experiment_evidence.md](../02_concepts/experiment_evidence.md) |
| **Problem statement** | Experiment likelihood term is PyMC research path; `prod_decisioning_allowed: false`. |
| **Why it matters** | Distinct from Ridge weighted replay prod path. |
| **Risk type** | scientific validity |
| **Production impact** | None on prod decide |
| **Current behavior** | `exp_likelihood_research_only` must stay true. |
| **Desired end state** | Clear reporting tier on extension artifacts. |
| **Blocking dependencies** | N/A for v1 |
| **Suggested validation** | Artifact tier disclosure tests |
| **Suggested owner area** | Bayesian calibration |
| **Recommended phase** | Research track |
| **Related investigations** | INV-032, INV-003 |
| **Notes** | Research opportunity, not defect. |

---

### INV-034 — Bayesian hierarchy partial pooling is research-only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-034 |
| **Title** | `bayesian.use_hierarchy` does not unlock prod budget optimization |
| **Category** | statistical validity |
| **Severity** | informational |
| **Status** | research |
| **First identified in** | Bayesian hierarchy PR 4B docs |
| **Evidence sources** | [bayesian.md](../02_concepts/bayesian.md), [hierarchical_borrowing.md](../02_concepts/hierarchical_borrowing.md) |
| **Problem statement** | Hierarchy extension reports shrinkage diagnostics but `prod_decisioning_allowed: false`. |
| **Why it matters** | Prevents misusing hierarchy as prod gate pass. |
| **Risk type** | scientific validity |
| **Production impact** | None on prod decide |
| **Current behavior** | Research extension only. |
| **Desired end state** | If promoted, requires VAL coverage on geo_world archetype. |
| **Blocking dependencies** | Synthetic geo_world; product ADR |
| **Suggested validation** | `bayesian_hierarchy_report` governance fields |
| **Suggested owner area** | Bayesian modeling |
| **Recommended phase** | Research track |
| **Related investigations** | INV-032, INV-020, INV-056, INV-063–069, [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md) |
| **Notes** | Bayes-H1–H5 track supersedes ad hoc hierarchy promotion; prod remains blocked (INV-032). |

---

### INV-035 — Ridge monetary confidence intervals not production-grade

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-035 |
| **Title** | Bootstrap/conformal Ridge intervals blocked for monetary decisioning |
| **Category** | uncertainty |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Ridge uncertainty research doc |
| **Evidence sources** | [ridge_uncertainty_research.md](../04_governance/ridge_uncertainty_research.md), [uncertainty_propagation.md](../02_concepts/uncertainty_propagation.md), [documentation_truth_audit.md](../documentation_truth_audit.md) |
| **Problem statement** | `decision_uncertainty.uncertainty_available` stays false for Ridge prod; conformal not implemented. |
| **Why it matters** | Finance teams may expect CIs on Δμ in prod. |
| **Risk type** | financial decision |
| **Production impact** | Point-estimate decisions with disclosure. |
| **Current behavior** | Research extension `ridge_uncertainty_research`; proxy labels in robust optimization research. |
| **Desired end state** | Either validated interval path or permanent disclosure standard. |
| **Blocking dependencies** | Research methodology review |
| **Suggested validation** | Coverage on synthetic panels; adstock exchangeability analysis |
| **Suggested owner area** | `mmm/governance/ridge_uncertainty_research.py` |
| **Recommended phase** | Research / post-v1 |
| **Related investigations** | INV-036, INV-037 |
| **Notes** | |

---

### INV-036 — Conformal intervals not implemented for Ridge

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-036 |
| **Title** | `ridge_summarize_conformal` reserved; returns `not_implemented` |
| **Category** | uncertainty |
| **Severity** | low |
| **Status** | research |
| **First identified in** | uncertainty_propagation doc |
| **Evidence sources** | [uncertainty_propagation.md](../02_concepts/uncertainty_propagation.md) |
| **Problem statement** | Conformal path documented as future; enabling flag does not produce decision intervals. |
| **Why it matters** | Avoids false confidence from partial implementation. |
| **Risk type** | scientific validity |
| **Production impact** | None |
| **Current behavior** | Explicit not_implemented status in report. |
| **Desired end state** | Implement or remove flag from public config docs. |
| **Blocking dependencies** | Research design |
| **Suggested validation** | Unit tests on propagation report status |
| **Suggested owner area** | Uncertainty extensions |
| **Recommended phase** | Research |
| **Related investigations** | INV-035 |
| **Notes** | |

---

### INV-037 — Robust optimization extension is research-only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-037 |
| **Title** | `robust_optimization_research` must not wire to prod budget systems |
| **Category** | optimization |
| **Severity** | informational |
| **Status** | research |
| **First identified in** | Robust optimization research doc |
| **Evidence sources** | [robust_optimization_research.md](../02_concepts/robust_optimization_research.md), [v1_release_notes.md](../04_governance/v1_release_notes.md) |
| **Problem statement** | Compares candidates using point Δμ and uncertainty proxies, not prod optimizer. |
| **Why it matters** | Prevents accidental prod allocator substitution. |
| **Risk type** | financial decision |
| **Production impact** | `prod_decisioning_allowed: false` |
| **Current behavior** | Extension off by default. |
| **Desired end state** | Clear separation from `optimize_budget_via_simulation`. |
| **Blocking dependencies** | N/A v1 |
| **Suggested validation** | Extension report guardrail fields |
| **Suggested owner area** | Research extensions |
| **Recommended phase** | Research |
| **Related investigations** | INV-001, INV-035 |
| **Notes** | |

---

### INV-038 — MLflow remote artifact backend experimental

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-038 |
| **Title** | Remote MLflow tracking/artifacts not contract-tested for prod |
| **Category** | operational |
| **Severity** | low |
| **Status** | accepted limitation |
| **First identified in** | Documentation truth audit |
| **Evidence sources** | [documentation_truth_audit.md](../documentation_truth_audit.md), [dev_setup.md](../dev_setup.md) |
| **Problem statement** | Local artifact backend is default; MLflow file-store experimental in CI only. |
| **Why it matters** | Enterprise deployments need supported artifact lineage story. |
| **Risk type** | operational |
| **Production impact** | Operators use LOCAL backend unless they accept experimental risk. |
| **Current behavior** | `ArtifactBackend.LOCAL` default; MLflow deprecation warnings in deps. |
| **Desired end state** | Supported remote backend with roundtrip contract tests. |
| **Blocking dependencies** | Infra + MLflow migration guidance |
| **Suggested validation** | Artifact lifecycle E2E on chosen backend |
| **Suggested owner area** | `mmm/artifacts/` |
| **Recommended phase** | Platform operations |
| **Related investigations** | INV-013, INV-026 |
| **Notes** | |

---

### INV-039 — Auto-retrain, auto-promotion, agentic orchestration out of v1 scope

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-039 |
| **Title** | No automatic retraining, promotion, or agentic orchestration in v1.0.0 |
| **Category** | operational |
| **Severity** | informational |
| **Status** | accepted limitation |
| **First identified in** | v1 release notes; CHANGELOG; synthetic roadmap non-goals |
| **Evidence sources** | [v1_release_notes.md](../04_governance/v1_release_notes.md), [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) §11, `CHANGELOG.md` |
| **Problem statement** | Drift detection and freshness warn; they do not trigger retrain or budget changes. |
| **Why it matters** | Enterprise MMM platforms often expect MLOps loops. |
| **Risk type** | operational |
| **Production impact** | Manual operator workflows only. |
| **Current behavior** | `calibration_freshness`, drift extensions, promotion workflow explicit. |
| **Desired end state** | If built, separate product ADR outside validation framework. |
| **Blocking dependencies** | Product scope |
| **Suggested validation** | N/A |
| **Suggested owner area** | Product / platform |
| **Recommended phase** | Post-v1 product |
| **Related investigations** | INV-041 |
| **Notes** | |

---

### INV-040 — State-space and time-varying coefficients not production-supported

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-040 |
| **Title** | State-space MMM and dynamic coefficients are roadmap/research, not prod contract |
| **Category** | statistical validity |
| **Severity** | informational |
| **Status** | postponed |
| **First identified in** | Synthetic roadmap non-goals; v1 release notes |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) §11, [v1_release_notes.md](../04_governance/v1_release_notes.md) |
| **Problem statement** | No production contract to validate time-varying β or state-space dynamics. |
| **Why it matters** | Common market request; scope creep risk if conflated with v1 validation. |
| **Risk type** | scientific validity |
| **Production impact** | None in v1 prod stack. |
| **Current behavior** | Static coefficients per train; drift warnings only. |
| **Desired end state** | Explicit ADR before any prod support. |
| **Blocking dependencies** | Modeling program |
| **Suggested validation** | New VAL rows + archetype if added |
| **Suggested owner area** | Modeling research |
| **Recommended phase** | Post-v1 modeling |
| **Related investigations** | INV-041, INV-040 |
| **Notes** | Post-v1 roadmap candidate. |

---

### INV-041 — Coefficient shift handling limited to drift detection and freshness warnings

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-041 |
| **Title** | No adaptive coefficient mechanism; drift/freshness advisory only |
| **Category** | drift |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Calibration freshness; v1 scope |
| **Evidence sources** | [calibration_freshness.md](../04_governance/calibration_freshness.md), [v1_release_notes.md](../04_governance/v1_release_notes.md), [documentation_truth_audit.md](../documentation_truth_audit.md) |
| **Problem statement** | `coefficient_shift_score` compares to accepted run; does not auto-recalibrate or retrain. |
| **Why it matters** | Stale models may remain “planning_allowed” until operator acts. |
| **Risk type** | operational |
| **Production impact** | Warnings in readiness and freshness reports. |
| **Current behavior** | Drift historical extension + run registry compare. |
| **Desired end state** | Operator playbooks; optional future auto-retrain (INV-039). |
| **Blocking dependencies** | INV-039 product scope |
| **Suggested validation** | VAL-008 when worlds include drift_truth |
| **Suggested owner area** | Governance / evaluation |
| **Recommended phase** | Operations + synthetic VAL-008 |
| **Related investigations** | INV-039, INV-007 |
| **Notes** | |

---

### INV-042 — Collinear channels: separability diagnostic does not fix attribution

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-042 |
| **Title** | High correlation / separability flags warn but do not stabilize split attribution |
| **Category** | statistical validity |
| **Severity** | high |
| **Status** | open |
| **First identified in** | feature_separability; synthetic DGP tests |
| **Evidence sources** | [feature_separability.md](../02_concepts/feature_separability.md), [synthetic_certification.md](../04_governance/synthetic_certification.md), `tests/test_synthetic_certification_exact.py` |
| **Problem statement** | Collinear channel tests assert warnings, not stable β splits. |
| **Why it matters** | Core MMM pain point for world-class attribution credibility. |
| **Risk type** | scientific validity |
| **Production impact** | Optimization and Δμ may proceed with separability warnings. |
| **Current behavior** | Identifiability + separability extensions; experiment scheduler recommends tests. |
| **Desired end state** | GroundTruthWorld correlation axes + VAL measurable false attach rate. |
| **Blocking dependencies** | ScenarioBuilder; VAL thresholds |
| **Suggested validation** | VAL on collinearity worlds; experiment_scheduler integration |
| **Suggested owner area** | Extensions + validation |
| **Recommended phase** | Synthetic Phase 3+ / research |
| **Related investigations** | INV-009, INV-020 |
| **Notes** | |

---

### INV-043 — `log_log` and non-canonical transforms blocked in prod

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-043 |
| **Title** | Prod path requires semi_log + geometric adstock + Hill; other transforms blocked |
| **Category** | configuration |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | v1 prod canonical contract |
| **Evidence sources** | [config_yaml.md](../01_getting_started/config_yaml.md), [v1_release_notes.md](../04_governance/v1_release_notes.md), `CHECK_REGISTRY` transform_policy_consistency |
| **Problem statement** | Weibull/logistic/log adstock are registry stubs for prod train/decide/simulate. |
| **Why it matters** | Customers on alternate transforms cannot use prod gates without migration. |
| **Risk type** | operational |
| **Production impact** | Config validation fails prod paths. |
| **Current behavior** | `assert_prod_decision_not_log_log`; canonical transform enforcement. |
| **Desired end state** | Any new transform requires ADR + VAL + synthetic archetype. |
| **Blocking dependencies** | Synthetic roadmap non-goals until base stack characterized |
| **Suggested validation** | transform_policy_consistency check |
| **Suggested owner area** | Config schema, transforms |
| **Recommended phase** | Post-v1 modeling expansion |
| **Related investigations** | INV-008, INV-020 |
| **Notes** | |

---

### INV-044 — Performance certification uses synthetic panels only

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-044 |
| **Title** | `performance_certification` measures runtime on synthetic DGP panels, not client IO |
| **Category** | performance |
| **Severity** | low |
| **Status** | accepted limitation |
| **First identified in** | performance_certification doc |
| **Evidence sources** | [performance_certification.md](../04_governance/performance_certification.md) |
| **Problem statement** | Large/medium scenarios opt-in; does not profile production data pipelines. |
| **Why it matters** | Scale surprises may appear only in deployment. |
| **Risk type** | scalability |
| **Production impact** | Advisory extension only. |
| **Current behavior** | Optional extension off by default. |
| **Desired end state** | Representative prod-scale fixtures or published complexity bounds. |
| **Blocking dependencies** | Representative datasets (non-public) |
| **Suggested validation** | Benchmark jobs on anonymized panels |
| **Suggested owner area** | `mmm/evaluation/performance_audit.py` |
| **Recommended phase** | Platform performance |
| **Related investigations** | INV-016, INV-044 |
| **Notes** | |

---

### INV-045 — External benchmarks (Phase 6) must not set prod approval

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-045 |
| **Title** | Google/Robyn/PyMC examples are sanity checks only per ADR 5 |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | postponed |
| **First identified in** | Synthetic ADR 5 |
| **Evidence sources** | [synthetic_architecture_decisions.md](../05_validation/synthetic_architecture_decisions.md), [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) Phase 6 |
| **Problem statement** | External datasets must not wire to `approved_for_prod` or registry thresholds. |
| **Why it matters** | Prevents false equivalence between public examples and internal truth worlds. |
| **Risk type** | operational |
| **Production impact** | None until adapters exist. |
| **Current behavior** | Not implemented. |
| **Desired end state** | Thin adapters + documented skip reasons. |
| **Blocking dependencies** | Phase 6 |
| **Suggested validation** | CI job cannot promote models based on external pass |
| **Suggested owner area** | Validation platform |
| **Recommended phase** | Synthetic Phase 6 |
| **Related investigations** | INV-003, INV-045 |
| **Notes** | |

---

### INV-046 — Continuous validation requires manual accepted-run registries

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-046 |
| **Title** | Continuous validation compares predictions to experiments using curated registries |
| **Category** | calibration |
| **Severity** | low |
| **Status** | research |
| **First identified in** | continuous_validation doc |
| **Evidence sources** | [continuous_validation.md](../02_concepts/continuous_validation.md) |
| **Problem statement** | No remote SaaS; requires `accepted_runs` curation. |
| **Why it matters** | Limits automation of model-vs-experiment monitoring. |
| **Risk type** | operational |
| **Production impact** | Extension hook only. |
| **Current behavior** | Local registry JSON contract. |
| **Desired end state** | Optional integration with customer experiment warehouse. |
| **Blocking dependencies** | Product integrations |
| **Suggested validation** | Extension smoke tests |
| **Suggested owner area** | Evaluation extensions |
| **Recommended phase** | Research / integrations |
| **Related investigations** | INV-003, INV-039 |
| **Notes** | Research opportunity. |

---

### INV-047 — Decision validation extension is post-hoc research

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-047 |
| **Title** | `decision_validation` matches prior decisions to later experiments — not a prod gate |
| **Category** | governance |
| **Severity** | low |
| **Status** | research |
| **First identified in** | decision_validation doc |
| **Evidence sources** | [decision_validation.md](../02_concepts/decision_validation.md), [v1_release_notes.md](../04_governance/v1_release_notes.md) |
| **Problem statement** | Retrospective alignment check; does not block decide. |
| **Why it matters** | Useful for learning loops; not substitute for INV-003 causal evidence. |
| **Risk type** | scientific validity |
| **Production impact** | None |
| **Current behavior** | Research extension. |
| **Desired end state** | Clear tier labeling on reports. |
| **Blocking dependencies** | N/A |
| **Suggested validation** | Doc + extension tier tests |
| **Suggested owner area** | Evaluation |
| **Recommended phase** | Research |
| **Related investigations** | INV-003, INV-046 |
| **Notes** | |

---

### INV-048 — Multi-world planning scenarios not implemented

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-048 |
| **Title** | `world_assumption=multi_world` reserved; prod bundles fail validation |
| **Category** | architecture |
| **Severity** | low |
| **Status** | postponed |
| **First identified in** | artifact_schema; planning_execution |
| **Evidence sources** | [artifact_schema.md](../04_governance/artifact_schema.md), [planning_execution.md](../03_planning/planning_execution.md) |
| **Problem statement** | Cannot optimize across weighted scenario worlds in prod contract. |
| **Why it matters** | Limits advanced planning workflows. |
| **Risk type** | usability |
| **Production impact** | Validation error if used. |
| **Current behavior** | `historical_panel` and `explicit_scenario` only. |
| **Desired end state** | Spec + VAL for multi-world if product approves. |
| **Blocking dependencies** | Product ADR |
| **Suggested validation** | Semantic contract tests |
| **Suggested owner area** | Planning assumptions |
| **Recommended phase** | Post-v1 planning |
| **Related investigations** | INV-014 |
| **Notes** | |

---

### INV-049 — Stan backend stub; PyMC is default Bayesian path

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-049 |
| **Title** | `StanMMMTrainer` stub — packaged `.stan` models not shipped |
| **Category** | technical debt |
| **Severity** | low |
| **Status** | research |
| **First identified in** | bayesian.md |
| **Evidence sources** | [bayesian.md](../02_concepts/bayesian.md) |
| **Problem statement** | Stan path documented as optional stub. |
| **Why it matters** | Teams expecting Stan must use PyMC or contribute models. |
| **Risk type** | operational |
| **Production impact** | None (Bayesian prod decide blocked anyway). |
| **Current behavior** | PyMC default trainer. |
| **Desired end state** | Implement or remove stub from public API surface. |
| **Blocking dependencies** | Modeling resourcing |
| **Suggested validation** | API docs vs imports |
| **Suggested owner area** | Bayesian trainers |
| **Recommended phase** | Research |
| **Related investigations** | INV-032 |
| **Notes** | |

---

### INV-050 — Production catalog index file for worlds not created

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-050 |
| **Title** | `validation/worlds/catalog.json` (or equivalent) not committed per bundle schema |
| **Category** | DGP/synthetic |
| **Severity** | low |
| **Status** | postponed |
| **First identified in** | world_bundle_schema Phase 1B |
| **Evidence sources** | [world_bundle_schema.md](../05_validation/world_bundle_schema.md), [world_catalog.md](../05_validation/world_catalog.md) |
| **Problem statement** | Worlds exist as directories; machine-readable catalog index deferred. |
| **Why it matters** | CI and scorecard need discoverable world list. |
| **Risk type** | operational |
| **Production impact** | None |
| **Current behavior** | Four smoke worlds in repo paths documented in README. |
| **Desired end state** | Versioned catalog index with tags and dependencies. |
| **Blocking dependencies** | Phase 1B catalog implementation |
| **Suggested validation** | Validator reads catalog for required worlds |
| **Suggested owner area** | Validation synthetic |
| **Recommended phase** | Synthetic Phase 2 infra |
| **Related investigations** | INV-026 |
| **Notes** | |

---

### INV-051 — Geo-rank CV not supported in prod

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-051 |
| **Title** | Geo-rank CV mode documented as unsupported for prod |
| **Category** | statistical validity |
| **Severity** | medium |
| **Status** | accepted limitation |
| **First identified in** | Documentation truth audit |
| **Evidence sources** | [documentation_truth_audit.md](../documentation_truth_audit.md), [cv_modes.md](../02_concepts/cv_modes.md) |
| **Problem statement** | Prod requires explicit calendar CV strategy; `cv.mode=auto` forbidden. |
| **Why it matters** | Wrong CV choice invalidates holdout interpretation. |
| **Risk type** | scientific validity |
| **Production impact** | Config validation on prod. |
| **Current behavior** | Calendar rolling/expanding documented for prod path. |
| **Desired end state** | If geo-rank added, requires VAL and runbook update. |
| **Blocking dependencies** | Product decision |
| **Suggested validation** | Prod config regression |
| **Suggested owner area** | CV / training |
| **Recommended phase** | Post-v1 |
| **Related investigations** | INV-005, INV-007 |
| **Notes** | |

---

### INV-052 — Dynamic priors explicitly out of synthetic validation scope

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-052 |
| **Title** | Dynamic priors listed as modeling expansion, not current validation target |
| **Category** | research |
| **Severity** | informational |
| **Status** | postponed |
| **First identified in** | Synthetic roadmap non-goals |
| **Evidence sources** | [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md) §11 |
| **Problem statement** | No contract for time-varying priors in validation framework. |
| **Why it matters** | Prevents scope creep in Phase 2 DGP. |
| **Risk type** | scientific validity |
| **Production impact** | None |
| **Current behavior** | Static priors in Bayesian research path. |
| **Desired end state** | ADR if product adopts dynamic priors. |
| **Blocking dependencies** | Modeling program |
| **Suggested validation** | N/A until scoped |
| **Suggested owner area** | Modeling research |
| **Recommended phase** | Post-v1 modeling |
| **Related investigations** | INV-040 |
| **Notes** | Post-v1 roadmap candidate. |

---

### INV-053 — L3 replay and optimizer numeric tolerances still TBD_v1

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-053 |
| **Title** | Validator L3-004 budget tolerance and WARN-* bounds unset |
| **Category** | DGP/synthetic |
| **Severity** | medium |
| **Status** | blocked |
| **First identified in** | world_validator_spec |
| **Evidence sources** | [world_validator_spec.md](../05_validation/world_validator_spec.md), [validation_registry.md](../05_validation/validation_registry.md) |
| **Problem statement** | Cross-object numeric checks reference `TBD_v1` tolerances. |
| **Why it matters** | L3 cannot gate certification-grade bundles on optimizer truth yet. |
| **Risk type** | reproducibility |
| **Production impact** | None today (L3 structural checks only). |
| **Current behavior** | L3-004 not enforced with numeric tolerance in code (spec only). |
| **Desired end state** | Tolerances copied from signed threshold doc. |
| **Blocking dependencies** | INV-017, INV-031 |
| **Suggested validation** | Unit tests with golden worlds |
| **Suggested owner area** | `mmm/validation/synthetic/validator.py` |
| **Recommended phase** | After threshold pilot |
| **Related investigations** | INV-017, INV-018 |
| **Notes** | |

---

### INV-054 — Cross-branch features quarantined in archive docs

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-054 |
| **Title** | Some `02_concepts` topics may be documented but not shipped on all branches |
| **Category** | documentation |
| **Severity** | low |
| **Status** | open |
| **First identified in** | cross_branch_not_shipped; documentation_truth_audit |
| **Evidence sources** | [_archive/cross_branch_not_shipped.md](../_archive/cross_branch_not_shipped.md), [documentation_truth_audit.md](../documentation_truth_audit.md) |
| **Problem statement** | Experiment scheduler / separability docs may exceed merge state on some branches. |
| **Why it matters** | Doc readers may assume features exist in their checkout. |
| **Risk type** | usability |
| **Production impact** | Depends on branch/deployment. |
| **Current behavior** | Inventory + quarantine doc; validate_docs in CI. |
| **Desired end state** | Inventory flags experimental pages with branch gates. |
| **Blocking dependencies** | Release branch discipline |
| **Suggested validation** | `validate_docs.py`; inventory review each release |
| **Suggested owner area** | Documentation |
| **Recommended phase** | Each release |
| **Related investigations** | INV-019 |
| **Notes** | |

---

### INV-055 — Archived causal/calibration roadmap is not current product spec

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-055 |
| **Title** | `_archive/roadmap_causal_calibration_governance.md` must not be used as operator guidance |
| **Category** | documentation |
| **Severity** | low |
| **Status** | accepted limitation |
| **First identified in** | Documentation inventory |
| **Evidence sources** | [_archive/roadmap_causal_calibration_governance.md](../_archive/roadmap_causal_calibration_governance.md), [DOCUMENTATION_INVENTORY.md](../DOCUMENTATION_INVENTORY.md) |
| **Problem statement** | Historical roadmap may contradict v1 governance docs. |
| **Why it matters** | Prevents resurrecting superseded gate designs. |
| **Risk type** | operational |
| **Production impact** | None if operators use canonical docs. |
| **Current behavior** | Archived under `docs/_archive/`. |
| **Desired end state** | Banner + inventory status `archived`. |
| **Blocking dependencies** | None |
| **Suggested validation** | Link check excludes archive from operator paths |
| **Suggested owner area** | Documentation |
| **Recommended phase** | Ongoing |
| **Related investigations** | INV-054 |
| **Notes** | |

---

### INV-055 — Dedicated VAL-012 drift_detection_runner (Phase 5E)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-055 |
| **Title** | Dedicated VAL-012 drift_detection_runner |
| **Status** | **closed** — Phase 5E |
| **Category** | certification reliability gaps |
| **Platform track** | 2 — Reliability & Validation |
| **Current behavior** | `mmm/validation/synthetic/drift_detection_runner.py` executes on WORLD-011 and L5B drift lattice cells; severity `none`/`minor`/`moderate`/`severe`; outcomes `pass`/`warning`/`severe`. |
| **Desired end state** | ✅ Delivered — see [drift_detection.md](../05_validation/drift_detection.md), [trust_report_semantics.md](../05_validation/trust_report_semantics.md) |
| **Related investigations** | INV-027, INV-057, INV-060 |
| **Notes** | Production `build_drift_report` alignment remains a separate track; synthetic VAL-012 validates trust semantics under known truth. |

---

### INV-056 — Exact recovery failure analysis (Phase 5C)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-056 |
| **Title** | Exact recovery failure analysis — WORLD-008 and behavioral lattice evidence |
| **Category** | certification reliability gaps |
| **Severity** | **critical** |
| **Status** | **closed** — explained |
| **Platform track** | 2 — Reliability & Validation Program |
| **First identified in** | Phase 4B-2 recovery certification; Phase 5B behavioral lattice sweep |
| **Evidence sources** | [exact_recovery_investigation.md](../05_validation/exact_recovery_investigation.md), [behavioral_lattice_sweep.md](../05_validation/behavioral_lattice_sweep.md), `validation/reports/behavioral_lattice_sweep_mvp_report.json`, WORLD-008–012 certification reports |
| **Problem statement** | Behavioral lattice shows **high structural reliability** (~0.89) but **materially lower behavioral recovery** (~0.57). WORLD-008 / L5B exact-recovery worlds fail coefficient and transform recovery (REC-4B2-001–003) despite same functional family as production Ridge BO + geometric adstock + Hill. Platform contracts remain stable. |
| **Why it matters** | Behavioral reliability is the primary limit on scientific confidence — not lack of modeling sophistication. Fixes and thresholds must follow explanation, not feature expansion. |
| **Risk type** | scientific validity |
| **Production impact** | None directly — blocks **major production modeling expansion** per platform roadmap gate until investigation completes |
| **Current behavior** | [exact_recovery_investigation_report.md](../05_validation/exact_recovery_investigation_report.md) published; failure taxonomy and root-cause ranking complete |
| **Desired end state** | ✅ Delivered — see Phase 5D [reliability_threshold_governance.md](../05_validation/reliability_threshold_governance.md) for threshold semantics |
| **Questions to answer** | Ridge shrinkage vs truth? Hyperparameter search vs truth? Identifiability? Decay/Hill compensating for coef error? CV objective vs recovery? Tolerances unrealistic? Truth-pinned transforms? Recovery vs noise/geos/periods/channels? Theoretical recovery ceiling for current architecture? |
| **Blocking dependencies** | None for investigation (docs/analysis); blocks Track 4 prod modeling and Bayes-H implementation |
| **Suggested validation** | Controlled sensitivity worlds; truth-pinned transform ablation; no new MMM methods in this phase |
| **Suggested owner area** | Modeling science, synthetic validation |
| **Recommended phase** | Phase 5C ✅ complete |
| **Related investigations** | INV-017, INV-027, INV-031, INV-055, INV-057–060, INV-063–069 |
| **Notes** | Closed 2026-05-29. Follow-ups: INV-057 (governance), INV-058 (compensation monitoring), INV-060 (Monte Carlo thresholds). |

---

### INV-057 — Decision vs attribution threshold separation

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-057 |
| **Title** | Decision-grade vs diagnostic attribution threshold separation |
| **Category** | certification reliability gaps |
| **Severity** | high |
| **Status** | **closed** — Phase 5D governance doc |
| **Platform track** | 2 — Reliability & Validation Program |
| **Evidence sources** | INV-056 report; [reliability_threshold_governance.md](../05_validation/reliability_threshold_governance.md); [validation_registry.md](../05_validation/validation_registry.md) §14 |
| **Problem statement** | Coef/transform failures were treated alongside Δμ in undifferentiated scorecard rollups, misstating Ridge BO decision reliability. |
| **Desired end state** | ✅ Metric classes, DR-04 draft, registry governance columns, scorecard v1.1 class scores |
| **Related investigations** | INV-056, INV-059, INV-031 (DR-04) |

---

### INV-058 — Transform/β compensation monitoring

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-058 |
| **Title** | Transform and coefficient compensation — ongoing synthetic monitoring |
| **Category** | certification reliability gaps |
| **Severity** | medium |
| **Status** | open |
| **Platform track** | 2 |
| **Evidence sources** | [investigations/hyperparameter_coupling.json](../05_validation/investigations/hyperparameter_coupling.json); exact recovery report |
| **Problem statement** | Alternative (decay, Hill) settings achieve similar RMSE with different β; flat objective regions persist. |
| **Desired end state** | Periodic lattice/Monte Carlo reports flagging compensation severity; TrustReport warning when grid flatness exceeds threshold |
| **Related investigations** | INV-056, INV-060 |

---

### INV-059 — ReliabilityScorecard metric-class refactor

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-059 |
| **Title** | ReliabilityScorecard metric-class refactor |
| **Category** | certification reliability gaps |
| **Severity** | medium |
| **Status** | **closed** — `reliability_scorecard_v1.1.0` |
| **Evidence sources** | `mmm/validation/synthetic/reliability_scorecard.py`; [reliability_scorecard.md](../05_validation/reliability_scorecard.md) |
| **Desired end state** | ✅ Separate `decision_reliability_score`, `attribution_diagnostic_score`, `structural_reliability_score`, `trust_modifier_status` |
| **Related investigations** | INV-057 |

---

### INV-060 — Monte Carlo threshold calibration

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-060 |
| **Title** | Monte Carlo threshold calibration (Phase 5F) |
| **Category** | certification reliability gaps |
| **Severity** | medium |
| **Status** | **partial** — tier-0 pilot complete; tier-1 approval open |
| **Platform track** | 2 |
| **Problem statement** | `TBD_v1_runtime` bounds are not statistically grounded at scale. |
| **Desired end state** | Versioned **approved** thresholds with confidence intervals from tier-1+ |
| **Current behavior** | [monte_carlo_threshold_recommendations.md](../05_validation/monte_carlo_threshold_recommendations.md) published; not approved |
| **Blocking dependencies** | Tier-1 N=100 batch; DR-04 sign-off |
| **Related investigations** | INV-022, INV-031, INV-057 |

---

### INV-063 — Bayesian hierarchical geo MMM design (Bayes-H1)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-063 |
| **Title** | Bayesian hierarchical geo-level MMM — hierarchy and estimand design |
| **Category** | contract compatibility |
| **Severity** | informational |
| **Status** | **closed** — Bayes-H1 + Bayes-H2 ADRs accepted |
| **Platform track** | 4 — Research Sandbox |
| **First identified in** | Bayesian hierarchical geo MMM roadmap (2026-05) |
| **Evidence sources** | [bayes_h1_decision_surface_preservation_adr.md](../05_validation/bayes_h1_decision_surface_preservation_adr.md), [bayes_h2_calibration_signal_mapping_adr.md](../05_validation/bayes_h2_calibration_signal_mapping_adr.md) |
| **Problem statement** | ~~Experiment-prior / CalibrationSignal mapping undefined~~ — resolved by Bayes-H2 ADR. |
| **Why it matters** | World materialization remains before sampler code. |
| **Risk type** | scientific validity |
| **Production impact** | None until Bayes-H5 |
| **Current behavior** | Bayes-H1–H2b ADRs **accepted**; no PyMC until `WORLD-BAYES-*` in catalog |
| **Desired end state** | `hierarchy_evidence_validator` + `VAL-BAYES-H2B-SMOKE` CI pass |
| **Blocking dependencies** | Fixtures ✅ under `validation/worlds/WORLD-BAYES-*/`; validator stub pending |
| **Suggested validation** | MA-* assertions per [validation worlds catalog](../BAYES_H2B_VALIDATION_WORLDS_001.md) |
| **Suggested owner area** | Synthetic validation, platform contracts |
| **Recommended phase** | Bayes-H2c worlds |
| **Related investigations** | INV-032, INV-034, INV-056, INV-069 |
| **Notes** | Does not authorize code changes to current Bayesian modules. |

---

### INV-064 — Partial pooling validation on hierarchical worlds (Bayes-H2/H4)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-064 |
| **Title** | Partial pooling and small-geo shrinkage — synthetic recovery criteria |
| **Category** | certification reliability gaps |
| **Severity** | medium |
| **Status** | open |
| **Platform track** | 4 — Research Sandbox / 2 — Reliability |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md), INV-034 |
| **Problem statement** | No hierarchical worlds with known \(\mu_c\), \(\tau_c\), and per-geo \(\beta_{g,c}\) for shrinkage certification. |
| **Why it matters** | Partial pooling is the core scientific claim of the track; unvalidated shrinkage misleads geo planning narratives. |
| **Risk type** | scientific validity |
| **Production impact** | None |
| **Current behavior** | WORLD-008–012 are national/DGP recovery worlds, not geo-pooling worlds. |
| **Desired end state** | `WORLD-BAYES-*` + geo pooling worlds + Bayes-H4 shrinkage checks. |
| **Blocking dependencies** | Materialize worlds per [bayes_h2b ADR](../05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md); INV-017 |
| **Suggested validation** | VAL extensions or REC-Bayes-* checks on geo_world bundles |
| **Suggested owner area** | Synthetic validation, Bayesian modeling |
| **Recommended phase** | Bayes-H2 → Bayes-H4 |
| **Related investigations** | INV-020, INV-034, INV-056, INV-063 |
| **Notes** | |

---

### INV-065 — Posterior calibration and coverage (Bayes-H4)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-065 |
| **Title** | Bayesian posterior calibration and coverage on synthetic hierarchical worlds |
| **Category** | statistical validity |
| **Severity** | medium |
| **Status** | open |
| **Platform track** | 4 — Research Sandbox |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md), [statistical_validation.md](../02_concepts/statistical_validation.md) |
| **Problem statement** | No reliability-program criteria for posterior interval calibration, PPC adequacy, or coverage vs injected truth. |
| **Why it matters** | TrustReport uncertainty is misleading if intervals are miscalibrated. |
| **Risk type** | scientific validity |
| **Production impact** | None |
| **Current behavior** | PyMC research path emits diagnostics; no world-based coverage gate. |
| **Desired end state** | Versioned coverage/PPC thresholds on Bayes-H2 worlds. |
| **Blocking dependencies** | Bayes-H3 sampling stable on H2 worlds |
| **Suggested validation** | PPC + nominal coverage checks per geo and global parameters |
| **Suggested owner area** | Bayesian modeling, evaluation |
| **Recommended phase** | Bayes-H4 |
| **Related investigations** | INV-035, INV-064, INV-067 |
| **Notes** | |

---

### INV-066 — Local experiment priors for hierarchical Bayesian MMM (Bayes-H1/H4)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-066 |
| **Title** | Local vs global experiment priors — scope and CalibrationSignal alignment |
| **Category** | calibration |
| **Severity** | medium |
| **Status** | **closed** — Bayes-H2b scope ADR accepted |
| **Platform track** | 4 — Research Sandbox |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](../05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md), [bayes_h2_calibration_signal_mapping_adr.md](../05_validation/bayes_h2_calibration_signal_mapping_adr.md) |
| **Problem statement** | ~~DMA → national propagation rules undefined~~ — resolved by Bayes-H2b ADR. |
| **Why it matters** | Certification on `WORLD-BAYES-*` worlds validates propagation behavior. |
| **Risk type** | scientific validity |
| **Production impact** | None (INV-033 research-only) |
| **Current behavior** | `bayesian.use_experiment_likelihood` research path; no geo-prior recovery worlds. |
| **Desired end state** | `WORLD-BAYES-*` certification + Bayes-H4 recovery. |
| **Blocking dependencies** | World materialization (INV-064); Bayes-H3 |
| **Suggested validation** | Compare to injected `experiment_truth` on hierarchical bundles |
| **Suggested owner area** | Bayesian calibration, replay |
| **Recommended phase** | Bayes-H1 → Bayes-H4 |
| **Related investigations** | INV-003, INV-011, INV-033 |
| **Notes** | |

---

### INV-067 — Bayesian compute scalability for geo hierarchical models (Bayes-H3)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-067 |
| **Title** | Bayesian hierarchical geo MMM — compute and CI/runtime policy |
| **Category** | performance |
| **Severity** | medium |
| **Status** | open |
| **Platform track** | 4 — Research Sandbox |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md), INV-049, INV-044 |
| **Problem statement** | Geo × channel hierarchical sampling may be too slow for default CI or operator workflows. |
| **Why it matters** | Unbounded runtime blocks reliability sweeps and research usability. |
| **Risk type** | operational |
| **Production impact** | None |
| **Current behavior** | PyMC default path; no geo-hierarchy scale benchmarks. |
| **Desired end state** | Documented complexity tiers, optional variational research path, CI job timeouts. |
| **Blocking dependencies** | Bayes-H3 prototype |
| **Suggested validation** | Performance cert on representative H2 world sizes |
| **Suggested owner area** | Bayesian trainers, platform infra |
| **Recommended phase** | Bayes-H3 |
| **Related investigations** | INV-044, INV-049 |
| **Notes** | |

---

### INV-068 — Bayesian TrustReport compatibility (Bayes-H1/H4)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-068 |
| **Title** | TrustReport fields for Bayesian hierarchical posterior outputs |
| **Category** | TrustReport consistency |
| **Severity** | medium |
| **Status** | open |
| **Platform track** | 4 — Research Sandbox |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md), [production_readiness.md](../04_governance/production_readiness.md) |
| **Problem statement** | Posterior shrinkage, convergence, and coverage must map to TrustReport tiers without inventing parallel gate names. |
| **Why it matters** | Operators and orchestration consume TrustReport — not raw PyMC traces. |
| **Risk type** | operational |
| **Production impact** | None until Bayes-H5 |
| **Current behavior** | Research extension artifacts; prod TrustReport is Ridge-centric. |
| **Desired end state** | Research-tier TrustReport-compatible rollup with `prod_decisioning_allowed: false`. |
| **Blocking dependencies** | Bayes-H1 TrustReport schema |
| **Suggested validation** | Artifact schema tests; readiness must not imply prod approval |
| **Suggested owner area** | Governance, Bayesian modeling |
| **Recommended phase** | Bayes-H1 → Bayes-H3 |
| **Related investigations** | INV-002, INV-032, INV-069 |
| **Notes** | |

---

### INV-069 — Bayesian DecisionSurface compatibility (Bayes-H4/H5)

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-069 |
| **Title** | DecisionSurface and full-panel Δμ compatibility for Bayesian hierarchical estimator |
| **Category** | decision-surface fragmentation risk |
| **Severity** | high |
| **Status** | open |
| **Platform track** | 4 — Research Sandbox / 1 — Contracts |
| **First identified in** | Bayesian hierarchical geo MMM roadmap |
| **Evidence sources** | [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md), [decision_artifact_contract.md](../04_governance/decision_artifact_contract.md), INV-032 |
| **Problem statement** | Future Bayesian estimator must produce full-panel Δμ and optimizer scores on the **same Estimand** as Ridge without alternate prod APIs. |
| **Why it matters** | Contract preservation rule: estimator may change; decision contract may not. |
| **Risk type** | decision-surface fragmentation |
| **Production impact** | None until Bayes-H5 + explicit ADR |
| **Current behavior** | Prod decide blocked for Bayesian (INV-032). |
| **Desired end state** | Bayes-H4 Δμ/optimizer recovery on hierarchical worlds; Bayes-H5 candidacy review only. |
| **Blocking dependencies** | Bayes-H3–H4; VAL-004/005 on geo worlds |
| **Suggested validation** | Same reliability checks as WORLD-008/009; Ridge side-by-side scorecard |
| **Suggested owner area** | Decision service, platform contracts |
| **Recommended phase** | Bayes-H4 → Bayes-H5 |
| **Related investigations** | INV-001, INV-008, INV-032, INV-056, INV-063 |
| **Notes** | **Non-goal:** no prod decisioning before this investigation closes at H5 bar. |

---

### INV-070 — Validation package import cycles block full pytest collection

| Field | Value |
|-------|-------|
| **Investigation ID** | INV-070 |
| **Title** | Resolve pre-existing `mmm.validation` import cycles blocking full pytest collection |
| **Category** | certification reliability gaps |
| **Severity** | medium |
| **Status** | resolved (2026-06-01) |
| **Platform track** | 2 — Reliability & Validation |
| **First identified in** | Bayes-H3 guardrails checkpoint (2026-06-01) |
| **Evidence sources** | `pytest tests/ -m "not slow"` → `ImportError` on `mmm.validation` ↔ `mmm.models.ridge_bo.trainer` cycle via `certification_runner` / `recovery_certification` |
| **Problem statement** | Full test collection failed because `mmm/validation/__init__.py` and `mmm/validation/synthetic/__init__.py` eagerly imported synthetic runners that pull `RidgeBOMMMTrainer` while `ridge_bo.trainer` was still initializing. |
| **Why it matters** | CI runs `pytest tests/ -m "not slow"`; collection errors hide regressions outside focused suites. |
| **Risk type** | certification reliability |
| **Production impact** | None — import hygiene only. |
| **Current behavior** | Lightweight package initializers; consumers import concrete modules directly (e.g. `mmm.validation.synthetic.hierarchy_evidence_validator`). `pytest tests/ --collect-only` and `pytest tests/ -m "not slow"` pass. |
| **Desired end state** | Lazy imports or split `mmm.validation` package surface so `pytest tests/` collects cleanly. |
| **Resolution** | Removed eager synthetic/hierarchy exports from `mmm/validation/__init__.py`; emptied `mmm/validation/synthetic/__init__.py`; updated tests using package-level synthetic submodule imports. |
| **Blocking dependencies** | None |
| **Suggested validation** | `pytest tests/ --collect-only -q` and `pytest tests/ -m "not slow"` |
| **Suggested owner area** | `mmm/validation/__init__.py`, `mmm/validation/synthetic/__init__.py` |
| **Recommended phase** | Closed |
| **Related investigations** | INV-023, INV-008 |
| **Notes** | **Fix:** commit `Fix validation package import cycles`. Do not re-export synthetic validators from package `__init__` files. |
