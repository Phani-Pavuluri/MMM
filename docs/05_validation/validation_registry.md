# Validation registry (frozen)

**Registry ID:** `mmm_validation_registry_v1`  
**Status:** Frozen (Phase 0) — canonical mapping from capabilities to truth, metrics, and ownership.  
**Threshold policy:** All numeric pass thresholds are `TBD_v1` until calibrated in Phase 2 pilot worlds.

**Related:** [groundtruth_contract.md](groundtruth_contract.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md)

---

## 1. Purpose

This registry is the **single catalog** of what the synthetic validation framework measures. Certifications, CI jobs, and ReliabilityScorecards must reference `validation_id` rows — not ad hoc assertion names.

**Pass criteria** in this document are **placeholders** (`TBD_v1`). Filling them requires a signed threshold proposal after pilot runs (see [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) § Design review).

---

## 2. Registry schema

Each row defines:

| Column | Meaning |
|--------|---------|
| `validation_id` | Stable ID (`VAL-XXX`) |
| `capability` | Human-readable capability name |
| `truth_object` | Dot-path into `GroundTruthWorld` (see groundtruth contract) |
| `metric` | Short metric name |
| `metric_definition` | Precise definition sufficient for independent implementation |
| `pass_criteria` | `TBD_v1` or logical condition not requiring numeric threshold |
| `severity` | `blocker` \| `major` \| `minor` — legacy impact label; see `metric_class` for release policy |
| `owner` | Team role accountable for threshold calibration |
| `dependencies` | Other `validation_id` or platform modules |

### 2.1 Governance columns (Phase 5D)

Added per [reliability_threshold_governance.md](reliability_threshold_governance.md) and INV-056. Summary table in [§14](#14-governance-classification-val-00114).

| Column | Values | Meaning |
|--------|--------|---------|
| `metric_class` | `decision_grade` \| `diagnostic_attribution` \| `trust_modifier` \| `structural` | How the metric affects readiness and TrustReport |
| `can_block_release` | `true` \| `false` \| `conditional` | Default release gate when threshold is `approved` |
| `threshold_status` | `TBD_v1_runtime` \| `approved` \| `research_only` | Lifecycle of numeric bounds |
| `required_evidence` | text | World families / lattice / Monte Carlo needed to promote thresholds |

**Provisional rule:** While `threshold_status` is `TBD_v1_runtime`, **no** row blocks production release by itself.

---

## 3. Severity definitions

| Severity | Meaning |
|----------|---------|
| **blocker** | Failures invalidate production certification claims for the affected path |
| **major** | Failures require review; may warn in readiness until threshold set |
| **minor** | Diagnostic; does not alone block `approved_for_prod` |

---

## 4. Canonical registry (v1)

### VAL-001 — Coefficient recovery

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-001` |
| `capability` | Coefficient recovery |
| `truth_object` | `media_truth.coefficients` |
| `metric` | `coef_recovery_error` |
| `metric_definition` | Per-channel absolute error between estimated coef (modeling scale) and true coef; report max, mean, and sign mismatch rate |
| `pass_criteria` | `TBD_v1` |
| `severity` | blocker |
| `owner` | Modeling / platform |
| `dependencies` | Ridge BO train path; `baseline_world`, `geo_world` |

---

### VAL-002 — Adstock recovery

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-002` |
| `capability` | Adstock recovery |
| `truth_object` | `media_truth.adstock_parameters` |
| `metric` | `adstock_param_error` |
| `metric_definition` | Absolute error on decay vs true; optional carryover correlation at fixed impulse (week-1 state vs analytic) |
| `pass_criteria` | `TBD_v1` |
| `severity` | blocker |
| `owner` | Modeling / platform |
| `dependencies` | `adstock_world`; geometric adstock implementation |

---

### VAL-003 — Hill recovery

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-003` |
| `capability` | Hill saturation recovery |
| `truth_object` | `media_truth.saturation_parameters` |
| `metric` | `hill_param_and_monotone_error` |
| `metric_definition` | Error on `half_max` and `slope`; max violation of monotone feature path in design matrix vs true surface |
| `pass_criteria` | `TBD_v1` |
| `severity` | blocker |
| `owner` | Modeling / platform |
| `dependencies` | `saturation_world`; Hill implementation |

---

### VAL-004 — Δμ recovery

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-004` |
| `capability` | Δμ recovery |
| `truth_object` | `decision_truth.true_delta_mu` |
| `metric` | `delta_mu_error` |
| `metric_definition` | Absolute and relative error between `mmm decide simulate` (or planning simulate) Δμ and true Δμ per `scenario_id` on full panel |
| `pass_criteria` | `TBD_v1` |
| `severity` | blocker |
| `owner` | Planning / platform |
| `dependencies` | `VAL-001`, `VAL-002`, `VAL-003`; `decision_simulate` |

---

### VAL-005 — Optimizer recovery

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-005` |
| `capability` | Optimizer recovery |
| `truth_object` | `decision_truth.true_optimal_budget`, `decision_truth.true_regret` |
| `metric` | `optimizer_l1_and_regret` |
| `metric_definition` | L1 distance from observed allocation to true optimum normalized by budget; regret = true Δμ(optimum) − true Δμ(observed) |
| `pass_criteria` | `TBD_v1` (corner-dominant worlds may use directional mode per `artifact_truth.expected_certification_levels`) |
| `severity` | blocker |
| `owner` | Planning / platform |
| `dependencies` | `optimizer_world`; `VAL-004`; existing `optimizer_certification` semantics |

---

### VAL-006 — Replay consistency

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-006` |
| `capability` | Replay consistency |
| `truth_object` | `experiment_truth.treatment_effects`, `experiment_truth.lift_definitions` |
| `metric` | `replay_loss_and_coverage` |
| `metric_definition` | Replay calibration loss vs holdout split; fraction of units matched to panel; generalization gap severity vs declared experiment quality |
| `pass_criteria` | `TBD_v1` |
| `severity` | blocker |
| `owner` | Calibration / governance |
| `dependencies` | `experiment_world`; replay ETL; prod replay gate |

---

### VAL-007 — Calibration robustness

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-007` |
| `capability` | Calibration robustness |
| `truth_object` | `experiment_truth.uncertainty`, `experiment_truth.freshness` |
| `metric` | `calibration_stability` |
| `metric_definition` | Sensitivity of calibration weights and coef shifts to experiment quality tier; false attach rate when truth lift is zero |
| `pass_criteria` | `TBD_v1` |
| `severity` | major |
| `owner` | Calibration / governance |
| `dependencies` | `VAL-006`; weighted replay |

---

### VAL-008 — Decision safety

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-008` |
| `capability` | Decision safety |
| `truth_object` | `artifact_truth.expected_gates` |
| `metric` | `decision_safe_and_disclosure` |
| `metric_definition` | `decision_safe` flag and unsupported-questions set match expected; no forbidden keys in prod simulation JSON |
| `pass_criteria` | Logical match to `artifact_truth` (no numeric threshold) |
| `severity` | blocker |
| `owner` | Governance / platform |
| `dependencies` | `mmm decide` paths; artifact allowlists |

---

### VAL-009 — Artifact integrity

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-009` |
| `capability` | Artifact integrity |
| `truth_object` | `artifact_truth.expected_gates`, `metadata.world_id` |
| `metric` | `artifact_contract_compliance` |
| `metric_definition` | Decision bundle tier, fingerprint match train→decide, semantic contract validation, economics metadata presence on prod |
| `pass_criteria` | `TBD_v1` for tolerance-based fields; hard fail on tier/contract violation |
| `severity` | blocker |
| `owner` | Platform / governance |
| `dependencies` | `VAL-008`; fingerprint module |

---

### VAL-010 — Reproducibility

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-010` |
| `capability` | Reproducibility |
| `truth_object` | `metadata.generation_seed`, run snapshot |
| `metric` | `repro_match_rate` |
| `metric_definition` | Fraction of declared outputs matching reference run within documented tolerances (coef, Δμ, allocation, report hashes) |
| `pass_criteria` | `TBD_v1` |
| `severity` | major |
| `owner` | Platform |
| `dependencies` | `reproducibility_certification`; `reference_run_path` pattern |

---

### VAL-011 — Promotion workflow

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-011` |
| `capability` | Promotion workflow |
| `truth_object` | `artifact_truth.expected_gates`, promotion record fixture |
| `metric` | `promotion_validation` |
| `metric_definition` | `require_promoted_model_for_prod_decision` paths accept valid promotion and reject expired / fingerprint-mismatch records as declared |
| `pass_criteria` | Logical match to `artifact_truth` |
| `severity` | major |
| `owner` | Governance |
| `dependencies` | `promotion_workflow` module |

---

### VAL-012 — Drift detection

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-012` |
| `capability` | Drift detection |
| `truth_object` | `shift_truth.changepoints`, `shift_truth.coefficient_drift` |
| `metric` | `drift_detection_performance` |
| `metric_definition` | Time-to-first warning or block vs changepoint; false positive rate on stable worlds |
| `pass_criteria` | `TBD_v1` |
| `severity` | major |
| `owner` | Governance / calibration |
| `dependencies` | `drift_world`; calibration readiness |

---

### VAL-013 — Governance behavior

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-013` |
| `capability` | Governance behavior |
| `truth_object` | `artifact_truth.expected_gates`, `artifact_truth.expected_failures` |
| `metric` | `gate_outcome_match` |
| `metric_definition` | Set equality between expected and observed gate outcomes (pass/fail/warn) per `gate_id` |
| `pass_criteria` | Exact match (logical) |
| `severity` | blocker |
| `owner` | Governance |
| `dependencies` | production readiness; optimization gates |

---

### VAL-014 — Certification behavior

| Field | Value |
|-------|-------|
| `validation_id` | `VAL-014` |
| `capability` | Certification behavior |
| `truth_object` | `artifact_truth.expected_certification_levels`, `artifact_truth.expected_warnings` |
| `metric` | `cert_level_match` |
| `metric_definition` | Reported certification levels (synthetic, optimizer mode, readiness) match world expectations |
| `pass_criteria` | Logical match to `artifact_truth` |
| `severity` | blocker |
| `owner` | Governance / platform |
| `dependencies` | SyntheticCertification; OptimizerCertification; production_readiness |

---

## 5. Cross-reference: certifications → registry

| Planned certification (roadmap §7) | Primary `validation_id` rows |
|-----------------------------------|------------------------------|
| SyntheticCertification | VAL-001, VAL-002, VAL-003, VAL-004, VAL-014 |
| DecisionCertification | VAL-004, VAL-008 |
| ReplayCertification | VAL-006, VAL-007 |
| OptimizerCertification | VAL-005, VAL-014 |
| GovernanceCertification | VAL-013, VAL-011, VAL-012 |
| ArtifactCertification | VAL-009, VAL-008 |
| ReliabilityCertification | All rows (aggregated) |

---

## 6. v1 CHECK_REGISTRY migration map (informative)

| Current check / module | Target `validation_id` | Target `world_id` (Phase 1) |
|------------------------|------------------------|----------------------------|
| `semi_log_delta_mu_exact` | VAL-004 | `TBD_v1` |
| `geometric_adstock_carryover` | VAL-002 | `TBD_v1` |
| `hill_saturation_analytic` | VAL-003 | `TBD_v1` |
| `two_channel_optimizer_direction` | VAL-005 | `TBD_v1` |
| `transform_policy_consistency` | VAL-013 | `TBD_v1` |

**Rule:** New work must not add checks outside this registry without a registry amendment ADR.

---

## 7. Change control

| Action | Requirement |
|--------|-------------|
| Add `validation_id` | Update this file + ADR if new truth domain |
| Set threshold | Replace `TBD_v1` with versioned threshold doc; owner sign-off |
| Deprecate row | Mark deprecated; retain ID for scorecard history |

**Frozen as of Phase 0:** Row IDs VAL-001 through VAL-014 and truth_object dot-paths.

---

## 8. Phase 4A structural certification mapping

Phase 4A executes **CERT-4A-001** through **CERT-4A-013** on materialized bundles via `certification_runner.py`. Registry rows **VAL-001** through **VAL-014** are listed in the certification report as **skipped** with explicit reasons — they are not executed and must not pass.

| CERT ID | Maps to registry / platform contract |
|---------|--------------------------------------|
| CERT-4A-001 | Bundle L1–L3 (`validator.py`) |
| CERT-4A-002 | INV-004 checksum reproducibility |
| CERT-4A-003 | Replay loader / VAL-006 prerequisite |
| CERT-4A-004–007 | Truth structure (transform, metadata, governance, decision) |
| CERT-4A-008–011 | CalibrationSignal + replay payload |
| CERT-4A-009 | DecisionSurface (`semi_log`, full-panel replay) |
| CERT-4A-010 | Estimand declarations |
| CERT-4A-012 | TrustReport gate expectations |
| CERT-4A-013 | Release-gate semantics |

See [certification_runner.md](certification_runner.md).

---

## 9. Phase 4B-2 recovery execution (WORLD-008)

On `WORLD-008-exact-recovery`, `run_world_certification(..., include_recovery=True)` executes:

| REC ID | VAL row | Status |
|--------|---------|--------|
| REC-4B2-001 | VAL-001 | pass/fail (TBD_v1_runtime coef tolerances) |
| REC-4B2-005 | VAL-004 | pass/fail (analytic Δμ replaces placeholder truth) |
| REC-4B2-006 | VAL-005 | skipped `requires_optimizer_truth_thresholds` |
| REC-4B2-002–004 | VAL-002/003 partial | transform consistency |

Tolerances are **provisional** (`TBD_v1_runtime` in code) — not production thresholds.

---

## 10. Phase 4B-3 optimizer execution (WORLD-009)

On `WORLD-009-optimizer-recovery`, `run_world_certification(..., include_recovery=True)` executes:

| REC ID | VAL row | Status |
|--------|---------|--------|
| REC-4B3-TRAIN | — | pass — Ridge coef fit with truth-pinned transforms |
| REC-4B3-OPT-PATH | — | pass — `optimize_budget_via_simulation` completes |
| REC-4B3-OPT | VAL-005 | pass/fail — allocation vs `true_optimal_budget` + `expected_allocation_band` |

Truth optimum: **161-step grid search** on true coefficients (`optimizer_truth.grid_search_true_optimum`). Surfaces that are flat or nearly equal-split are rejected by `validate_optimizer_surface`.

Train uses `channel_modulated` spend for identifiability; optimizer path is production simulation optimizer (not curve interpolation).

---

## 11. Phase 4B-4 replay execution (WORLD-010)

On `WORLD-010-replay-recovery`, `run_world_certification(..., include_recovery=True)` executes:

| REC ID | VAL row | Status |
|--------|---------|--------|
| REC-4B4-LOAD | — | pass — `load_calibration_units_from_json` on `replay_units.json` |
| REC-4B4-TRAIN | — | pass — Ridge coef fit with truth-pinned transforms |
| REC-4B4-REPLAY | VAL-006 | pass/fail — fitted replay-implied lift vs `experiment_truth` true lift |

True lift: truth coefficients + `implied_lift_from_counterfactual` under `full_panel_transform_estimand_mask` (pre-window adstock preserved; spend shock only inside estimand mask). Fails loudly if replay frames are window-sliced.

---

## 12. Phase 4B-5 reliability execution (WORLD-011 / WORLD-012)

| World | REC ID | VAL rows | Behavior |
|-------|--------|----------|----------|
| WORLD-011 | REC-4B5-DRIFT | VAL-012 (partial), VAL-014 | Pre-period Ridge train; post/pre MAE degradation + KPI shift; coef recovery **skipped** |
| WORLD-012 | REC-4B5-ID | VAL-013 (partial), VAL-014 | Collinear spend; VIF warning; coef recovery **skipped** (`recovery_marked_unstable`) |

VAL-012 executed via [drift_detection.md](drift_detection.md) `drift_detection_runner` on drift-recovery worlds (Phase 5E ✅).

---

## 13. Phase 4C — ReliabilityScorecard rollup

`mmm/validation/synthetic/reliability_scorecard.py` aggregates `synthetic_world_certification_report.json` from WORLD-008–012 into `validation/synthetic_reliability_scorecard.json`.

| Score | Meaning |
|-------|---------|
| 1.0 | pass |
| 0.5 | partial (e.g. VAL-012 partial) |
| 0.0 | fail |
| null | skipped / unsupported / out of scope |

Expected per-world skips are not penalized. See [reliability_scorecard.md](reliability_scorecard.md).

---

## 14. Governance classification (VAL-001–014)

| `validation_id` | `metric_class` | `can_block_release` | `threshold_status` | `required_evidence` |
|-----------------|----------------|---------------------|--------------------|---------------------|
| VAL-001 | diagnostic_attribution | false | TBD_v1_runtime | WORLD-008, L5B exact_recovery, INV-056; per-channel transform worlds for promotion |
| VAL-002 | diagnostic_attribution | false | TBD_v1_runtime | WORLD-008 transform sensitivity; truth-pinned ablation |
| VAL-003 | diagnostic_attribution | false | TBD_v1_runtime | Same as VAL-002 |
| VAL-004 | decision_grade | true | TBD_v1_runtime | WORLD-008–012; behavioral lattice; INV-056 Δμ evidence |
| VAL-005 | decision_grade | true | TBD_v1_runtime | WORLD-009 optimizer world; corner-dominant stratum |
| VAL-006 | decision_grade | true | TBD_v1_runtime | WORLD-010 replay world; experiment_truth enrichment |
| VAL-007 | decision_grade | conditional | TBD_v1_runtime | VAL-006 + calibration freshness gates |
| VAL-008 | structural | true | approved (logical) | Negative worlds; artifact_truth gate match |
| VAL-009 | structural | true | TBD_v1_runtime | Artifact contract suite; fingerprint paths |
| VAL-010 | structural | true | TBD_v1_runtime | Reproducibility certification reference runs |
| VAL-011 | structural | true | approved (logical) | Promotion workflow fixtures |
| VAL-012 | trust_modifier | conditional | TBD_v1_runtime | WORLD-011 + L5B drift cells; `drift_detection_runner` (Phase 5E ✅) |
| VAL-013 | trust_modifier | conditional | TBD_v1_runtime | WORLD-012; governance gate fixtures |
| VAL-014 | structural | true | approved (logical) | Certification level match per world |

**CERT-4A-001–013** (structural certification): `metric_class: structural`, `can_block_release: true`, `threshold_status: approved` where logical (bundle integrity); numeric tolerances `TBD_v1_runtime` where applicable.

**Severity vs metric_class:** Legacy `severity: blocker` on VAL-001–003 remains for historical reports; **release policy** follows `metric_class` and `can_block_release` per DR-04.
