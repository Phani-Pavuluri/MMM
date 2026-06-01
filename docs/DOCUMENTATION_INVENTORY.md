# Documentation inventory

Canonical docs live under `docs/` in numbered journey folders. There are **no** duplicate flat copies at `docs/*.md` (except this inventory, `README.md`, and `documentation_truth_audit.md`).

| Source file | Canonical location | Status | Notes |
|-------------|-------------------|--------|-------|
| `docs/01_getting_started/best_practices.md` | `01_getting_started/best_practices.md` | canonical | — |
| `docs/01_getting_started/config_yaml.md` | `01_getting_started/config_yaml.md` | canonical | — |
| `docs/01_getting_started/overview.md` | `01_getting_started/overview.md` | canonical | — |
| `docs/01_getting_started/python_api.md` | `01_getting_started/python_api.md` | canonical | — |
| `docs/01_getting_started/quickstart.md` | `01_getting_started/quickstart.md` | canonical | — |
| `docs/02_concepts/architecture.md` | `02_concepts/architecture.md` | canonical | — |
| `docs/02_concepts/bayesian.md` | `02_concepts/bayesian.md` | canonical | — |
| `docs/02_concepts/calibration.md` | `02_concepts/calibration.md` | canonical | — |
| `docs/02_concepts/control_templates.md` | `02_concepts/control_templates.md` | canonical | — |
| `docs/02_concepts/cv_modes.md` | `02_concepts/cv_modes.md` | canonical | — |
| `docs/02_concepts/statistical_validation.md` | `02_concepts/statistical_validation.md` | canonical | — |
| `docs/02_concepts/decision_vs_research.md` | `02_concepts/decision_vs_research.md` | canonical | — |
| `docs/02_concepts/diagnostics.md` | `02_concepts/diagnostics.md` | canonical | — |
| `docs/02_concepts/experiment_scheduler.md` | `02_concepts/experiment_scheduler.md` | canonical | — |
| `docs/02_concepts/feature_separability.md` | `02_concepts/feature_separability.md` | canonical | — |
| `docs/02_concepts/response_curves.md` | `02_concepts/response_curves.md` | canonical | — |
| `docs/02_concepts/ridge_bo.md` | `02_concepts/ridge_bo.md` | canonical | — |
| `docs/03_planning/budget_optimization.md` | `03_planning/budget_optimization.md` | canonical | — |
| `docs/03_planning/decision_runbook.md` | `03_planning/decision_runbook.md` | canonical | — |
| `docs/03_planning/planning_execution.md` | `03_planning/planning_execution.md` | canonical | — |
| `docs/03_planning/planning_howto.md` | `03_planning/planning_howto.md` | canonical | — |
| `docs/05_validation/platform_roadmap.md` | `05_validation/platform_roadmap.md` | canonical | **Master** contract-driven platform roadmap (5 tracks) |
| `docs/05_validation/synthetic_validation_roadmap.md` | `05_validation/synthetic_validation_roadmap.md` | canonical | Track 2 — Reliability Program; phases 0–6 preserved |
| `docs/05_validation/scenario_builder.md` | `05_validation/scenario_builder.md` | canonical | Phase 3B ScenarioBuilder MVP spec |
| `docs/05_validation/certification_runner.md` | `05_validation/certification_runner.md` | canonical | Phase 4A structural certification runner |
| `docs/05_validation/dgp_materialization.md` | `05_validation/dgp_materialization.md` | canonical | Phase 4B-1 rich DGP materialization |
| `mmm/validation/synthetic/dgp_materializer.py` | (code) | canonical | Equation-backed panel + diagnostics from truth |
| `mmm/validation/synthetic/recovery_certification.py` | (code) | canonical | Phase 4B-2–4B-4 recovery (WORLD-008/009/010) |
| `mmm/validation/synthetic/optimizer_truth.py` | (code) | canonical | Grid optimum + WORLD-009 decision_truth authoring |
| `mmm/validation/synthetic/replay_truth.py` | (code) | canonical | True replay lift + WORLD-010 experiment_truth authoring |
| `mmm/validation/synthetic/reliability_truth.py` | (code) | canonical | WORLD-011/012 drift + identifiability world builders |
| `validation/worlds/WORLD-008-exact-recovery/train_config.yaml` | (fixture) | canonical | Recovery certification Ridge train fixture |
| `validation/worlds/WORLD-009-optimizer-recovery/` | `validation/worlds/WORLD-009-optimizer-recovery/` | canonical | Optimizer recovery world (Phase 4B-3) |
| `validation/worlds/WORLD-009-optimizer-recovery/train_config.yaml` | (fixture) | canonical | WORLD-009 recovery train fixture |
| `validation/worlds/WORLD-010-replay-recovery/` | `validation/worlds/WORLD-010-replay-recovery/` | canonical | Replay calibration recovery world (Phase 4B-4) |
| `validation/worlds/WORLD-010-replay-recovery/train_config.yaml` | (fixture) | canonical | WORLD-010 recovery train fixture |
| `tests/test_synthetic_optimizer_recovery.py` | (tests) | canonical | Phase 4B-3 optimizer recovery tests |
| `tests/test_synthetic_replay_recovery.py` | (tests) | canonical | Phase 4B-4 replay recovery tests |
| `validation/worlds/WORLD-011-drift-recovery/` | `validation/worlds/WORLD-011-drift-recovery/` | canonical | Drift recovery world (Phase 4B-5) |
| `validation/worlds/WORLD-012-identifiability-recovery/` | `validation/worlds/WORLD-012-identifiability-recovery/` | canonical | Identifiability recovery world (Phase 4B-5) |
| `tests/test_synthetic_drift_identifiability_recovery.py` | (tests) | canonical | Phase 4B-5 drift/identifiability tests |
| `mmm/validation/synthetic/reliability_scorecard.py` | (code) | canonical | Phase 4C ReliabilityScorecard MVP |
| `docs/05_validation/reliability_scorecard.md` | `05_validation/reliability_scorecard.md` | canonical | Phase 4C scorecard spec |
| `validation/synthetic_reliability_scorecard.json` | (artifact) | canonical | Generated aggregate scorecard (WORLD-008–012) |
| `tests/test_synthetic_reliability_scorecard.py` | (tests) | canonical | Phase 4C scorecard tests |
| `mmm/validation/synthetic/lattice_sweep.py` | (code) | canonical | Phase 5A lattice sweep MVP |
| `docs/05_validation/lattice_sweep.md` | `05_validation/lattice_sweep.md` | canonical | Phase 5A lattice sweep spec |
| `docs/05_validation/bayesian_hierarchical_geo_mmm_roadmap.md` | `05_validation/bayesian_hierarchical_geo_mmm_roadmap.md` | canonical | Research Sandbox — Bayesian hierarchical geo MMM (Bayes-H1–H5, planning only) |
| `validation/reports/lattice_sweep_mvp_report.json` | (artifact) | canonical | Generated lattice sweep report |
| `tests/test_synthetic_lattice_sweep.py` | (tests) | canonical | Phase 5A lattice sweep tests |
| `mmm/validation/synthetic/behavioral_lattice_sweep.py` | (code) | canonical | Phase 5B behavioral lattice sweep MVP |
| `docs/05_validation/behavioral_lattice_sweep.md` | `05_validation/behavioral_lattice_sweep.md` | canonical | Phase 5B behavioral lattice spec |
| `docs/05_validation/exact_recovery_investigation.md` | `05_validation/exact_recovery_investigation.md` | canonical | Phase 5C exact recovery investigation program |
| `docs/05_validation/exact_recovery_investigation_report.md` | `05_validation/exact_recovery_investigation_report.md` | canonical | Phase 5C INV-056 investigation report |
| `docs/05_validation/reliability_threshold_governance.md` | `05_validation/reliability_threshold_governance.md` | canonical | Phase 5D metric classes and threshold governance |
| `docs/05_validation/drift_detection.md` | `05_validation/drift_detection.md` | canonical | Phase 5E VAL-012 drift_detection_runner |
| `docs/05_validation/trust_report_semantics.md` | `05_validation/trust_report_semantics.md` | canonical | Phase 5E TrustReport interpretation |
| `mmm/validation/synthetic/drift_detection_runner.py` | (code) | canonical | Phase 5E VAL-012 runner |
| `mmm/validation/synthetic/trust_report_semantics.py` | (code) | canonical | Phase 5E TrustReport rollup |
| `tests/test_drift_detection_runner.py` | (tests) | canonical | Phase 5E drift runner tests |
| `docs/05_validation/monte_carlo_reliability_program.md` | `05_validation/monte_carlo_reliability_program.md` | canonical | Phase 5F Monte Carlo program design |
| `docs/05_validation/monte_carlo_threshold_recommendations.md` | `05_validation/monte_carlo_threshold_recommendations.md` | canonical | Phase 5F threshold recommendations |
| `mmm/validation/synthetic/monte_carlo_reliability.py` | (code) | canonical | Phase 5F pilot characterization |
| `tests/test_monte_carlo_reliability.py` | (tests) | canonical | Phase 5F tests |
| `docs/05_validation/bayes_h1_decision_surface_preservation_adr.md` | `05_validation/bayes_h1_decision_surface_preservation_adr.md` | canonical | Bayes-H1 binding ADR — DecisionSurface preservation |
| `docs/05_validation/bayes_h2_calibration_signal_mapping_adr.md` | `05_validation/bayes_h2_calibration_signal_mapping_adr.md` | canonical | Bayes-H2 binding ADR — CalibrationSignal → hierarchy mapping |
| `docs/05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md` | `05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md` | canonical | Bayes-H2b binding ADR — hierarchy propagation & claim semantics |
| `docs/05_validation/bayes_h2d_hierarchical_model_spec_adr.md` | `05_validation/bayes_h2d_hierarchical_model_spec_adr.md` | canonical | Bayes-H2d binding ADR — hierarchical model spec (architecture only) |
| `docs/BAYES_H2B_VALIDATION_WORLDS_001.md` | `BAYES_H2B_VALIDATION_WORLDS_001.md` | canonical | Seven WORLD-BAYES-* validation world specifications (Track 2) |
| `docs/BAYES_H2B_VALIDATION_RUNNER_002.md` | `BAYES_H2B_VALIDATION_RUNNER_002.md` | canonical | hierarchy_evidence_validator contract + VAL-BAYES-001–012 |
| `docs/ROADMAP_ALIGNMENT_GATE.md` | `ROADMAP_ALIGNMENT_GATE.md` | canonical | MIP north star, risk-tiered gates, anti-drift policy |
| `docs/ROADMAP_ALIGNMENT_REGISTRY.md` | `ROADMAP_ALIGNMENT_REGISTRY.md` | canonical | Living tier/gate/status table (Bayes-H2b path) |
| `docs/MIP_PLATFORM_AUDIT_TEMPLATE.md` | `MIP_PLATFORM_AUDIT_TEMPLATE.md` | canonical | Recurring platform audit template (mini / phase / promotion) |
| `docs/audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md` | `audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md` | canonical | Phase audit — pre–Bayes-H3 sandbox (Yellow) |
| `docs/ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md` | `ACCIDENTAL_GEOX_TRACK_D_PASTE_QUARANTINE.md` | quarantine | Rollback pointer — GeoX Track B/D docs removed from MMM repo |
| `validation/worlds/world_catalog.index.json` | `validation/worlds/world_catalog.index.json` | canonical | Catalog index incl. seven WORLD-BAYES-* Bayes-H2b fixtures |
| `validation/worlds/WORLD-BAYES-*/` | `validation/worlds/WORLD-BAYES-*/` | canonical | Bayes-H2b no-fit hierarchy evidence fixture bundles |
| `docs/05_validation/bayesian_hierarchical_geo_mmm_refinement.md` | `05_validation/bayesian_hierarchical_geo_mmm_refinement.md` | canonical | Track 4 Bayesian architecture refinement |
| `docs/05_validation/investigations/` | `05_validation/investigations/` | canonical | Phase 5C supporting JSON artifacts |
| `mmm/validation/synthetic/exact_recovery_investigation.py` | (code) | canonical | Phase 5C analysis runner (regenerate artifacts) |
| `tests/test_exact_recovery_investigation.py` | (tests) | canonical | Phase 5C investigation smoke tests |
| `validation/reports/behavioral_lattice_sweep_mvp_report.json` | (artifact) | canonical | Generated behavioral lattice report |
| `tests/test_synthetic_behavioral_lattice_sweep.py` | (tests) | canonical | Phase 5B behavioral lattice tests |
| `mmm/validation/synthetic/certification_runner.py` | (code) | canonical | Phase 4A bundle certification orchestration |
| `mmm/validation/synthetic/certification_registry.py` | (code) | canonical | CERT-4A-* + deferred VAL-* registry |
| `docs/05_validation/groundtruth_contract.md` | `05_validation/groundtruth_contract.md` | canonical | Frozen GroundTruthWorld contract; Phase 3A generator rules §5 |
| `docs/05_validation/validation_registry.md` | `05_validation/validation_registry.md` | canonical | Frozen VAL-001–VAL-014 registry; thresholds TBD_v1 |
| `docs/05_validation/synthetic_architecture_decisions.md` | `05_validation/synthetic_architecture_decisions.md` | canonical | ADR bundle; DR-01/DR-02 resolved |
| `docs/05_validation/world_materialization.md` | `05_validation/world_materialization.md` | canonical | Option C bundle layout (Phase 1A) |
| `docs/05_validation/truth_versioning.md` | `05_validation/truth_versioning.md` | canonical | Three-version policy (Phase 1A) |
| `docs/05_validation/world_catalog.md` | `05_validation/world_catalog.md` | canonical | Catalog record spec; illustrative IDs only |
| `docs/05_validation/world_schema.md` | `05_validation/world_schema.md` | canonical | Frozen `world_truth.json` field schema (Phase 1B) |
| `docs/05_validation/world_bundle_schema.md` | `05_validation/world_bundle_schema.md` | canonical | `metadata.json`, `checksums.json`, catalog index |
| `docs/05_validation/world_validator_spec.md` | `05_validation/world_validator_spec.md` | canonical | Validator levels L1–L4 + invariants; L1–L3 implemented |
| `validation/worlds/WORLD-001-baseline/` | `validation/worlds/WORLD-001-baseline/` | canonical | Smoke world bundle (truth + materialized artifacts) |
| `validation/worlds/WORLD-002-replay/` | `validation/worlds/WORLD-002-replay/` | canonical | Replay smoke world + `replay_units.json` |
| `validation/worlds/WORLD-003-generated-baseline/` | `validation/worlds/WORLD-003-generated-baseline/` | canonical | Generator baseline smoke (`archetype_gen_v1.0.0`) |
| `validation/worlds/WORLD-004-generated-replay/` | `validation/worlds/WORLD-004-generated-replay/` | canonical | Generator replay smoke (`archetype_gen_v1.0.0`) |
| `validation/worlds/WORLD-005-scenario-low-noise/` | `validation/worlds/WORLD-005-scenario-low-noise/` | canonical | ScenarioBuilder smoke |
| `validation/worlds/WORLD-006-scenario-high-collinearity/` | `validation/worlds/WORLD-006-scenario-high-collinearity/` | canonical | ScenarioBuilder collinearity smoke |
| `validation/worlds/WORLD-007-scenario-replay-drift/` | `validation/worlds/WORLD-007-scenario-replay-drift/` | canonical | ScenarioBuilder replay + drift smoke |
| `validation/worlds/WORLD-008-exact-recovery/` | `validation/worlds/WORLD-008-exact-recovery/` | canonical | Rich DGP exact-recovery world (Phase 4B-1) |
| `docs/04_governance/artifact_schema.md` | `04_governance/artifact_schema.md` | canonical | Replaces legacy `planning_artifact_schema.md` |
| `docs/04_governance/operator_runbook.md` | `04_governance/operator_runbook.md` | canonical | — |
| `docs/04_governance/prod_safety_checklist.md` | `04_governance/prod_safety_checklist.md` | canonical | — |
| `docs/04_governance/ridge_uncertainty_research.md` | `04_governance/ridge_uncertainty_research.md` | canonical | Research findings only |
| `docs/README.md` | `README.md` | canonical | Entry index |
| `docs/documentation_truth_audit.md` | `documentation_truth_audit.md` | canonical | Truth audit |
| `docs/06_investigations/open_investigations.md` | `06_investigations/open_investigations.md` | canonical | Living investigation backlog (INV-001+) |
| `docs/06_investigations/investigation_index.md` | `06_investigations/investigation_index.md` | canonical | Grouped index by risk area |
| `docs/planning_artifact_schema.md` | `04_governance/artifact_schema.md` | deprecated | Redirect stub only |
| `docs/DOCUMENTATION_INVENTORY.md` | `DOCUMENTATION_INVENTORY.md` | canonical | This file |
| `docs/_archive/cross_branch_not_shipped.md` | `_archive/cross_branch_not_shipped.md` | archived | Cross-branch quarantine policy |
| `docs/_archive/roadmap_causal_calibration_governance.md` | `_archive/roadmap_causal_calibration_governance.md` | archived | Pre-platform roadmap; superseded by `platform_roadmap.md` narrative |

## Deprecated redirects

| Former path | Redirect |
|-------------|----------|
| `docs/planning_artifact_schema.md` | Use [`04_governance/artifact_schema.md`](04_governance/artifact_schema.md) |
| `docs/quickstart.md` (flat) | Use [`01_getting_started/quickstart.md`](01_getting_started/quickstart.md) |
| `docs/decision_runbook.md` (flat) | Use [`03_planning/decision_runbook.md`](03_planning/decision_runbook.md) |

## Validation

```bash
python scripts/validate_docs.py
```
