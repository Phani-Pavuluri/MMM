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
| `docs/05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md` | `05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md` | canonical | MIP-C1 Ridge diagnostic attachment contract (context-only) |
| `docs/audits/AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md` | `audits/AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md` | canonical | MIP-C1 CalibrationSignal → MMM integration audit gate |
| `mmm/diagnostics/calibration_signal_attachment.py` | (code) | canonical | MIP-C1 context-only CalibrationSignal attachment helpers |
| `tests/mip/test_calibration_signal_mmm_attachment_contract.py` | (tests) | canonical | MIP-C1 attachment contract tests |
| `docs/06_investigations/H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_MANIFEST.md` | `06_investigations/H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_MANIFEST.md` | canonical | H11 real-bundle manifest (benchmark geo panel v1) |
| `docs/06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md` | `06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md` | canonical | H11 investigation — real-bundle Ridge diagnostic hardening |
| `mmm/diagnostics/ridge_real_bundle_hardening.py` | (code) | canonical | H11 real-bundle Ridge diagnostic runner |
| `tests/diagnostics/test_ridge_diagnostics_real_bundle_compat.py` | (tests) | canonical | H11 schema/bundle compatibility tests |
| `docs/05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json` | `05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json` | canonical | H11 redacted diagnostic archive (benchmark panel) |
| `docs/05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md` | `05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md` | canonical | H11 operator summary archive |
| `docs/audits/AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md` | `audits/AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md` | canonical | MIP-C2 train/extension CalibrationSignal wiring audit |
| `mmm/diagnostics/calibration_signal_ingestion.py` | (code) | canonical | MIP-C2 signal load + lineage + report ingest |
| `tests/mip/test_calibration_signal_train_boundary_ingestion.py` | (tests) | canonical | MIP-C2 train-boundary ingestion tests |
| `docs/05_validation/archives/MIP_C2_RIDGE_DIAGNOSTICS_WITH_CALIBRATION_SIGNAL_CONTEXT_20260601.json` | `05_validation/archives/MIP_C2_RIDGE_DIAGNOSTICS_WITH_CALIBRATION_SIGNAL_CONTEXT_20260601.json` | canonical | MIP-C2 representative diagnostic archive with context |
| `docs/05_validation/geox_cls_to_calibration_signal_adapter_contract.md` | `05_validation/geox_cls_to_calibration_signal_adapter_contract.md` | canonical | MIP-C3 GeoX/CLS export adapter contract |
| `docs/audits/AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md` | `audits/AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md` | canonical | MIP-C3 adapter gate audit |
| `mmm/diagnostics/calibration_signal_adapters.py` | (code) | canonical | MIP-C3 GeoX/CLS → CalibrationSignal adapters |
| `tests/mip/test_calibration_signal_adapters.py` | (tests) | canonical | MIP-C3 adapter tests |
| `docs/05_validation/archives/MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json` | `05_validation/archives/MIP_C3_ADAPTED_GEOX_CLS_SIGNALS_20260601.json` | canonical | MIP-C3 adapted signal bundle for C2 ingest |
| `docs/audits/AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md` | `audits/AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md` | canonical | MIP-C4 ETL dry-run gate audit |
| `mmm/diagnostics/calibration_signal_etl.py` | (code) | canonical | MIP-C4 dry-run ETL module + CLI |
| `tests/mip/test_calibration_signal_etl_dry_run.py` | (tests) | canonical | MIP-C4 ETL dry-run tests |
| `docs/05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json` | `05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json` | canonical | MIP-C4 dry-run signals artifact |
| `docs/05_validation/archives/MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json` | `05_validation/archives/MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json` | canonical | MIP-C4 train consumption proof archive |
| `docs/audits/AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md` | `audits/AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md` | canonical | MIP-C5 scheduled drop-zone ETL wrapper audit |
| `mmm/diagnostics/calibration_signal_etl_job.py` | (code) | canonical | MIP-C5 scheduled ETL job module + CLI |
| `tests/mip/test_calibration_signal_etl_job.py` | (tests) | canonical | MIP-C5 scheduled ETL job tests |
| `docs/05_validation/archives/mip_c5_etl_outputs/` | `05_validation/archives/mip_c5_etl_outputs/` | canonical | MIP-C5 reference job outputs (manifest, signals, summary) |
| `docs/05_validation/archives/MIP_C5_TRAIN_WITH_SCHEDULED_ETL_SIGNALS_20260601.json` | `05_validation/archives/MIP_C5_TRAIN_WITH_SCHEDULED_ETL_SIGNALS_20260601.json` | canonical | MIP-C5 train consumption proof |
| `docs/audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md` | `audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md` | canonical | MIP-C6 integration readiness checkpoint — verdict `continue_with_pause_before_live_scheduler` |
| `docs/05_validation/mmm_package_side_agents_roadmap.md` | `05_validation/mmm_package_side_agents_roadmap.md` | canonical | Future MMM package-side support agents — deferred roadmap; prerequisites and boundaries |
| `docs/05_validation/mmm_to_mip_export_contract_inventory.md` | `05_validation/mmm_to_mip_export_contract_inventory.md` | canonical | MMM-EXPORT-001 governed MMM→MIP export contract inventory (docs only) |
| `docs/05_validation/mmm_export_schema_and_fixture_contract.md` | `05_validation/mmm_export_schema_and_fixture_contract.md` | canonical | MMM-EXPORT-002 typed schemas, validators, fixture bundles |
| `docs/05_validation/mmm_typed_failure_packet_contract.md` | `05_validation/mmm_typed_failure_packet_contract.md` | canonical | MMM-owned versioned typed failure packets and producer export outcomes; not MIP platform policy |
| `docs/05_validation/mmm_typed_run_manifest_contract.md` | `05_validation/mmm_typed_run_manifest_contract.md` | canonical | MMM-owned versioned typed run-manifest evidence and additive producer export-boundary linkage; not MIP policy |
| `docs/05_validation/mmm_calibration_treatment_lineage_contract.md` | `05_validation/mmm_calibration_treatment_lineage_contract.md` | canonical | MMM-owned typed calibration-treatment lineage; preserves producer evidence without changing fitting or MIP policy |
| `docs/05_validation/mmm_typed_diagnostics_limitations_contract.md` | `05_validation/mmm_typed_diagnostics_limitations_contract.md` | canonical | MMM-owned typed diagnostics, limitations, and technical-claim evidence |
| `docs/05_validation/MMM_MIP_HANDOFF_V1_REMAINING_GAP_SELECTION_AUDIT_001_REPORT.md` | `05_validation/MMM_MIP_HANDOFF_V1_REMAINING_GAP_SELECTION_AUDIT_001_REPORT.md` | canonical | Evidence reconciliation and single next MMM producer task selection |
| `docs/05_validation/mmm_producer_golden_fixture_suite.md` | `05_validation/mmm_producer_golden_fixture_suite.md` | canonical | Versioned deterministic public producer golden fixture suite |
| `docs/05_validation/MMM_MIP_HANDOFF_V1_POST_GOLDEN_FIXTURES_GAP_SELECTION_AUDIT_001_REPORT.md` | `05_validation/MMM_MIP_HANDOFF_V1_POST_GOLDEN_FIXTURES_GAP_SELECTION_AUDIT_001_REPORT.md` | canonical | Post-R13 evidence audit selecting the R15 compatibility-policy prerequisite |
| `docs/05_validation/mmm_mip_schema_compatibility_policy.md` | `05_validation/mmm_mip_schema_compatibility_policy.md` | canonical | MMM-owned public producer schema compatibility, deprecation, fail-closed, and fixture-version policy; not an interface freeze or MIP parser |
| `docs/05_validation/archives/MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001_registry.json` | `05_validation/archives/MMM_MIP_HANDOFF_V1_SCHEMA_COMPATIBILITY_POLICY_001_registry.json` | canonical | Deterministic R15 public-contract and golden-fixture support registry |
| `mmm/contracts/mip_export.py` | (code) | canonical | MMM-owned producer export Pydantic schemas, structural validation, and technical claim evidence; not a frozen MIP interface |
| `mmm/contracts/mip_export_adapter.py` | (code) | canonical | MMM-owned conservative producer serialization adapter; emits partial, non-consumable bundles only |
| `mmm/contracts/mip_failure.py` | (code) | canonical | MMM-owned typed failure taxonomy, remediation policy, JSON-safe packet serialization, and discriminated producer outcome |
| `mmm/contracts/run_manifest.py` | (code) | canonical | MMM-owned typed run-manifest, step, and safe artifact-reference contracts; legacy dict helper retained for compatibility |
| `mmm/contracts/calibration_treatment.py` | (code) | canonical | MMM-owned typed calibration-treatment freshness, compatibility, disposition, role, transformation, and failure lineage |
| `tests/fixtures/mip_export/` | (fixtures) | canonical | MMM-owned synthetic producer-schema fixtures (non-production) |
| `tests/contracts/test_mmm_mip_export_contracts.py` | (tests) | canonical | MMM-EXPORT-002 schema and claim-safety tests |
| `tests/contracts/test_mmm_failure_packet.py` | (tests) | canonical | Typed producer failure packet, outcome, fixture, and boundary-emission tests |
| `tests/contracts/test_mmm_run_manifest.py` | (tests) | canonical | Typed producer run-manifest, fixture, serialization, and boundary-linkage tests |
| `tests/contracts/test_mmm_mip_producer_boundary.py` | (tests) | canonical | MMM producer-boundary regression: no MIP parser or conversational policy ships here |
| `docs/05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md` | `05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md` | canonical | Bayes-H2b binding ADR — hierarchy propagation & claim semantics |
| `docs/05_validation/bayes_h2d_hierarchical_model_spec_adr.md` | `05_validation/bayes_h2d_hierarchical_model_spec_adr.md` | canonical | Bayes-H2d binding ADR — hierarchical model spec (architecture only) |
| `docs/BAYES_H2B_VALIDATION_WORLDS_001.md` | `BAYES_H2B_VALIDATION_WORLDS_001.md` | canonical | Seven WORLD-BAYES-* validation world specifications (Track 2) |
| `docs/BAYES_H2B_VALIDATION_RUNNER_002.md` | `BAYES_H2B_VALIDATION_RUNNER_002.md` | canonical | hierarchy_evidence_validator contract + VAL-BAYES-001–012 |
| `docs/ROADMAP_ALIGNMENT_GATE.md` | `ROADMAP_ALIGNMENT_GATE.md` | canonical | MIP north star, risk-tiered gates, anti-drift policy |
| `docs/ROADMAP_ALIGNMENT_REGISTRY.md` | `ROADMAP_ALIGNMENT_REGISTRY.md` | canonical | Living tier/gate/status table (Bayes-H2b path) |
| `docs/MIP_PLATFORM_AUDIT_TEMPLATE.md` | `MIP_PLATFORM_AUDIT_TEMPLATE.md` | canonical | Recurring platform audit template (mini / phase / promotion) |
| `docs/audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md` | `audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md` | canonical | Phase audit — pre–Bayes-H3 sandbox (Yellow) |
| `docs/audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md` | `audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md` | canonical | Bayes-H5p shadow workflow audit gate (H5l–H5o; expansion criteria; prod blocked) |
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
| `docs/06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md` | `06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md` | canonical | Bayes-H4 sparse partial-pooling shrinkage investigation |
| `docs/05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json` | `05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json` | canonical | INV-H4-001b sparse diagnostic variant sweep (research only) |
| `docs/05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json` | `05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json` | canonical | H4b-refresh extended repeated pilot (primary + legacy shrinkage) |
| `docs/05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json` | `05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json` | canonical | H4c extended recovery reliability map (research only) |
| `docs/05_validation/bayes_h5_model_spec_improvement_adr.md` | `05_validation/bayes_h5_model_spec_improvement_adr.md` | canonical | Bayes-H5 sandbox model-spec ADR (H5a–H5r complete) |
| `docs/05_validation/bayes_h6_synthetic_lane_adr.md` | `05_validation/bayes_h6_synthetic_lane_adr.md` | canonical | Bayes-H6 production-shaped synthetic validation (H6a–H6f) |
| `docs/05_validation/ridge_production_diagnostics_contract.md` | `05_validation/ridge_production_diagnostics_contract.md` | canonical | H7 Ridge production diagnostic contract |
| `docs/06_investigations/INV-H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX.md` | `06_investigations/INV-H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX.md` | canonical | H6f Ridge vs H5 synthetic benchmark matrix |
| `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json` | `05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json` | canonical | H7 representative Ridge diagnostics artifact |
| `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_REPORT_20260601.json` | `05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_REPORT_20260601.json` | canonical | H8 exported Ridge diagnostic JSON |
| `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SUMMARY_20260601.md` | `05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SUMMARY_20260601.md` | canonical | H8 operator Markdown summary |
| `docs/05_validation/ridge_diagnostic_severity_policy.md` | `05_validation/ridge_diagnostic_severity_policy.md` | canonical | H9 Ridge severity and output eligibility policy |
| `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SEVERITY_20260601.json` | `05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SEVERITY_20260601.json` | canonical | H9 severity + output_eligibility example |
| `docs/audits/AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md` | `audits/AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md` | canonical | H10 Ridge diagnostic end-to-end audit gate |
| `docs/05_validation/archives/BAYES_H10_RIDGE_DIAGNOSTIC_E2E_AUDIT_20260601.json` | `05_validation/archives/BAYES_H10_RIDGE_DIAGNOSTIC_E2E_AUDIT_20260601.json` | canonical | H10 automated E2E audit results |
| `docs/06_investigations/INV-H5R_SPARSE_CHANNEL_REMEDY_REPLAY.md` | `06_investigations/INV-H5R_SPARSE_CHANNEL_REMEDY_REPLAY.md` | canonical | H5r sparse-radio remedy on H5q panel |
| `docs/05_validation/archives/BAYES_H5R_REMEDY_COMPARISON_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json` | `05_validation/archives/BAYES_H5R_REMEDY_COMPARISON_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json` | canonical | H5q vs H5r remedy comparison |
| `docs/06_investigations/INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md` | `06_investigations/INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md` | canonical | Bayes-H5q third panel (triangulation; failed_convergence) |
| `docs/05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json` | `05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json` | canonical | H5q shadow run artifact |
| `docs/06_investigations/INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md` | `06_investigations/INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md` | canonical | Bayes-H5o second real-panel shadow (benchmark panel) |
| `docs/05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json` | `05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json` | canonical | H5o second-panel shadow run artifact |
| `docs/06_investigations/INV-H5N_SHADOW_POLICY_RECOMMENDER.md` | `06_investigations/INV-H5N_SHADOW_POLICY_RECOMMENDER.md` | canonical | Bayes-H5n shadow-policy recommender (complete) |
| `docs/05_validation/archives/BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json` | `05_validation/archives/BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json` | canonical | H5n sample-panel shadow-policy recommendation artifact |
| `docs/06_investigations/investigation_index.md` | `06_investigations/investigation_index.md` | canonical | Grouped index by risk area |
| `docs/planning_artifact_schema.md` | `04_governance/artifact_schema.md` | deprecated | Redirect stub only |
| `docs/DOCUMENTATION_INVENTORY.md` | `DOCUMENTATION_INVENTORY.md` | canonical | This file |
| `docs/_archive/cross_branch_not_shipped.md` | `_archive/cross_branch_not_shipped.md` | archived | Cross-branch quarantine policy |
| `docs/_archive/roadmap_causal_calibration_governance.md` | `_archive/roadmap_causal_calibration_governance.md` | archived | Pre-platform roadmap; superseded by `platform_roadmap.md` narrative |
| `docs/02_concepts/continuous_validation.md` | `02_concepts/continuous_validation.md` | canonical | Documented concept page |
| `docs/02_concepts/decision_validation.md` | `02_concepts/decision_validation.md` | canonical | Documented concept page |
| `docs/02_concepts/experiment_evidence.md` | `02_concepts/experiment_evidence.md` | canonical | Documented concept page |
| `docs/02_concepts/hierarchical_borrowing.md` | `02_concepts/hierarchical_borrowing.md` | canonical | Documented concept page |
| `docs/02_concepts/robust_optimization_research.md` | `02_concepts/robust_optimization_research.md` | canonical | Documented concept page |
| `docs/02_concepts/uncertainty_propagation.md` | `02_concepts/uncertainty_propagation.md` | canonical | Documented concept page |
| `docs/04_governance/calibration_freshness.md` | `04_governance/calibration_freshness.md` | canonical | Documented governance page |
| `docs/04_governance/decision_artifact_contract.md` | `04_governance/decision_artifact_contract.md` | canonical | Documented governance page |
| `docs/04_governance/decision_stress.md` | `04_governance/decision_stress.md` | canonical | Documented governance page |
| `docs/04_governance/decision_trace.md` | `04_governance/decision_trace.md` | canonical | Documented governance page |
| `docs/04_governance/optimizer_certification.md` | `04_governance/optimizer_certification.md` | canonical | Documented governance page |
| `docs/04_governance/performance_certification.md` | `04_governance/performance_certification.md` | canonical | Documented governance page |
| `docs/04_governance/production_readiness.md` | `04_governance/production_readiness.md` | canonical | Documented governance page |
| `docs/04_governance/promotion_workflow.md` | `04_governance/promotion_workflow.md` | canonical | Documented governance page |
| `docs/04_governance/reproducibility_certification.md` | `04_governance/reproducibility_certification.md` | canonical | Documented governance page |
| `docs/04_governance/synthetic_certification.md` | `04_governance/synthetic_certification.md` | canonical | Documented governance page |
| `docs/04_governance/v1_release_notes.md` | `04_governance/v1_release_notes.md` | canonical | Documented governance page |
| `docs/05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_summary.md` | `05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_summary.md` | canonical | MIP-C5 dry-run summary |
| `docs/05_validation/bayes_h3_research_sandbox_backend_adr.md` | `05_validation/bayes_h3_research_sandbox_backend_adr.md` | canonical | Bayesian research sandbox ADR |
| `docs/05_validation/bayes_h4_recovery_worlds_adr.md` | `05_validation/bayes_h4_recovery_worlds_adr.md` | canonical | Bayesian recovery-worlds ADR |
| `docs/06_investigations/H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | `06_investigations/H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | canonical | Investigation manifest |
| `docs/06_investigations/H5O_SECOND_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | `06_investigations/H5O_SECOND_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | canonical | Investigation manifest |
| `docs/06_investigations/H5Q_THIRD_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | `06_investigations/H5Q_THIRD_REAL_PANEL_SHADOW_RUN_MANIFEST.md` | canonical | Investigation manifest |
| `docs/06_investigations/INV-071_BAYES_H4_TRUE_EFFECT_RECOVERY_THRESHOLDS.md` | `06_investigations/INV-071_BAYES_H4_TRUE_EFFECT_RECOVERY_THRESHOLDS.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md` | `06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5B_REPEATED_PILOT_AND_DIAGNOSTICS.md` | `06_investigations/INV-H5B_REPEATED_PILOT_AND_DIAGNOSTICS.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5C_EXTENDED_MCMC_CONFIRMATION.md` | `06_investigations/INV-H5C_EXTENDED_MCMC_CONFIRMATION.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5D_TRUST_DIAGNOSTIC_MAPPING.md` | `06_investigations/INV-H5D_TRUST_DIAGNOSTIC_MAPPING.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md` | `06_investigations/INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5F_REAL_PANEL_SHADOW_RUN_HARNESS.md` | `06_investigations/INV-H5F_REAL_PANEL_SHADOW_RUN_HARNESS.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md` | `06_investigations/INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5H_REAL_PANEL_SHADOW_HARDENING.md` | `06_investigations/INV-H5H_REAL_PANEL_SHADOW_HARDENING.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5I_REAL_PANEL_CONVERGENCE_DIAGNOSTICS.md` | `06_investigations/INV-H5I_REAL_PANEL_CONVERGENCE_DIAGNOSTICS.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5J_COLLINEARITY_GEOMETRY_ABLATIONS.md` | `06_investigations/INV-H5J_COLLINEARITY_GEOMETRY_ABLATIONS.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5K_GEOMETRY_STABILIZATION.md` | `06_investigations/INV-H5K_GEOMETRY_STABILIZATION.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5L_HIERARCHY_GEOMETRY_REFINEMENT.md` | `06_investigations/INV-H5L_HIERARCHY_GEOMETRY_REFINEMENT.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H5M_FROZEN_SHADOW_POLICY_REPLAY.md` | `06_investigations/INV-H5M_FROZEN_SHADOW_POLICY_REPLAY.md` | canonical | Investigation report |
| `docs/06_investigations/INV-H6_SYNTHETIC_TO_REAL_SHADOW_COMPARISON.md` | `06_investigations/INV-H6_SYNTHETIC_TO_REAL_SHADOW_COMPARISON.md` | canonical | Investigation report |
| `docs/audits/MIP_MINI_AUDIT_20260601_BAYES_H3_SANDBOX_MVP.md` | `audits/MIP_MINI_AUDIT_20260601_BAYES_H3_SANDBOX_MVP.md` | canonical | Audit report |
| `docs/dev_setup.md` | `dev_setup.md` | canonical | Development setup |
| `docs/dev_validation_workflow.md` | `dev_validation_workflow.md` | canonical | Docker/devcontainer validation workflow |
| `docs/track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md` | `track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md` | canonical | Track D readout |

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
