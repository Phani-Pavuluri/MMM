# Documentation

Entry point for the MMM geo-level library. Docs are grouped by **user journey**, not by implementation history.

## New here?

Start with what the system is, then run a train:

1. [01_getting_started/overview.md](01_getting_started/overview.md) — purpose and design goals  
2. [01_getting_started/quickstart.md](01_getting_started/quickstart.md) — install, train, first artifacts  
3. [01_getting_started/config_yaml.md](01_getting_started/config_yaml.md) — configuration reference  
4. [01_getting_started/python_api.md](01_getting_started/python_api.md) — programmatic train and decide APIs  

Optional: [01_getting_started/best_practices.md](01_getting_started/best_practices.md)

## Understand the system

Concepts and modeling behavior (read before production decisioning):

| Topic | Doc |
|--------|-----|
| System shape | [02_concepts/architecture.md](02_concepts/architecture.md) |
| Decision vs diagnostic outputs | [02_concepts/decision_vs_research.md](02_concepts/decision_vs_research.md) |
| Experiments & calibration | [02_concepts/calibration.md](02_concepts/calibration.md) |
| Post-fit diagnostics | [02_concepts/diagnostics.md](02_concepts/diagnostics.md) |
| Split-channel separability | [02_concepts/feature_separability.md](02_concepts/feature_separability.md) |
| Experiment prioritization | [02_concepts/experiment_scheduler.md](02_concepts/experiment_scheduler.md) |
| Control CSV templates (onboarding) | [02_concepts/control_templates.md](02_concepts/control_templates.md) |
| Response curves (diagnostic) | [02_concepts/response_curves.md](02_concepts/response_curves.md) |
| CV modes | [02_concepts/cv_modes.md](02_concepts/cv_modes.md) |
| Ridge + BO | [02_concepts/ridge_bo.md](02_concepts/ridge_bo.md) |
| Bayesian MMM | [02_concepts/bayesian.md](02_concepts/bayesian.md) |

## Plan or optimize budgets

Operational workflows for analysts and data scientists:

| Topic | Doc |
|--------|-----|
| **How-to (start here)** | [03_planning/planning_howto.md](03_planning/planning_howto.md) |
| Decision rules & CLI contract | [03_planning/decision_runbook.md](03_planning/decision_runbook.md) |
| Budget optimization concepts | [03_planning/budget_optimization.md](03_planning/budget_optimization.md) |
| Execution pipeline (developers) | [03_planning/planning_execution.md](03_planning/planning_execution.md) |

## Platform & validation roadmap

**Planning frame:** Contract-driven **Marketing Intelligence Platform** (MMM is one implementation surface). Reliability proving is the primary gate for modeling expansion.

| Topic | Doc |
|--------|-----|
| **Roadmap alignment gate** (north star, risk tiers, anti-drift) | [ROADMAP_ALIGNMENT_GATE.md](ROADMAP_ALIGNMENT_GATE.md) |
| **Roadmap alignment registry** (Bayes-H2b/H2d/H3 status) | [ROADMAP_ALIGNMENT_REGISTRY.md](ROADMAP_ALIGNMENT_REGISTRY.md) |
| **MIP platform audit template** (phase / promotion / mini) | [MIP_PLATFORM_AUDIT_TEMPLATE.md](MIP_PLATFORM_AUDIT_TEMPLATE.md) |
| Pre–Bayes-H3 phase audit (2026-06-01) | [audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md](audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md) |
| **Master platform roadmap** | [05_validation/platform_roadmap.md](05_validation/platform_roadmap.md) (MMM tracks 1–5) |
| Track 2 — Reliability program (phases 0–6, preserved) | [05_validation/synthetic_validation_roadmap.md](05_validation/synthetic_validation_roadmap.md) |
| ScenarioBuilder (Phase 3B ✅) | [05_validation/scenario_builder.md](05_validation/scenario_builder.md) |
| Structural certification runner (Phase 4A ✅) | [05_validation/certification_runner.md](05_validation/certification_runner.md) |
| Rich DGP materialization (Phase 4B-1 ✅) | [05_validation/dgp_materialization.md](05_validation/dgp_materialization.md) |
| ReliabilityScorecard (Phase 4C / 5D metric classes) | [05_validation/reliability_scorecard.md](05_validation/reliability_scorecard.md) |
| Reliability threshold governance (Phase 5D ✅) | [05_validation/reliability_threshold_governance.md](05_validation/reliability_threshold_governance.md) |
| Lattice sweep MVP (Phase 5A ✅) | [05_validation/lattice_sweep.md](05_validation/lattice_sweep.md) |
| Behavioral lattice sweep MVP (Phase 5B ✅) | [05_validation/behavioral_lattice_sweep.md](05_validation/behavioral_lattice_sweep.md) |
| Exact recovery investigation (Phase 5C ✅) | [05_validation/exact_recovery_investigation.md](05_validation/exact_recovery_investigation.md) |
| Drift detection / TrustReport semantics (Phase 5E ✅) | [05_validation/drift_detection.md](05_validation/drift_detection.md), [trust_report_semantics.md](05_validation/trust_report_semantics.md) |
| Monte Carlo reliability (Phase 5F ✅) | [05_validation/monte_carlo_reliability_program.md](05_validation/monte_carlo_reliability_program.md), [monte_carlo_threshold_recommendations.md](05_validation/monte_carlo_threshold_recommendations.md) |
| Bayesian Hierarchical Geo MMM (Research Sandbox) | [bayesian_hierarchical_geo_mmm_roadmap.md](05_validation/bayesian_hierarchical_geo_mmm_roadmap.md) |
| Bayes-H1 DecisionSurface Preservation ADR ✅ | [05_validation/bayes_h1_decision_surface_preservation_adr.md](05_validation/bayes_h1_decision_surface_preservation_adr.md) |
| Bayes-H2 CalibrationSignal Mapping ADR ✅ | [05_validation/bayes_h2_calibration_signal_mapping_adr.md](05_validation/bayes_h2_calibration_signal_mapping_adr.md) |
| Bayes-H2b Hierarchical Experiment-Prior Scope Rules ADR ✅ | [05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) |
| Bayes-H2b validation worlds catalog (001) ✅ | [BAYES_H2B_VALIDATION_WORLDS_001.md](BAYES_H2B_VALIDATION_WORLDS_001.md) |
| Bayes-H2b validation runner contract (002) ✅ | [BAYES_H2B_VALIDATION_RUNNER_002.md](BAYES_H2B_VALIDATION_RUNNER_002.md) |
| Bayes-H2b hierarchy evidence validator (no-fit stub) | `mmm.validation.synthetic.hierarchy_evidence_validator` — `python -m mmm.validation.synthetic.hierarchy_evidence_validator --smoke VAL-BAYES-H2B-SMOKE` |
| Bayes-H2d hierarchical model spec ADR ✅ | [05_validation/bayes_h2d_hierarchical_model_spec_adr.md](05_validation/bayes_h2d_hierarchical_model_spec_adr.md) |
| Bayes-H3 sandbox backend ADR (PyMC; NumPyro deferred) ✅ | [05_validation/bayes_h3_research_sandbox_backend_adr.md](05_validation/bayes_h3_research_sandbox_backend_adr.md) |
| Bayes-H4 recovery worlds ADR ✅ | [05_validation/bayes_h4_recovery_worlds_adr.md](05_validation/bayes_h4_recovery_worlds_adr.md) |
| Bayes-H4a threshold pilot JSON | [05_validation/archives/BAYES_H4_THRESHOLD_PILOT_20260601.json](05_validation/archives/BAYES_H4_THRESHOLD_PILOT_20260601.json) |
| Bayes-H4b repeated recovery pilot JSON (legacy field) | [05_validation/archives/BAYES_H4_REPEATED_PILOT_20260601.json](05_validation/archives/BAYES_H4_REPEATED_PILOT_20260601.json) |
| Bayes-H4b-refresh primary-metric repeated pilot | [05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json](05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json) — **authoritative pooling evidence**; supersedes original H4b JSON for shrinkage interpretation |
| Bayes-H4b-disposition (C+A) | [06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md](06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md) §11 — pooling passes; true-effect recovery report-only policy (INV-071) |
| Bayes-H4c extended recovery pilot JSON | [05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json](05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json) — reliability map (research only) |
| INV-071 recovery threshold policy | [06_investigations/INV-071_BAYES_H4_TRUE_EFFECT_RECOVERY_THRESHOLDS.md](06_investigations/INV-071_BAYES_H4_TRUE_EFFECT_RECOVERY_THRESHOLDS.md); [policy JSON](05_validation/archives/BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json) — report-only, no prod promotion |
| INV-H4D sparse/τ stability pilot | [06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md](06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md); [fast](05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_20260601.json) + [extended MCMC](05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_EXTENDED_20260601.json) — report-only, no prod promotion |
| Bayes-H5 model-spec improvement ADR ✅ | [05_validation/bayes_h5_model_spec_improvement_adr.md](05_validation/bayes_h5_model_spec_improvement_adr.md) — **Accepted**; [H5a–H5d archives](05_validation/archives/) · [H5e–H5r investigations](06_investigations/) · [AUDIT-H5P](audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md) (research only; prod blocked) |
| INV-H4-001 sparse pooling investigation | [06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md](06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md) |
| INV-H4-001b variant sweep JSON | [05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json](05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json) |
| Bayesian Geo MMM architecture refinement | [05_validation/bayesian_hierarchical_geo_mmm_refinement.md](05_validation/bayesian_hierarchical_geo_mmm_refinement.md) |
| Track D research — D5-POW SCM+JK readout | [track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md](track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md) (research lane; not production) |
| Investigation backlog | [06_investigations/open_investigations.md](06_investigations/open_investigations.md) |
| Investigation index (by track) | [06_investigations/investigation_index.md](06_investigations/investigation_index.md) |
| GroundTruthWorld contract | [05_validation/groundtruth_contract.md](05_validation/groundtruth_contract.md) |
| Validation registry | [05_validation/validation_registry.md](05_validation/validation_registry.md) |
| Architecture decisions (ADR) | [05_validation/synthetic_architecture_decisions.md](05_validation/synthetic_architecture_decisions.md) |
| **World bundle materialization (DR-01)** | [05_validation/world_materialization.md](05_validation/world_materialization.md) |
| **Truth versioning (DR-02)** | [05_validation/truth_versioning.md](05_validation/truth_versioning.md) |
| **World catalog spec** | [05_validation/world_catalog.md](05_validation/world_catalog.md) |
| **World truth schema (`world_truth.json`)** | [05_validation/world_schema.md](05_validation/world_schema.md) |
| **Bundle + catalog JSON schema** | [05_validation/world_bundle_schema.md](05_validation/world_bundle_schema.md) |
| **Validator specification (L1–L4)** | [05_validation/world_validator_spec.md](05_validation/world_validator_spec.md) |
| **Smoke world bundle** | `validation/worlds/WORLD-001-baseline/` |
| **Replay smoke world** | `validation/worlds/WORLD-002-replay/` |
| **Generated baseline world** | `validation/worlds/WORLD-003-generated-baseline/` |
| **Generated replay world** | `validation/worlds/WORLD-004-generated-replay/` |
| **Scenario: low noise** | `validation/worlds/WORLD-005-scenario-low-noise/` |
| **Scenario: high collinearity** | `validation/worlds/WORLD-006-scenario-high-collinearity/` |
| **Scenario: replay + drift** | `validation/worlds/WORLD-007-scenario-replay-drift/` |
| **Exact-recovery DGP world** | `validation/worlds/WORLD-008-exact-recovery/` |

## Run in production

Governance, safety checks, and operations:

| Topic | Doc |
|--------|-----|
| **v1.0.0 release notes** | [04_governance/v1_release_notes.md](04_governance/v1_release_notes.md) |
| Operator workflows | [04_governance/operator_runbook.md](04_governance/operator_runbook.md) |
| Prod safety checklist | [04_governance/prod_safety_checklist.md](04_governance/prod_safety_checklist.md) |
| Artifact & bundle fields | [04_governance/artifact_schema.md](04_governance/artifact_schema.md) |

## Investigations backlog

Living register of open gaps, design debt, and deferred work (evidence-linked; no fixes in-repo):

| Topic | Doc |
|--------|-----|
| **Full investigation records** | [06_investigations/open_investigations.md](06_investigations/open_investigations.md) |
| **Index by risk area** | [06_investigations/investigation_index.md](06_investigations/investigation_index.md) |

## Documentation maintenance

- [DOCUMENTATION_INVENTORY.md](DOCUMENTATION_INVENTORY.md) — canonical paths and deprecated redirects
- [documentation_truth_audit.md](documentation_truth_audit.md) — docs vs code audit
- Validate links: `python scripts/validate_docs.py`

## Archive

Historical / roadmap notes (not current operator guidance):

- [_archive/roadmap_causal_calibration_governance.md](_archive/roadmap_causal_calibration_governance.md)
