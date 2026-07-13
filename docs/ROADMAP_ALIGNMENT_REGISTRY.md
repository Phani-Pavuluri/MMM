# Roadmap Alignment Registry

**Status:** Active (living register)  
**Policy:** [ROADMAP_ALIGNMENT_GATE.md](ROADMAP_ALIGNMENT_GATE.md) · **Audit:** [MIP_PLATFORM_AUDIT_TEMPLATE.md](MIP_PLATFORM_AUDIT_TEMPLATE.md)  
**Scope:** Bayes-H2b / H2d / H3 path (MMM research sandbox) — extend with new rows as phases land  
**Last updated:** 2026-06-01 (post [phase audit](audits/MIP_PLATFORM_AUDIT_20260601_PRE_BAYES_H3.md))

---

## How to use

Each row is a **check-and-balance** against the alignment gate:

- **Research allowed?** — exploration permitted when labeled `RESEARCH ONLY — NOT DECISION GRADE`
- **Production promotion status** — whether the item may affect prod decisioning, optimizers, or release artifacts

**Operational principle:** Research allowed by default. Production promotion gated by default.

Do not add rows without tier, gate, proof artifact, and next authorized step.

---

## Bayes hierarchy evidence & Bayesian sandbox (current path)

| Roadmap item | Tier | MIP goal | Contract touched | Failure mode reduced | Proof artifact | Gate level | Status | Next authorized step | Research allowed? | Production promotion status |
|---|---:|---|---|---|---|---|---|---|---|---|
| [Bayes-H2b ADR](05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) | 1 | Trust-aware measurement; experiment-informed decisioning | CalibrationSignal, Estimand, TrustReport, Release Gates | scope drift; local→national point mass; silent conflict merge | ADR accepted | Architecture | **Accepted** | validation worlds + runner contract | Yes (architecture) | Not applicable — policy only |
| [Bayes-H2b validation worlds](BAYES_H2B_VALIDATION_WORLDS_001.md) | 1 | Reliability / governance; trust-aware measurement | CalibrationSignal, TrustReport | stale evidence; missing SE; estimand mismatch; conflicts | `WORLD-BAYES-*` specs (7) | Architecture | **Accepted** | fixture materialization | Yes (spec) | Blocked — specs not prod artifacts |
| [Bayes-H2b validation runner contract](BAYES_H2B_VALIDATION_RUNNER_002.md) | 1 | Reliability / governance | CalibrationSignal, TrustReport, Release Gates | unenforceable evidence-routing contract | RUNNER_002 + `VAL-BAYES-001`–`012` | Architecture | **Accepted** | fixture bundles + validator | Yes (contract) | Blocked — contract not implementation |
| Bayes-H2b fixture bundles | 2 | Reliability / governance | CalibrationSignal, TrustReport | fixture drift; missing negative cases | `validation/worlds/WORLD-BAYES-*/` (5 files × 7 worlds) | Implementation | **Complete** | validator stub + smoke | Yes (fixtures) | Blocked — not decision-grade |
| `hierarchy_evidence_validator` stub | 2 | Reliability / governance | CalibrationSignal, TrustReport, Release Gates | fixture contract not enforceable; silent promotion of bad evidence routing | `mmm.validation.synthetic.hierarchy_evidence_validator`; pytest | Implementation | **Complete** | Bayes-H2d architecture ADR | Yes (validator is no-fit) | Blocked — does not authorize prod Bayesian |
| `VAL-BAYES-H2B-SMOKE` | 2 | Reliability / governance | CalibrationSignal, TrustReport | seven-world contract regression | `validate_world_catalog`; CLI `--smoke VAL-BAYES-H2B-SMOKE` | Implementation | **Complete** | Bayes-H2d architecture ADR | Yes (CI smoke) | Blocked — smoke ≠ prod release |
| [Bayes-H2d model spec ADR](05_validation/bayes_h2d_hierarchical_model_spec_adr.md) | 1 | MMM calibration ecosystem | DecisionSurface, CalibrationSignal, TrustReport, Estimand | premature PyMC; posterior-as-decision; ABI drift | Bayes-H2d ADR accepted | Architecture | **Accepted** | Bayes-H3 research sandbox only | Yes (architecture-only) | Blocked — spec not implementation |
| Bayes-H3 sandbox guardrails (P0 audit) | 2 | Reliability / governance | (sandbox labels) | prod optimizer/decide misuse; unlabeled artifacts | `mmm/research/bayes_h3_sandbox/`; `tests/research/test_bayes_h3_sandbox_guardrails.py`; CI smoke + guardrails in `.github/workflows/ci.yml` | Implementation | **Complete** | Bayes-H3 sandbox **fit** work (PyMC prototype) | **Yes** — guardrails only | Blocked — fences ≠ prod Bayesian |
| [Bayes-H3 sandbox backend ADR](05_validation/bayes_h3_research_sandbox_backend_adr.md) | 1 | MMM calibration ecosystem | TrustReport (diagnostic) | premature backend churn; dual-stack before recovery | PyMC initial; NumPyro deferred | Architecture | **Accepted** | Bayes-H4 recovery worlds | Yes (sandbox backend policy) | Blocked — backend ≠ prod |
| Bayes-H3 research sandbox MVP fit | 2 | MMM calibration ecosystem | (sandbox — maps to ABI on promotion) | ungrounded algorithm change | `mmm/research/bayes_h3_sandbox/model.py`; `run_sandbox_fit`; `tests/research/test_bayes_h3_sandbox_mvp_fit.py` | Implementation + sandbox | **Complete** | Bayes-H4 recovery worlds | **Yes** — diagnostic hierarchical fit only | Blocked — not decision-grade |
| [Bayes-H4 recovery worlds ADR](05_validation/bayes_h4_recovery_worlds_adr.md) | 1 | Reliability / governance | TrustReport (diagnostic) | unfounded prod promotion; unmeasured recovery | H4 ADR + `recovery_worlds` / `recovery_runner` | Architecture | **Accepted** | H4 pilot thresholds + slow CI optional | Yes (validation spec) | Blocked — recovery ≠ prod |
| Bayes-H4 recovery world scaffolding | 2 | Reliability / governance | TrustReport (diagnostic) | sandbox correctness unproven | `WORLD-BAYES-H4-*`; `tests/research/test_bayes_h4_recovery_worlds.py` | Implementation | **Complete** (scaffolding) | H4a threshold pilot | **Yes** — metrics report-only | Blocked — not decision-grade |
| Bayes-H4a recovery threshold pilot | 2 | Reliability / governance | TrustReport (diagnostic) | uncalibrated recovery gates | `h4_threshold_pilot.py`; `archives/BAYES_H4_THRESHOLD_PILOT_20260601.json` | Implementation | **Complete** (provisional bands) | H4b repeated pilot | **Yes** — report-only thresholds | Blocked — pilot ≠ prod |
| Bayes-H4b repeated recovery pilot | 2 | Reliability / governance | TrustReport (diagnostic) | misread fast-MCMC shrinkage as model readiness | `h4_repeated_pilot.py`; `archives/BAYES_H4_REPEATED_PILOT_20260601.json` | Implementation | **Complete** (superseded for pooling) | H4b-refresh primary metric | **Yes** | Blocked |
| Bayes-H4b-refresh primary-metric repeated pilot | 2 | Reliability / governance | TrustReport (diagnostic) | legacy metric overstated sparse failure | [PRIMARY_METRIC JSON](05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json) | Implementation | **Complete** (pooling mechanics) | H4b-disposition | **Yes** | Blocked — not true-effect recovery proof |
| Bayes-H4b-disposition (metric + recovery posture) | 2 | Reliability / governance | TrustReport (diagnostic) | conflate pooling with truth recovery | [INV-H4-001 §11](06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md) | Governance | **Complete** (C+A accepted) | Bayes-H4c research worlds | **Yes** | Blocked — disposition ≠ prod |
| INV-H4-001 sparse pooling behavior | 2 | Reliability / governance | TrustReport (diagnostic) | wrong shrinkage metric; weak pooling on sparse geo | [INV-H4-001](06_investigations/INV-H4-001_SPARSE_POOLING_BEHAVIOR.md) | Investigation | **Closed** (disposition C+A) | sparse world / τ tuning (research) | **Yes** | Blocked — no production promotion |
| INV-H4-001b sparse variant sweep | 2 | Reliability / governance | TrustReport (diagnostic) | mis-tuned sparse world; τ prior | [sweep JSON](05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json) | Investigation | **Closed** | — | **Yes** | Blocked |
| Bayes-H4c extended recovery worlds | 2 | Reliability / governance | TrustReport (diagnostic) | premature hard gates; prod promotion | `h4c_recovery_worlds.py`; [H4C pilot JSON](05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json) | Implementation | **Complete** (reliability map) | sparse/τ tuning (research) | **Yes** | Blocked — not production ready |
| INV-071 true-effect recovery threshold policy | 2 | Reliability / governance | TrustReport (diagnostic) | global thresholds; stress worlds as hard fail | [INV-071](06_investigations/INV-071_BAYES_H4_TRUE_EFFECT_RECOVERY_THRESHOLDS.md); [policy JSON](05_validation/archives/BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json) | Investigation | **Complete** (report-only) | H4d stability pilot | **Yes** | Blocked — no production promotion |
| Bayes-H4d sparse/τ stability (INV-H4D) | 2 | Reliability / governance | TrustReport (diagnostic) | conflate sparse recovery with sparse stress; prod τ promotion | [INV-H4D](06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md); [H4D fast](05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_20260601.json); [H4D extended](05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_EXTENDED_20260601.json) | Investigation | **Complete** (fast + extended confirmed) | Bayes-H5 model-spec ADR | **Yes** | Blocked — sandbox τ only |
| [Bayes-H5 model-spec improvement ADR](05_validation/bayes_h5_model_spec_improvement_adr.md) | 1 | MMM calibration ecosystem | TrustReport (diagnostic) | transform mismatch; weak ID; false prod promotion | H5 ADR accepted 2026-06-01; H4c/H4d evidence | Architecture | **Accepted** | H5 pilot + spec validation | Yes (architecture-only) | Blocked — spec not implementation |
| Bayes-H5a sandbox validation worlds + gated fit | 2 | Reliability / governance | TrustReport (diagnostic) | prod promotion from sandbox pilot | `h5_validation_worlds.py`; [H5 pilot JSON](05_validation/archives/BAYES_H5_SANDBOX_PILOT_20260601.json) (fast MCMC ✅) | Implementation | **Complete (H5a)** | H5c extended MCMC (optional) | **Yes** | Blocked — research only |
| Bayes-H5b diagnostic polish + repeated pilot (INV-H5B) | 2 | Reliability / governance | TrustReport (diagnostic) | false promotion from pilot variance | [INV-H5B](06_investigations/INV-H5B_REPEATED_PILOT_AND_DIAGNOSTICS.md); [H5b repeated JSON](05_validation/archives/BAYES_H5B_REPEATED_PILOT_20260601.json) | Investigation | **Complete** | H5c extended MCMC | **Yes** | Blocked — research only |
| Bayes-H5c extended MCMC confirmation (INV-H5C) | 2 | Reliability / governance | TrustReport (diagnostic) | overfit fast MCMC conclusions | [INV-H5C](06_investigations/INV-H5C_EXTENDED_MCMC_CONFIRMATION.md); [H5c extended JSON](05_validation/archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json) | Investigation | **Complete** | H5d diagnostic mapping | **Yes** | Blocked — research only |
| Bayes-H5d TrustReport diagnostic mapping (INV-H5D) | 2 | Reliability / governance | TrustReport (diagnostic) | prod TrustReport wiring without Promotion Gate | [INV-H5D](06_investigations/INV-H5D_TRUST_DIAGNOSTIC_MAPPING.md); [H5d mapping JSON](05_validation/archives/BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json) | Investigation | **Complete** | H5e shadow protocol | **Yes** | Blocked — research only |
| Bayes-H5e real-panel shadow-run protocol (INV-H5E) | 2 | Reliability / governance | TrustReport (diagnostic) | shadow output into prod optimizer | [INV-H5E](06_investigations/INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md); [H5e schema JSON](05_validation/archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json) | Investigation | **Protocol defined** | H5f harness (research execution) | **Yes** | Blocked — design only |
| Bayes-H5f shadow-run harness (INV-H5F) | 2 | Reliability / governance | TrustReport (diagnostic) | prod wiring / optimizer from shadow JSON | [INV-H5F](06_investigations/INV-H5F_REAL_PANEL_SHADOW_RUN_HARNESS.md); [H5f dry-run JSON](05_validation/archives/BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json) | Investigation | **Complete (research)** | H5g first real-panel shadow | **Yes** | Blocked — research only |
| Bayes-H5g first real-panel shadow (INV-H5G) | 2 | Reliability / governance | TrustReport (diagnostic) | treating shadow JSON as decision grade | [INV-H5G](06_investigations/INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md); [H5g artifact JSON](05_validation/archives/BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5h shadow hardening | **Yes** | Blocked — illustrative panel |
| Bayes-H5h real-panel shadow hardening (INV-H5H) | 2 | Reliability / governance | TrustReport (diagnostic) | batching panels before convergence gates | [INV-H5H](06_investigations/INV-H5H_REAL_PANEL_SHADOW_HARDENING.md); [H5h artifact JSON](05_validation/archives/BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5i convergence diagnostics | **Yes** | Blocked — failed_convergence |
| Bayes-H5i real-panel convergence diagnostics (INV-H5I) | 2 | Reliability / governance | TrustReport (diagnostic) | more panels before pilot converges | [INV-H5I](06_investigations/INV-H5I_REAL_PANEL_CONVERGENCE_DIAGNOSTICS.md); [H5I matrix JSON](05_validation/archives/BAYES_H5I_CONVERGENCE_EXPERIMENT_MATRIX_20260601.json) | Investigation | **Complete (research)** | H5j collinearity ablations | **Yes** | Blocked — collinearity |
| Bayes-H5j collinearity geometry ablations (INV-H5J) | 2 | Reliability / governance | TrustReport (diagnostic) | batching panels before weak/converged pilot | [INV-H5J](06_investigations/INV-H5J_COLLINEARITY_GEOMETRY_ABLATIONS.md); [H5J ablations JSON](05_validation/archives/BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json) | Investigation | **Complete (research)** | H5k geometry stabilization | **Yes** | Blocked — weak_convergence max |
| Bayes-H5k geometry stabilization (INV-H5K) | 2 | Reliability / governance | TrustReport (diagnostic) | prod Bayes without hierarchy-faithful pass | [INV-H5K](06_investigations/INV-H5K_GEOMETRY_STABILIZATION.md); [H5K JSON](05_validation/archives/BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json) | Investigation | **Complete (research)** | H5l hierarchy-faithful refinement | **Yes** | Blocked — pooled/fixed-τ diagnostic only |
| Bayes-H5l hierarchy geometry refinement (INV-H5L) | 2 | Reliability / governance | TrustReport (diagnostic) | real-panel batch before faithful convergence | [INV-H5L](06_investigations/INV-H5L_HIERARCHY_GEOMETRY_REFINEMENT.md); [H5L JSON](05_validation/archives/BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json) | Investigation | **Complete (research)** | H5m frozen policy replay | **Yes** | Blocked — see artifact |
| Bayes-H5m frozen shadow-policy replay (INV-H5M) | 2 | Reliability / governance | TrustReport (diagnostic) | second real panel before governed replay | [INV-H5M](06_investigations/INV-H5M_FROZEN_SHADOW_POLICY_REPLAY.md); [H5M replay JSON](05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5n shadow-policy recommender | **Yes** | Blocked |
| Bayes-H5n shadow-policy recommender (INV-H5N) | 2 | Reliability / governance | TrustReport (diagnostic) | treating recommender as prod decisioning; silent channel collapse | [INV-H5N](06_investigations/INV-H5N_SHADOW_POLICY_RECOMMENDER.md); [H5N recommendation JSON](05_validation/archives/BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5o second real panel | **Yes** | Blocked — research only |
| Bayes-H5o second real-panel shadow (INV-H5O) | 2 | Reliability / governance | TrustReport (diagnostic) | batching panels; forcing run on do_not_run | [INV-H5O](06_investigations/INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md); [H5O shadow JSON](05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5p workflow audit gate | **Yes** | Blocked — one panel only |
| Bayes-H5p shadow workflow audit gate (AUDIT-H5P) | 2 | Reliability / governance | TrustReport (diagnostic) | treating two-panel pilot as prod-ready; batching without manifest | [AUDIT-H5P](audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md); [H5 ADR § H5p](05_validation/bayes_h5_model_spec_improvement_adr.md#h5p-shadow-workflow-audit-gate-audit-h5p) | Audit | **Complete (research)** | H5q third panel | **Yes** | Blocked — expansion criteria only |
| Bayes-H5q third real-panel shadow (INV-H5Q) | 2 | Reliability / governance | TrustReport (diagnostic) | batching; forcing run on do_not_run; treating failed convergence as promotable | [INV-H5Q](06_investigations/INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md); [H5Q shadow JSON](05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H5r sparse-channel remedy | **Yes** | Blocked — failed_convergence |
| Bayes-H5r sparse-channel remedy replay (INV-H5R) | 2 | Reliability / governance | TrustReport (diagnostic) | new panel instead of remedy; silent sparse drop | [INV-H5R](06_investigations/INV-H5R_SPARSE_CHANNEL_REMEDY_REPLAY.md); [H5R comparison JSON](05_validation/archives/BAYES_H5R_REMEDY_COMPARISON_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json) | Investigation | **Complete (research)** | H6 synthetic lane | **Yes** | Blocked — research only |
| Bayes-H6 synthetic validation lane (H6a–H6f) | 2 | Reliability / governance | TrustReport (diagnostic) | toy-only Bayes validation; no Ridge comparison | [H6 ADR](05_validation/bayes_h6_synthetic_lane_adr.md); [H6F matrix JSON](05_validation/archives/BAYES_H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX_20260601.json) | Implementation | **Complete (research)** | H7 Ridge production diagnostics | **Yes** | Blocked — synthetic ≠ real proof |
| H7 Ridge production diagnostic hardening | 2 | Reliability / governance; decision safety | TrustReport (diagnostic) | silent missing controls; no transform reporting on prod path | [Ridge diagnostics contract](05_validation/ridge_production_diagnostics_contract.md); `mmm/diagnostics/ridge_diagnostics.py`; [sample JSON](05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json) | Implementation | **Complete** | H8 operator surfacing | **Yes** (metadata only) | Allowed — diagnostics only; not hard gates |
| H8 Ridge diagnostics operator surfacing | 2 | Reliability / governance; decision safety | TrustReport (diagnostic) | analysts miss forbidden claims in JSON-only output | `ridge_diagnostic_summary.py`; train artifacts + `mmm train` CLI block; [summary MD](05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_SUMMARY_20260601.json) | Implementation | **Complete** | H9 severity policy | **Yes** (UX only) | Allowed — no optimizer/DecisionSurface changes |
| H9 Ridge diagnostic severity policy | 2 | Reliability / governance; decision safety | TrustReport (diagnostic) | inconsistent analyst interpretation of warnings | [severity policy](05_validation/ridge_diagnostic_severity_policy.md); `ridge_severity_policy.py` | Implementation | **Complete** | H10 E2E audit | **Yes** (policy only) | Allowed — output eligibility; not automatic gates |
| H10 Ridge diagnostic E2E audit (AUDIT-H10) | 2 | Reliability / governance; decision safety | TrustReport (diagnostic) | undocumented chain breaks between H7–H9 | [AUDIT-H10](audits/AUDIT-H10_RIDGE_DIAGNOSTIC_E2E_GATE.md); `test_ridge_diagnostic_e2e_audit.py` | Audit | **Pass** | CalibrationSignal integration audit | **Yes** (verification) | Allowed — no fitting/optimizer changes |
| MIP-C1 CalibrationSignal → MMM diagnostic attachment (AUDIT-MIP-C1) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | external evidence overrides MMM; silent GeoX merge | [AUDIT-MIP-C1](audits/AUDIT-MIP-C1_CALIBRATIONSIGNAL_MMM_INTEGRATION_GATE.md); [attachment contract](05_validation/calibration_signal_mmm_diagnostic_attachment_contract.md); `test_calibration_signal_mmm_attachment_contract.py` | Audit | **Pass (contract)** | H11 real-bundle hardening | **Yes** (context-only) | Allowed — no coef/optimizer/DecisionSurface changes |
| H11 Real-bundle Ridge diagnostic hardening (INV-H11) | 2 | Reliability / governance; decision safety | TrustReport (diagnostic) | diagnostics fail on real bundles; silent missing metadata | [INV-H11](06_investigations/INV-H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_HARDENING.md); [H11 manifest](06_investigations/H11_REAL_BUNDLE_RIDGE_DIAGNOSTIC_MANIFEST.md); `ridge_real_bundle_hardening.py`; [H11 archive JSON](05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) | Investigation | **Complete** | MIP-C2 train-boundary wiring | **Yes** (diagnostics only) | Allowed — illustrative panel; not client prod sign-off |
| MIP-C2 CalibrationSignal train-boundary wiring (AUDIT-MIP-C2) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | signals fed into refit/optimizer; silent override | [AUDIT-MIP-C2](audits/AUDIT-MIP-C2_CALIBRATIONSIGNAL_TRAIN_BOUNDARY_WIRING.md); `calibration_signal_ingestion.py`; `test_calibration_signal_train_boundary_ingestion.py` | Audit | **Pass (wiring)** | GeoX/CLS adapter (MIP-C3) | **Yes** (context-only) | Allowed — file path / YAML only; not prod approval |
| MIP-C3 GeoX/CLS CalibrationSignal adapter (AUDIT-MIP-C3) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | live API before contract; export shape drift | [AUDIT-MIP-C3](audits/AUDIT-MIP-C3_GEOX_CLS_SIGNAL_ADAPTER_GATE.md); [adapter contract](05_validation/geox_cls_to_calibration_signal_adapter_contract.md); `calibration_signal_adapters.py` | Audit | **Pass (adapter)** | ETL dry-run (MIP-C4) | **Yes** (export-only) | Allowed — no live API; not prod approval |
| MIP-C4 CalibrationSignal ETL dry-run (AUDIT-MIP-C4) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | prod scheduler before dry-run proof | [AUDIT-MIP-C4](audits/AUDIT-MIP-C4_CALIBRATIONSIGNAL_ETL_DRY_RUN.md); `calibration_signal_etl.py`; [dry-run JSON](05_validation/archives/MIP_C4_DRY_RUN_CALIBRATION_SIGNALS_20260601.json); [train proof](05_validation/archives/MIP_C4_TRAIN_WITH_DRY_RUN_SIGNALS_20260601.json) | Audit | **Pass (dry-run)** | MIP-C5 scheduled ETL wrapper | **Yes** (artifact + train proof) | Allowed — no scheduler; not prod approval |
| MIP-C5 Scheduled CalibrationSignal ETL wrapper (AUDIT-MIP-C5) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | live cron before drop-zone proof | [AUDIT-MIP-C5](audits/AUDIT-MIP-C5_CALIBRATIONSIGNAL_SCHEDULED_ETL_WRAPPER.md); `calibration_signal_etl_job.py`; [manifest](05_validation/archives/mip_c5_etl_outputs/MIP_C5_DRY_RUN_20260601_manifest.json) | Audit | **Pass (drop-zone)** | MIP-C6 integration readiness checkpoint | **Yes** (batch + train proof) | Allowed — no live API; not prod deployment |
| MIP-C6 MIP integration readiness checkpoint (AUDIT-MIP-C6) | 2 | MMM calibration ecosystem; decision safety | CalibrationSignal, TrustReport | prod scheduler/API before bridge audit; silent decision promotion | [AUDIT-MIP-C6](audits/AUDIT-MIP-C6_INTEGRATION_READINESS_CHECKPOINT.md) | Audit | **Pass (pause before live scheduler)** | GeoX estimator/inference OC (recommended) or C6 governance if urgent | **Yes** (roadmap only) | Allowed — audit/docs only; live scheduler/API blocked |
| Future MMM package-side support agents (roadmap) | 5 | Orchestration compatibility; decision safety | TrustReport, run manifests | agents before typed diagnostics; agents as modeling authority | [mmm_package_side_agents_roadmap.md](05_validation/mmm_package_side_agents_roadmap.md); [INV-039](06_investigations/open_investigations.md#inv-039--auto-retrain-auto-promotion-agentic-orchestration-out-of-v1-scope) | Roadmap | **Deferred** | MMMRunManifest + MMMFailurePacket contracts; MIP agent contracts | **No** | Blocked — docs-only until prerequisites §4 complete |
| MMM-EXPORT-001 MMM→MIP export contract inventory | 2 | MMM calibration ecosystem; decision safety | DecisionSurface, TrustReport, estimand exports | MIP exposes ROI/recs from internal JSON; silent claim promotion | [mmm_to_mip_export_contract_inventory.md](05_validation/mmm_to_mip_export_contract_inventory.md); INV-MMM-EXPORT-CONTRACTS-001 | Contract inventory | **Complete (docs)** | MMM-EXPORT-002 typed schemas + fixture bundle | **Yes** (boundary only) | Allowed — inventory only; no MIP-consumable exports yet |
| MMM-EXPORT-002 Typed MMM export schemas + fixtures | 2 | MMM calibration ecosystem; decision safety | DecisionSurface, TrustReport | treating fixtures as production truth; ROI without uncertainty | [mmm_export_schema_and_fixture_contract.md](05_validation/mmm_export_schema_and_fixture_contract.md); `mmm/contracts/mip_export.py`; `tests/fixtures/mip_export/` | Schema/fixture | **Complete** | MMM-EXPORT-003 runtime adapter | **Yes** (fixtures only) | Allowed — synthetic fixtures; production claims blocked |
| Bayes-H3 production promotion | 3 | MMM calibration ecosystem; budget optimization | DecisionSurface, CalibrationSignal, TrustReport, Release Gates | posterior→optimizer; coef planning; missing TrustReport | H5 validation + Promotion Gate; decision trace | Promotion | **Blocked** | not until H5 implementation + reproducible Δμ | Yes in sandbox only | **Blocked** — full promotion chain required |

---

## Dependency chain (authorization order)

```text
Bayes-H2b ADR ✅
  → validation worlds ✅
  → runner contract ✅
  → fixture bundles ✅
  → hierarchy_evidence_validator ✅
  → VAL-BAYES-H2B-SMOKE ✅
  → Bayes-H2d model spec ADR ✅
  → Bayes-H3 sandbox guardrails (P0) ✅
  → Bayes-H3 research sandbox MVP fit ✅
  → Bayes-H4 recovery worlds ADR + scaffolding ✅
  → Bayes-H4a threshold pilot (provisional JSON) ✅
  → Bayes-H4b repeated recovery pilot (extended JSON) ✅
  → INV-H4-001 / 001b (metric + variant sweep) ✅
  → H4b-refresh primary-metric repeated pilot ✅
  → H4b-disposition (C+A accepted) ✅
  → Bayes-H4c extended recovery worlds (reliability map) ✅
  → INV-071 true-effect threshold calibration (report-only policy) ✅
  → Bayes-H4d sparse/τ stability pilot (INV-H4D) ✅
  → Bayes-H5 model-spec improvement ADR ✅
  → Bayes-H5a sandbox validation worlds + gated fit ✅ (fast MCMC pilot)
  → Bayes-H5b diagnostic polish + repeated pilot ✅ (INV-H5B)
  → Bayes-H5c extended MCMC confirmation ✅ (INV-H5C)
  → Bayes-H5d TrustReport diagnostic mapping ✅ (INV-H5D)
  → Bayes-H5e real-panel shadow-run protocol ✅ (design only)
  → Bayes-H5f shadow-run harness ✅ (research execution; prod blocked)
  → Bayes-H5g first real-panel shadow ✅ (examples sample panel; prod blocked)
  → Bayes-H5h shadow hardening ✅ (diagnostics + extended MCMC; evidence still blocked)
  → Bayes-H5i convergence diagnostics ✅ (sample panel matrix; do not batch panels)
  → Bayes-H5j collinearity geometry ablations ✅ (explicit channel_policy; pilot weak_convergence max)
  → Bayes-H5k geometry stabilization ✅ (ablation vs faithful hierarchy)
  → Bayes-H5l hierarchy-faithful refinement ✅ (H5L-B σ_floor + full pooling)
  → Bayes-H5m frozen shadow-policy replay ✅ (governed policy JSON + `--policy-path`)
  → Bayes-H5n shadow-policy recommender ✅ (diagnostics → governed policy + forbidden claims)
  → Bayes-H5o second real-panel shadow ✅ (benchmark_geo_panel_v1; keep-all policy; converged)
  → Bayes-H5p shadow workflow audit gate ✅ (AUDIT-H5P — eligibility, stops, expansion criteria)
  → Bayes-H5q third real-panel shadow ✅ (triangulation panel; workflow pass; failed_convergence recorded)
  → Bayes-H5r sparse-channel remedy ✅ (same panel; drop radio → converged)
  → Further panels / recommender updates — per AUDIT-H5P; do not batch
  → Production TrustReport integration ← blocked (Promotion Gate required)
  → Bayes-H5 production promotion (blocked)
  → Bayes-H3 production promotion (blocked)
```

---

## Explicit non-goals (current path)

| Item | Does not authorize |
|------|-------------------|
| Entire H2b/H2d track through smoke | PyMC, samplers, posterior decisioning, prod optimizer changes |
| Bayes-H2d (when started) | Implementation, production release, new CalibrationSignal ingress without ADR |
| Bayes-H3 research sandbox | Production recommendations, prod DecisionSurface, release without Promotion Gate |
| Bayes-H3 production promotion | Any bypass of TrustReport, Release Gates, or full-panel Δμ |

---

## Adding rows

Copy the table header from [ROADMAP_ALIGNMENT_GATE.md § Roadmap traceability table](ROADMAP_ALIGNMENT_GATE.md#roadmap-traceability-table). New Tier 1–3 items **must** include research vs. production promotion columns.
