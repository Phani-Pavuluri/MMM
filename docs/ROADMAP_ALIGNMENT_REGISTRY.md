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
| Bayes-H5d TrustReport diagnostic mapping (INV-H5D) | 2 | Reliability / governance | TrustReport (diagnostic) | prod TrustReport wiring without Promotion Gate | [INV-H5D](06_investigations/INV-H5D_TRUST_DIAGNOSTIC_MAPPING.md); [H5d mapping JSON](05_validation/archives/BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json) | Investigation | **Complete** | Prod TrustReport integration (blocked) | **Yes** | Blocked — research only |
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
