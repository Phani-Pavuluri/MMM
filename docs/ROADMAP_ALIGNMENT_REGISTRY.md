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
| Bayes-H3 research sandbox MVP fit | 2 | MMM calibration ecosystem | (sandbox — maps to ABI on promotion) | ungrounded algorithm change | `mmm/research/bayes_h3_sandbox/model.py`; `run_sandbox_fit`; `tests/research/test_bayes_h3_sandbox_mvp_fit.py` | Implementation + sandbox | **Complete** | Bayes-H4 recovery worlds | **Yes** — diagnostic hierarchical fit only | Blocked — not decision-grade |
| Bayes-H3 production promotion | 3 | MMM calibration ecosystem; budget optimization | DecisionSurface, CalibrationSignal, TrustReport, Release Gates | posterior→optimizer; coef planning; missing TrustReport | Bayes-H4+ worlds; Promotion Gate; decision trace | Promotion | **Blocked** | not until H2d + H4 gates + reproducible Δμ | Yes in sandbox only | **Blocked** — full promotion chain required |

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
  → Bayes-H4 recovery worlds  ← NEXT
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
