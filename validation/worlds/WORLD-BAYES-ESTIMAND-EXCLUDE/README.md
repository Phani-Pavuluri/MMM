# WORLD-BAYES-ESTIMAND-EXCLUDE

**Validation family:** `bayes-hierarchy-evidence` (no-fit contract validation only)

This bundle validates **CalibrationSignal routing**, hierarchy propagation, inclusion/exclusion,
conflict surfacing, and TrustReport obligations per Bayes-H2b ADRs. It does **not** train models,
run PyMC, compute posteriors, recover coefficients, run optimizers, or compute DecisionSurface Δμ.

## Files

| File | Role |
|------|------|
| `hierarchy_spec.json` | Shared 6-DMA scope graph |
| `calibration_signals.json` | Evidence ingress fixtures |
| `estimand_allowlist.json` | Estimand gate for validator |
| `hierarchy_evidence_fixture.json` | Expected routing / TrustReport / gate outcomes |

## Source docs

- `docs/BAYES_H2B_VALIDATION_WORLDS_001.md`
- `docs/BAYES_H2B_VALIDATION_RUNNER_002.md`
- `docs/05_validation/bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md`

## Assertions

Canonical: VAL-BAYES-001, VAL-BAYES-002, VAL-BAYES-003, VAL-BAYES-004, VAL-BAYES-005, VAL-BAYES-006, …
