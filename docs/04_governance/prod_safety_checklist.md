# Production safety checklist

Use this before enabling automated spend recommendations or publishing model-based targets.

## Data & fit

- [ ] Panel fingerprint matches training artifact (or documented exception).
- [ ] `run_environment: prod` set explicitly in config.
- [ ] No silent fallback from full-model to curve-only optimizer in prod.

## Governance

- [ ] `extension_report` governance / response diagnostics meet org thresholds (see `OptimizationSafetyGate`).
- [ ] Non-BAU baselines documented in optimization disclosure strings.

## Uncertainty & posterior planning

- [ ] If using **P10/P50/P90** or **risk-aware** optimization: `posterior_diagnostics_ok` and `posterior_predictive_ok` are true in `bayesian_fit_meta`.
- [ ] `extensions.product.posterior_planning_mode: draws` in prod whenever `linear_coef_draws` are consumed.
- [ ] Coef draws are generated from a documented process (e.g. Bayesian export or approved bootstrap), not ad-hoc noise.

## Experiments & replay

- [ ] `experiment_id` is UUID and immutable in downstream stores.
- [ ] `experiment_readiness` returns `ready: true` before replay (approved, signed, calibration ref present).
- [ ] Calibration artifact version pinned and traceable in metadata.

## Economics metadata

- [ ] Surfaces exposed to business users carry a complete `economics_output_metadata` block: `economics_version`, `economics_contract_version`, `surface`, `uncertainty_mode`, `computation_mode` (`exact` / `approximate` / `unknown`), `baseline_type` (not `unspecified` for simulation paths), `decision_safe` (bool for full-model simulation), and KPI column fields.
- [ ] Call `validate_business_economics_metadata` (or rely on `build_decision_bundle` / extension runner) so missing keys fail closed in CI or release gates.
