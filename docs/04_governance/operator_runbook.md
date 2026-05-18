# Operator runbook (planning & calibration)

## Before `optimize-budget`

1. Confirm YAML `data.path` points at the **same** panel schema used in training.
2. Attach `extension_report.json` with **`ridge_fit_summary`** (coef, intercept, best_params).
3. Run optimization safety gate expectations: governance section populated if your org requires it.
4. For **geo budgets**: set `budget.geo_budget_enabled` and validate `geo_*` constraints against feasible totals.

## Before `simulate` (CLI)

1. Scenario YAML: exactly one of `candidate_spend`, `candidate_spend_by_geo`, or `candidate_spend_path` (path may pair with `candidate_spend` for documentation).
2. **Control overlays**: each `(geo, week, column)` must match a panel row or the run fails closed.
3. For **posterior bands**: pass `uncertainty_mode: posterior` in API (CLI extension TBD); supply `linear_coef_draws` aligned with Ridge `coef` dimension and valid `bayesian_fit_meta`.

## Experiment / replay units

1. Generate `experiment_id` via `mmm.experiments.new_experiment_id()`; never reuse IDs for different payloads.
2. Sign canonical payloads with `sign_payload` using a managed secret; store signature on `ExperimentRecord`.
3. Move `approval` to `approved` only after human review; replay services should call `ExperimentRegistry.require_approved` or `experiment_readiness`.

## When things fail

- **PosteriorPlanningDisabled**: read `reasons` — usually missing draws, failed diagnostics, or prod without `posterior_planning_mode=draws`.
- **SLSQP budget failure**: relax bounds or check `total_budget` vs sum of mins/maxes.
