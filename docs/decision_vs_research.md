# Decision vs research use of MMM outputs

## Decision-grade (production-oriented)

- **Full-panel `simulate()`** with BAU baseline, `run_environment=prod`, governance flags satisfied, and `extension_report` + panel aligned to training.
- **Budget optimization** only with `allow_unsafe_decision_apis` + CLI flags where required, optimization safety gate passed, and **ridge_fit_summary** present for full-model scoring.
- **Posterior P10/P50/P90 on Δμ** only when `bayesian_fit_meta` reports `posterior_diagnostics_ok` and `posterior_predictive_ok`, `linear_coef_draws` are supplied, and in **prod** `extensions.product.posterior_planning_mode=draws`.
- **Risk-aware optimization** (`optimize_budget_risk_aware`) uses the same gates as posterior simulation.

## Research / exploratory

- Legacy **curve-local** optimizers, diagnostic-only scenarios, and **Ridge** runs without full governance artifacts.
- **Posterior draws** in non-prod without `posterior_planning_mode=draws` may still compute quantiles when diagnostics pass, but disclosures will flag configuration gaps.
- **Experiment registry** entries in `draft` are not replay-ready; use `experiment_readiness()` before wiring to services.

## Rule of thumb

If a downstream system spends money or sets public targets from the model, treat the path as **decision** and require the stricter checklist in `prod_safety_checklist.md`.
