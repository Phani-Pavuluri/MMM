# Bayesian framework

- Backend: **PyMC** (default). **Stan** wired as optional `StanMMMTrainer` stub pending packaged `.stan` models.
- Pooling: partial pooling uses non-centered hierarchical shrinkage on channel coefficients per geo; full pooling shares media slopes; none fits independent positive slopes per geo.
- Priors: HalfNormal media effects for identifiability; weakly informative intercepts and noise.
- Diagnostics: ArviZ summaries (`r_hat`, `ess_bulk`, divergences).
- Calibration: extend likelihood with matched experiment terms (see `calibration/`).

Prefer Bayesian when sample size per geo is moderate, priors are defensible, and posterior uncertainty is required for decisioning.

## Production checklist (decision path)

1. **Diagnostics:** `compute_bayesian_decision_diagnostics` must summarize **media, controls, intercepts / hierarchy**, and `sigma` — not `sigma` alone. Thresholds come from `extensions.governance` (`bayesian_max_rhat`, `bayesian_min_ess_bulk`, `bayesian_max_divergences`).
2. **PPC:** Run `posterior_predictive` in training (`bayesian.posterior_predictive_draws > 0`). The PPC artifact (`build_bayesian_predictive_artifact`) records `mean_abs_gap`, optional **empirical 90% coverage** vs `y_obs` (modeling scale), and `std_ratio_pp_over_obs`. When `bayesian_max_mean_abs_ppc_gap` is set, `posterior_predictive_ok` requires that gap; otherwise a substantive PPC summary (gap or coverage) is still required for `posterior_predictive_ok`.
3. **Transforms:** `extensions.product.bayesian_decision_transform_stance` defaults to `fixed_yaml_features_labeled` (YAML `transforms.*_params` — not Ridge+BO joint hyperparameter search). Cross-framework Ridge vs Bayesian comparability remains **explicitly non-automatic**; see `transform_policy.build_transform_policy_manifest`.
4. **Posterior planning draws:** For `pooling=full`, `BayesianMMMTrainer.fit` returns `linear_coef_draws` plus `ppc["linear_coef_draws_export"]` when export succeeds. For **partial / none pooling**, use `hierarchical_draw_pack` from `fit_out["hierarchical_draw_pack"]` (see `hierarchical_coefficient_draws_from_pymc_idata`) with `delta_mu_draws_hierarchical_geo_beta` / `simulate(..., hierarchical_draw_pack=...)` — not the global `linear_coef_draws` path.
5. **Prod gates:** `posterior_planning_mode=draws` and validated `bayesian_fit_meta` are required for decision-grade posterior APIs (`posterior_planning_gate`, CLI surfaces that honor governance).
