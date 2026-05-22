# Bayesian framework

- Backend: **PyMC** (default). **Stan** wired as optional `StanMMMTrainer` stub pending packaged `.stan` models.
- Pooling: partial pooling uses non-centered hierarchical shrinkage on channel coefficients per geo; full pooling shares media slopes; none fits independent positive slopes per geo.
- Priors: HalfNormal media effects for identifiability; weakly informative intercepts and noise.
- Diagnostics: ArviZ summaries (`r_hat`, `ess_bulk`, divergences).
- **Experiment likelihood (PR 3, research-only):** optional `bayesian.use_experiment_likelihood` adds
  `experiment_lift_i ~ Normal(mmm_implied_lift_i, adjusted_SE_i)` per compatible evidence unit.
  See `mmm/calibration/bayesian_experiment_likelihood.py` and [experiment_evidence.md](experiment_evidence.md).
  **Does not enable prod decisioning** (`prod_decisioning_allowed: false` in `bayesian_experiment_likelihood_report`).

Prefer Bayesian when sample size per geo is moderate, priors are defensible, and posterior uncertainty is required for decisioning (research/diagnostic surfaces only under current prod policy).

## Experiment likelihood (research-only)

```yaml
bayesian:
  use_experiment_likelihood: true
  experiment_registry_path: path/to/evidence.json
  experiment_likelihood_weight: 1.0
  min_experiment_quality_tier: medium
  allow_aggregate_only_evidence: true
  allow_allocated_shocks: true
  exp_likelihood_research_only: true   # must stay true
```

- Reuses ExperimentEvidence, compatibility resolver, shock planner, and quality scoring.
- Implied lift is on the **Bayesian modeling (log) scale**; level-reported experiment lift is rejected unless `allow_level_lift_mismatch_research: true`.
- `adjusted_SE` inflates with lower evidence weight, aggregate-only status, allocation, and staleness.
- Artifact: `extension_report.bayesian_experiment_likelihood_report` (tier: research).

**Ridge vs Bayesian:** Ridge weighted replay is an optimization penalty; Bayesian experiment likelihood is a probabilistic term in the generative model. Neither substitutes for designed experiments or prod gates on the other framework.

## Bayesian hierarchy (PR 4B, research-only)

```yaml
hierarchy:
  hierarchy_definition_path: path/to/hierarchy.json   # shared JSON contract with Ridge
bayesian:
  use_hierarchy: true
  hierarchy_research_only: true   # must stay true
  hierarchy_group_sigma_prior: 0.5
```

- Explicit channel/campaign (or `metadata.ridge_effect_pairs`) mapping only — never inferred from data.
- Generative term: **child media coefficient ~ Normal(parent media coefficient, hier_sigma_group)**.
- Requires `pooling=full` or `pooling=partial` and **`model_form=semi_log`** (`log_log` hierarchy is blocked).
- Artifact: `extension_report.bayesian_hierarchy_report` with parent/child mapping, posterior shrinkage, group variance, prior/posterior overlap, and governance warnings (**hierarchy ≠ causal proof**).
- **`prod_decisioning_allowed: false`** — prod Bayesian budget/planning remains blocked by existing policy.

## Production checklist (decision path)

1. **Diagnostics:** `compute_bayesian_decision_diagnostics` must summarize **media, controls, intercepts / hierarchy**, and `sigma` — not `sigma` alone. Thresholds come from `extensions.governance` (`bayesian_max_rhat`, `bayesian_min_ess_bulk`, `bayesian_max_divergences`).
2. **PPC:** Run `posterior_predictive` in training (`bayesian.posterior_predictive_draws > 0`). The PPC artifact (`build_bayesian_predictive_artifact`) records `mean_abs_gap`, optional **empirical 90% coverage** vs `y_obs` (modeling scale), and `std_ratio_pp_over_obs`. When `bayesian_max_mean_abs_ppc_gap` is set, `posterior_predictive_ok` requires that gap; otherwise a substantive PPC summary (gap or coverage) is still required for `posterior_predictive_ok`.
3. **Transforms:** `extensions.product.bayesian_decision_transform_stance` defaults to `fixed_yaml_features_labeled` (YAML `transforms.*_params` — not Ridge+BO joint hyperparameter search). Cross-framework Ridge vs Bayesian comparability remains **explicitly non-automatic**; see `transform_policy.build_transform_policy_manifest`.
4. **Posterior planning draws:** For `pooling=full`, `BayesianMMMTrainer.fit` returns `linear_coef_draws` plus `ppc["linear_coef_draws_export"]` when export succeeds. For **partial / none pooling**, use `hierarchical_draw_pack` from `fit_out["hierarchical_draw_pack"]` (see `hierarchical_coefficient_draws_from_pymc_idata`) with `delta_mu_draws_hierarchical_geo_beta` / `simulate(..., hierarchical_draw_pack=...)` — not the global `linear_coef_draws` path.
5. **Prod gates:** `posterior_planning_mode=draws` and validated `bayesian_fit_meta` are required for decision-grade posterior APIs (`posterior_planning_gate`, CLI surfaces that honor governance).
