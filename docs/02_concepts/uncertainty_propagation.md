# Uncertainty propagation (PR 5A, reports only)

PR 5A adds **`uncertainty_propagation_report`** to extension outputs. It aggregates uncertainty already produced by training and extension diagnostics — it does **not** run new optimizers or emit production monetary confidence intervals.

## Configuration

```yaml
extensions:
  uncertainty_propagation:
    enabled: false
    ridge_summarize_bootstrap: true
    ridge_summarize_conformal: false   # reserved; not implemented yet
```

When `enabled: false`, the report still lists passive **source breakdown** from artifacts present on the extension report. Set `enabled: true` for full Ridge bootstrap and Bayesian posterior summarization blocks.

## Uncertainty sources

| Source | Ridge | Bayesian |
|--------|-------|----------|
| **Parameter** | Identifiability bootstrap coef dispersion | Posterior width on media coefficients / `sigma` |
| **Experiment** | `evidence_weighted_replay_summary` SE / loss | `bayesian_experiment_likelihood_report` adjusted SEs |
| **Hierarchy** | `hierarchy_diagnostics` penalty | `bayesian_hierarchy_report` `hier_sigma_group` |
| **Allocation** | `counterfactual_shock_plan` bridge roles | Experiment likelihood allocation inflation |

Each source includes a **magnitude_proxy** (0–1 heuristic) and a qualitative label — not calibrated Δμ intervals.

## Ridge bootstrap / conformal

- **Bootstrap:** summarized from `identifiability` (and optionally feature separability flags) when `ridge_summarize_bootstrap: true`.
- **Conformal:** not implemented; enabling `ridge_summarize_conformal` returns an explicit `not_implemented` status.

## Bayesian posterior

When `framework: bayesian` and propagation is enabled, the report summarizes MCMC posterior width, R-hat / ESS gates, and PPC checks already in `fit_out` — without enabling prod decisioning.

## Production policy

- `prod_decisioning_allowed: false`
- `prod_monetary_ci_allowed: false`
- Ridge prod continues to forbid precise monetary CIs (`ridge_production_forbids_precise_monetary_ci`).

PR 5B (robust optimization research) is separate; do not use this report for prod `optimize-budget` until that phase is explicitly validated.
