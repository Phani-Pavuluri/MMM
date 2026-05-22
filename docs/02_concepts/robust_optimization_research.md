# Robust optimization research (PR 5B)

PR 5B adds **`robust_optimization_research`** to extension reports. It compares **candidate budget allocations** under multiple research objectives using **point Δμ** (full-panel `simulate`) and **uncertainty proxies** from PR 5A — not calibrated prod intervals.

This is **not** production optimization. It does **not** call `mmm decide optimize-budget` or `optimize_budget_via_simulation`.

## Configuration

```yaml
extensions:
  robust_optimization_research:
    enabled: false
    risk_lambda: 1.0
    lcb_z_score: 1.0
    n_candidates: 6
    n_stability_scenarios: 6
    budget_perturbation_pct: 0.05
    frontier_lambda_grid: [0.0, 0.5, 1.0, 2.0, 5.0]
```

Requires a successful **Ridge BO** fit (`framework: ridge_bo` with artifacts) for Δμ evaluation.

## Objectives (research)

| Objective | Score |
|-----------|--------|
| Maximize expected Δμ | `expected_delta_mu` (point simulate) |
| Maximize LCB proxy | `expected_delta_mu - lcb_z * uncertainty_proxy * scale` |
| Maximize risk-adjusted | `expected_delta_mu - risk_lambda * uncertainty_proxy * scale` |

`uncertainty_proxy` is derived from `uncertainty_propagation_report` (parameter, experiment, hierarchy, allocation sources).

## Artifact fields

- `baseline_allocation`, `candidate_allocations`
- `expected_delta_mu`, `uncertainty_proxy`, `lower_confidence_bound_proxy`, `risk_adjusted_score`
- `risk_return_frontier` — best candidate per λ on the frontier grid
- `allocation_stability` — Δμ std under budget perturbations
- `downside_risk_proxy` — gap between mean and min perturbed Δμ
- `ranking_stability` — overlap of top-3 ranks across objectives

## Guardrails

- `research_only: true`
- `prod_decisioning_allowed: false`
- `decision_safe: false` (always)
- `recommended_prod_allocation: null`
- `prod_optimize_budget_path_used: false`

When bootstrap/posterior summaries are missing, warnings state that scores use **proxy labels only**.

## Relation to prod optimize-budget

Prod **`optimize-budget`** remains the point-Δμ SLSQP path in `mmm.decision.service` → `optimize_budget_via_simulation`. Do not promote robust research winners to prod spend without a separate validation phase.
