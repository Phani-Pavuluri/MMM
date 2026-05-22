# Continuous validation (diagnostic)

**Continuous validation** compares **prior accepted-run MMM predictions** to **new experiment evidence** when results arrive. It closes the loop between model-implied lift and measured experiment lift without changing training, registry promotion, or budgets.

## Purpose

- Find prior model runs that completed **before** each experiment’s `freshness_date`.
- Retrieve stored or extension-embedded **predicted lift** for that experiment scope.
- Compare to **experiment lift** and optional **standard error** (standardized error when SE present).
- Classify each pair: `aligned`, `mild_miss`, `severe_miss`, or `not_evaluable`.

## Configuration (opt-in, default off)

```yaml
extensions:
  continuous_validation:
    enabled: false
    registry_dir: path/to/accepted_runs/    # accepted_runs.json or runs/*.json
    lookback_days: 365
    require_experiment_se: false
    experiment_registry_path: null          # defaults to calibration.evidence_registry_path
```

## Local accepted-run registry (no remote service)

Operators maintain a **local** JSON registry (not auto-promoted by this feature):

```json
{
  "registry_version": "mmm_accepted_run_registry_v1",
  "runs": [
    {
      "run_id": "run-2024-01-15",
      "completed_at": "2024-01-15",
      "extension_report_path": "/artifacts/ext.json",
      "predicted_lifts": [
        {"experiment_id": "exp-1", "predicted_lift": 0.42}
      ]
    }
  ]
}
```

Predicted lift may also be recovered from prior `extension_report` paths via:

- `bayesian_experiment_likelihood_report.posterior_experiment_fit`
- `evidence_weighted_replay_summary.units[].implied_delta`

## Artifact

When `extensions.continuous_validation.enabled: true`, `extension_report` may include:

- `continuous_validation_report` (tier: **diagnostic** in `run_manifest`)

Key fields: `n_experiments_evaluated`, classification counts, `per_experiment_results`, `calibration_drift`, `evidence_freshness_report`, `model_trust_score`, `recommended_action` (`monitor` | `recalibrate_recommended` | `experiment_refresh_recommended` | `model_review_required`).

## Guardrails

| Flag | Value |
|------|--------|
| `diagnostic_only` | `true` |
| `auto_retrain` | `false` |
| `auto_registry_promotion` | `false` |
| `auto_budget_change` | `false` |
| `prod_decisioning_allowed` | `false` |

Classification thresholds (when SE present): \|z\| &lt; 1 → aligned; &lt; 2 → mild_miss; else severe_miss.

## Limitations

- Requires manually curated **accepted-run** and **experiment** registries.
- Missing prior prediction or SE → `not_evaluable` (stricter when `require_experiment_se: true`).
- Does not prove causal validity beyond each experiment’s design.
- Stale evidence is reported in `evidence_freshness_report` but does not auto-downweight training.
