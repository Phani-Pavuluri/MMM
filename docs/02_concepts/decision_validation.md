# Decision validation (diagnostic)

**Decision validation** evaluates whether **prior budget recommendations** remain consistent with **later experiment evidence**. It is a governance and learning surface — not prod optimizer feedback.

## Purpose

- Load prior **decision registry** entries (allocation, channel ranking, predicted lifts).
- Match **subsequent** accepted experiment evidence (after `decided_at`, within `lookback_days`).
- Compare predicted vs experimental lift, ranking stability, and an **allocation regret proxy**.
- Emit `recommendation_trust_score` and `recommended_action` for human review.

## Configuration (opt-in, default off)

```yaml
extensions:
  decision_validation:
    enabled: false
    decision_registry_dir: path/to/decisions/
    experiment_registry_path: null          # defaults to calibration.evidence_registry_path
    lookback_days: 180
```

## Local decision registry (no remote service)

```json
{
  "registry_version": "mmm_decision_validation_registry_v1",
  "decisions": [
    {
      "decision_id": "dec-2024-03-01",
      "decided_at": "2024-03-01",
      "decision_bundle_path": "/artifacts/decision_bundle.json",
      "recommended_allocation": {"tv": 100, "search": 50},
      "predicted_lifts_by_channel": {"tv": 0.5},
      "channel_ranking": ["tv", "search"]
    }
  ]
}
```

## Artifact

When `extensions.decision_validation.enabled: true`, `extension_report` may include:

- `decision_validation_report` (tier: **diagnostic**)

Key fields: `n_decisions_evaluated`, `prediction_error_summary`, `ranking_stability`, `allocation_regret_proxy`, `recommendation_trust_score`, `per_decision_results`, `recommended_action` (`monitor` | `decision_policy_review` | `calibration_review` | `experiment_refresh_recommended`).

## Guardrails

| Flag | Value |
|------|--------|
| `diagnostic_only` | `true` |
| `decision_safe` | `false` |
| `auto_budget_change` | `false` |
| `auto_optimizer_change` | `false` |
| `prod_decisioning_allowed` | `false` |

**Observational evidence** (`metadata.validation_design: observational`, etc.) is classified **`not_evaluable`** — it must not be treated as experiment truth for causal validation of recommendations.

## Limitations

- Does **not** claim causal proof that a recommendation “worked” unless validation design supports it.
- Observational outcomes are excluded from validation counts.
- Regret proxy is a **diagnostic heuristic** (spend share vs lift-implied share), not realized ROI.
- Does not auto-update optimizer settings or prod decision bundles.
