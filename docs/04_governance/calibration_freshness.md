# Calibration freshness and drift readiness

Connects drift diagnostics to **decision readiness** without auto-retrain or budget changes.

## Config (`governance:`)

```yaml
governance:
  calibration_max_age_days: 180
  coefficient_shift_threshold: 0.30
  replay_miss_threshold: 0.25
  require_review_on_drift: false   # default: warnings only
```

## Artifact

`extension_report.calibration_readiness_report` (always emitted post-fit).

| Signal | Source |
|--------|--------|
| `stale_calibration_warning` | Evidence age / missing replay loss |
| `coefficient_shift_score` | Current vs accepted-run reference coef |
| `replay_miss_rate` | `continuous_validation_report` classifications |
| `recommended_action` | `monitor`, `recalibration_recommended`, `experiment_refresh_required`, `model_review_required` |

## Blocking behavior

When `require_review_on_drift: true` and thresholds are exceeded:

- `blocks_planning_allowed: true`
- `model_release.state` → `invalidated` with `calibration_drift_review_required`
- Prod `mmm decide` fails closed via policy precheck

No coefficients, budgets, or training runs are modified automatically.
