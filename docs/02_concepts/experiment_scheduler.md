# Experiment scheduler

The **experiment scheduler** extends [feature separability](feature_separability.md) into an **experiment prioritization** layer. It answers:

> Where should we spend our next experimentation budget, and why?

It does **not** design geo tests, assign treatment, select geos, or run experiments. Those belong in a separate experimentation package.

## Position in the workflow

```text
Identifiability
  → Feature separability
  → Calibration / experiment matching
  → Optimization sensitivity (curves + governance)
  → Experiment scheduler
  → Experiment requests (prioritized, not designed)
```

## Extension output

After training, `extension_report.experiment_scheduler_report` includes:

| Field | Purpose |
|--------|---------|
| `experiment_priority_summary` | Counts and top-ranked units |
| `scored_units` | Per channel/group scores and recommended actions |
| `high_priority_requests` | `ExperimentRequest` payloads (tier = high) |
| `medium_priority_requests` | Medium-tier requests |
| `low_priority_requests` | Low-tier requests |
| `stale_experiment_coverage` | Units with weak or missing calibration evidence |
| `unsupported_split_level_claims` | Carried from separability (split-level claims not supported) |
| `scheduler_notes` | Operator-facing notes |

Governance flags (always present):

- `diagnostic_only: true`
- `auto_experiment_execution_forbidden: true`
- `auto_budget_reallocation_forbidden: true`
- `auto_model_retraining_forbidden: true`

## Inputs (reused, not recomputed)

The scheduler reads existing extension artifacts:

- `feature_separability_report` — separability, contribution/coefficient stability, business importance
- `identifiability` — VIF and global identifiability stress
- `experiment_matching` — global evidence strength
- `governance` — optimization approval
- `curve_bundles` — per-channel curve safety / optimizer sensitivity proxy
- Optimization gate result (from the extension path)

It does **not** refit models or re-run calibration matching.

## Scoring (per channel or feature group)

Each unit receives scores in `[0, 1]`:

| Score | Meaning |
|--------|---------|
| `uncertainty_score` | Low separability, identifiability stress, unstable attribution |
| `business_importance_score` | Material spend and contribution share |
| `decision_impact_score` | Optimization relevance, governance approval, curve sensitivity |
| `calibration_evidence_score` | Strength of matched experiment evidence |
| `experiment_staleness_score` | Weak/absent calibration → higher staleness |
| `experiment_priority_score` | Weighted composite → `priority_tier` (`high` / `medium` / `low`) |

## Recommendation actions

| Action | Typical situation |
|--------|------------------|
| `no_action` | High separability, strong calibration, low uncertainty |
| `monitor` | Acceptable but worth watching |
| `keep_with_caution` | Moderate separability |
| `rollup_recommended` | Low separability, **low spend** — prefer taxonomy rollup over a test |
| `experiment_optional` | Material spend, moderate priority |
| `experiment_recommended` | Material spend, low separability, decision-relevant |
| `experiment_high_priority` | High composite priority + optimizer sensitivity |

**Tiny-spend guard:** experiment actions require group spend share ≥ `extensions.feature_separability.experiment_min_group_spend_share` (default 3%). Below that threshold, the scheduler recommends `monitor`, `rollup_recommended`, or `no_action` — not experiments.

## ExperimentRequest contract

When an experiment action is warranted, the report emits an `ExperimentRequest` (JSON-serializable):

- `request_id` — deterministic hash of unit + reason + uncertainty source
- `channel_or_group`, `reason`, `uncertainty_source`
- `priority_score`, `priority_tier`, `business_importance`
- `required_estimand`, `required_kpi`
- `suggested_test_type` — e.g. `geo_holdout`, `heavy_up`, `spend_shock`, `incrementality_test`
- `preferred_geo_level`, `notes`

The scheduler **does not** specify treatment assignment, geo selection, MDE, duration, estimator, or randomization.

## Configuration

```yaml
extensions:
  experiment_scheduler:
    enabled: true
    high_priority_threshold: 0.62
    low_priority_threshold: 0.38
```

Spend eligibility reuses `extensions.feature_separability.experiment_min_group_spend_share`.

## Relationship to external geo experimentation

Export `high_priority_requests` (and optionally medium) into your geo-experimentation or incrementality platform. That system owns design, power, geo selection, and execution. This library only **prioritizes** where effort is likely to reduce decision risk.

## Related docs

- [feature_separability.md](feature_separability.md) — separability diagnostics (stage 1)
- [calibration.md](calibration.md) — experiment matching and replay
- [diagnostics.md](diagnostics.md) — identifiability and response curves
- [../04_governance/prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) — production decision gates
