# Production readiness certification

`production_readiness_report` rolls up post-train and optional certification artifacts into a single **approved_for_prod** view. It does **not** prove causal incrementality or that budget moves will beat holdout.

## Inputs

| Source | Role |
|--------|------|
| `synthetic_certification_report` | Semi-log Δμ, geometric adstock, Hill saturation consistency |
| `optimizer_certification_report` | Deterministic optimizer surfaces (optional extension flag) |
| `calibration_readiness_report` | Stale calibration, coefficient shift |
| `calibration_summary` | Replay generalization gap severity |
| `reproducibility_certification_report` | Required when `extensions.reproducibility_certification.enabled` |
| `performance_certification_report` | Advisory when performance extension enabled |
| `ridge_fit_summary` + `transform_policy` + `data_fingerprint` | Decision contract completeness |
| `governance` / `model_release` | Promotion and planning gates |
| `decision_stress_report` | Stress severity advisory |

## Output fields

| Field | Meaning |
|-------|---------|
| `approved_for_prod` | `true` when no blocking reasons remain |
| `blocked_reasons` | Hard certification failures (contract, synthetic, severe replay, etc.) |
| `warnings` | Non-blocking issues (stale calibration, research extensions, missing optimizer cert) |
| `readiness_score` | Heuristic 0–1 score (research extensions reduce score) |
| `missing_requirements` | Named gaps for operators |
| `require_production_certification` | Echo of `governance.require_production_certification` |

## Evidence rules

- **Synthetic:** `certification_level` must be `exact` for approval (always enforced).
- **Optimizer:** required and must pass when `governance.require_production_certification: true`.
- **Reproducibility:** self-certification alone (`identical_output=null`) is **not** evidence; use `extensions.reproducibility_certification.reference_run_path` for independent-run comparison.
- **Replay:** severe generalization gap blocks approval.
- **model_release:** must be `planning_allowed` when strict gate enabled.

## Prod gate (warning vs hard block)

| Condition | Default prod decide | `require_production_certification: true` |
|-----------|---------------------|------------------------------------------|
| `approved_for_prod=false` | **Severe warning** on decide JSON; decide continues | **Fail closed** (`PolicyError`) |
| Missing `optimizer_certification_report` | Warning; may still approve if other checks pass | Blocks `approved_for_prod` |
| `optimizer_certification_mode=directional_fallback` | Warning (not full analytic recovery) | Warning only |

Default: prod decide **never silently ignores** failed readiness.

When `governance.require_production_certification: true`, prod `mmm decide` paths fail closed unless `approved_for_prod` is true.

Prod-bound trains: `examples/prod_train_template.yaml` (`optimizer_certification.enabled: true`).

## Module

`mmm/governance/production_readiness.py` — `build_production_readiness_report`, `production_readiness_decide_surface`, `require_production_readiness_for_prod_decide`.

## Validation

```bash
pytest tests/test_production_readiness.py -q -m "not slow"
```
