# Optimizer certification

`optimizer_certification_report` certifies the Ridge semi-log **simulation optimizer** on deterministic synthetic surfaces with known allocation structure. It is **not** a substitute for holdout predictive validation.

## Scenarios

| Scenario | Surface | Expected |
|----------|---------|----------|
| **A** | Linear-in-log coef ratio 2:1 | Majority budget to higher-elasticity channel |
| **B** | Saturated two-channel Hill | Allocation ratio within tight band of 2:1 |
| **repeatability** | Same surface, multiple seeds | Allocation L1 std below threshold |

Each scenario reports `analytic_optimum_allocation` (grid search on the same Δμ surface), `optimum_distance`, `optimizer_error`, and `objective_gap`.

## certification_mode (what was proven)

| Mode | Meaning |
|------|---------|
| `analytic_tolerance` | Observed allocation within L1 tolerance of grid optimum (non-corner surfaces) |
| `directional_fallback` | Corner-dominant grid optimum — pass means higher-elasticity channel gets more budget, **not** exact optimum recovery |
| `smoke` | Report-level rollup when any scenario fails |

When the grid optimum is corner-dominant (≥95% on one channel), scenario mode is `directional_fallback` (directional correctness + objective-gap tolerance). Do **not** describe that as exact optimizer recovery.

## Output fields

Per scenario: `certification_mode`, `observed_allocation`, `optimum_distance`, `feasibility`, `convergence`, `certification_status`.

Report-level: `certification_status`, `certification_mode`, `n_pass`, `n_scenarios`, `governance_warnings`.

## Enable on train

```yaml
extensions:
  optimizer_certification:
    enabled: true
```

Nightly CI always runs the suite via `mmm.evaluation.nightly_certification`.

## Module

`mmm/optimization/optimizer_certification.py` — `build_optimizer_certification_report`.

## Validation

```bash
pytest tests/test_optimizer_certification.py -q -m "not slow"
```
