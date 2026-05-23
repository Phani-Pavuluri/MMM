# Decision stress testing

`decision_stress_report` evaluates **decision stability** under adverse extension signals. It recommends **monitor**, **review**, or **block** for operators only — **`auto_budget_change` is always false**.

## Scope (v1)

`stress_scope` is always train-time only:

| `stress_scope` | When |
|----------------|------|
| `train_time` | `ridge_fit_summary` + training panel — behavioral probes via `optimize_budget_via_simulation` |
| `train_time_signal_only` | Missing panel — extension signals only (degraded) |

Stress is emitted on the **train** `extension_report` and is **not** recomputed at `mmm decide` time.

## Modes

| Mode | When |
|------|------|
| `behavioral` | Panel + ridge summary present |
| `signal_only` | Missing panel |

## Scenarios (behavioral when panel present)

| Scenario | Behavior |
|----------|----------|
| `stale_calibration` | Re-optimize under stale flag |
| `coefficient_perturbation` | Coef ×0.85 → re-optimize; measure allocation L1 + flip |
| `missing_evidence` | Extension signal |
| `budget_shock` | Budget ×1.25 → re-optimize |
| `replay_degradation` | Severe replay gap + coef stress |

## Outputs

| Field | Meaning |
|-------|---------|
| `stress_scope` | `train_time` \| `train_time_signal_only` |
| `stress_severity` | `low` \| `moderate` \| `high` \| `critical` |
| `allocation_stability` | Heuristic stability under perturbation |
| `decision_flip_rate` | Share of behavioral scenarios with top-channel flip |
| `decision_instability_index` | Mean normalized allocation L1 movement across scenarios |
| `recommended_action` | `monitor` \| `review` \| `block` |

Stress does **not** change budgets or estimand. It does **not** prove causal invalidity.

## Module

`mmm/governance/decision_stress.py` — `build_decision_stress_report`.

## Validation

```bash
pytest tests/test_decision_stress.py -q -m "not slow"
```
