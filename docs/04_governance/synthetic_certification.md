# Synthetic certification (Δμ and optimizer)

Controlled DGP fixtures prove **internal numerical consistency** of the Ridge + BO + full-panel Δμ path. They do **not** prove causal validity, external validity, or that real-world experiments were well designed.

## Runtime vs CI

`run_synthetic_certification_suite(mode="exact")` (emitted on every `extension_report`) runs the **same checks** as `tests/test_synthetic_certification_exact.py`:

| Check | What is proven |
|-------|----------------|
| `semi_log_delta_mu_exact` | `simulate()` Δμ consistency on fixed coef |
| `geometric_adstock_carryover` | Week-1 carryover on impulse spend |
| `hill_saturation_analytic` | Hill formula matches implementation |
| `geometric_adstock_design_matrix` | Carryover preserved in design matrix |
| `hill_saturation_monotone_design_matrix` | Monotone feature path in design matrix |
| `two_channel_optimizer_direction` | Optimizer prefers higher-β channel |
| `transform_policy_consistency` | Prod decide rejects bad transform_policy |

## certification_level

| Level | Meaning |
|-------|---------|
| `exact` | All exact-tier checks passed |
| `smoke` | Smoke subset only (3 micro checks) |
| `incomplete` | One or more exact checks failed |

Failed **exact** certification blocks `production_readiness_report.approved_for_prod`. `incomplete` lowers `readiness_score`.

**Single source of truth:** `run_synthetic_certification_suite(mode="exact")` and `run_exact_check(name)` in `mmm/governance/synthetic_certification.py`. CI tests call the same `CHECK_REGISTRY` — no duplicated DGP logic in `tests/`.

## Assumptions

- `model_form: semi_log` with canonical geometric adstock + Hill saturation.
- Full-panel simulation path (`mmm.planning.decision_simulate.simulate`).
- Optimizer uses the same simulate path (`optimize_budget_via_simulation`).
- Synthetic panels are noise-free or lightly structured unless a test adds noise explicitly.

## Limitations

- Certification does **not** replace replay calibration, governance approval, or promotion review.
- Optimizer certification is **directional** unless an analytic optimum is closed-form and wired in the test.
- Collinear-channel tests assert **warnings**, not stable attribution splits.
- Fold-aligned replay honesty is covered separately in `tests/test_replay_refit_mode.py`.

## Operational use

Run fast certification in CI:

```bash
pytest tests/test_synthetic_certification_exact.py tests/test_replay_refit_mode.py -q -m "not slow"
```

Failure indicates regression in transforms, Δμ semantics, or optimizer wiring — investigate before promoting models.
