# Synthetic certification (Δμ and optimizer)

Controlled DGP fixtures prove **internal numerical consistency** of the Ridge + BO + full-panel Δμ path. They do **not** prove causal validity, external validity, or that real-world experiments were well designed.

## Certified cases

| Fixture | What is proven | Typical tolerance |
|---------|----------------|-----------------|
| Linear semi-log, no adstock | `simulate()` Δμ matches analytic level-scale expectation from fixed coef | 5% relative or 1.0 absolute |
| Geometric adstock | Design matrix preserves carryover across zero-spend weeks | monotonicity vs zero-spend baseline |
| Hill saturation | Transformed spend is monotone in raw spend | non-decreasing feature path |
| Two-channel optimizer | Budget shifts toward higher-β channel under `optimize_budget_via_simulation` | directional (high ≥ low) |
| Collinear media | Identifiability / separability diagnostics fire on near-duplicate channels | score < 0.99 or separability report |

Tests live in `tests/test_synthetic_certification_exact.py` and related DGP modules.

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
