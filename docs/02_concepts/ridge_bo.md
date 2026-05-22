# Ridge + BO framework

Inner loop: **ridge regression** on the modeling scale (`log` revenue for `semi_log`; `log` revenue and `log` media for `log_log`) with explicit media transforms. Level predictions use `exp(linear_predictor)` for both forms (inverse link on `log(y)`).

Outer loop: **Optuna** TPE search over adstock decay, Hill parameters, and `log10(alpha)` ridge strength. If Optuna is absent, a seeded random grid preserves functionality.

Objective: composite of predictive CV loss (WMAPE by default), optional calibration mismatch (variance-weighted), stability across folds, plausibility penalties, and complexity. Each trial logs raw and normalized components plus weights — never a single opaque scalar.

Use Ridge+BO for fast iteration, many hyperparameters in transforms, or when MCMC cost is prohibitive.

## Production canonical path: `semi_log`

- **`run_environment=prod`** requires **`model_form=semi_log`** for Ridge BO and prod decision simulate/optimize.
- **`log_log`** is **research/diagnostic only** in this package until a future validation PR certifies:
  - DGP recovery
  - elasticity interpretation
  - level prediction
  - Δμ recovery
  - zero-spend behavior
- **LOG_LOG coefficient elasticity interpretation is not production-certified.**
- Prod config with `model_form=log_log` fails validation. Prod decisions also reject extension reports whose `economics_contract` or `ridge_fit_summary` still indicate `log_log` (stale artifacts).

## Research: `log_log`

Non-prod environments may set `model_form=log_log` for experimentation. Do not use LOG_LOG outputs for production budget decisions or hierarchical borrowing (see [hierarchical_borrowing.md](hierarchical_borrowing.md)).
