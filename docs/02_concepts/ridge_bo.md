# Ridge + BO framework

Inner loop: **ridge regression** on the modeling scale (`log` revenue for `semi_log`; `log` spend for `log_log`) with explicit media transforms.

Outer loop: **Optuna** TPE search over adstock decay, Hill parameters, and `log10(alpha)` ridge strength. If Optuna is absent, a seeded random grid preserves functionality.

Objective: composite of predictive CV loss (WMAPE by default), optional calibration mismatch (variance-weighted), stability across folds, plausibility penalties, and complexity. Each trial logs raw and normalized components plus weights — never a single opaque scalar.

Use Ridge+BO for fast iteration, many hyperparameters in transforms, or when MCMC cost is prohibitive.
