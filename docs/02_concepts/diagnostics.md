# Diagnostics and troubleshooting

- **Ridge+BO**: inspect Optuna `user_attrs` / leaderboard JSON for objective decomposition; watch for unstable `decay` near 0/1.
- **Bayesian**: check `r_hat`, `ess_bulk`, divergences; shorten horizon or simplify transforms if E-BFMI is low.
- **Data**: schema errors list missing columns or duplicates explicitly.
- **CV**: if no splits, lower `min_train_weeks` or increase history.
- **Continuous validation** (`continuous_validation_report`): prior-run predicted lift vs experiment lift; check `model_trust_score`, classification counts, and `not_evaluable` reasons before recalibration.
- **Decision validation** (`decision_validation_report`): prior recommendation vs subsequent experiment lift; `decision_safe` is always false; observational rows must stay `not_evaluable`.
