# Diagnostics and troubleshooting

- **Ridge+BO**: inspect Optuna `user_attrs` / leaderboard JSON for objective decomposition; watch for unstable `decay` near 0/1.
- **Bayesian**: check `r_hat`, `ess_bulk`, divergences; shorten horizon or simplify transforms if E-BFMI is low.
- **Data**: schema errors list missing columns or duplicates explicitly.
- **CV**: if no splits, lower `min_train_weeks` or increase history.
