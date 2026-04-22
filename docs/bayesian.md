# Bayesian framework

- Backend: **PyMC** (default). **Stan** wired as optional `StanMMMTrainer` stub pending packaged `.stan` models.
- Pooling: partial pooling uses non-centered hierarchical shrinkage on channel coefficients per geo; full pooling shares media slopes; none fits independent positive slopes per geo.
- Priors: HalfNormal media effects for identifiability; weakly informative intercepts and noise.
- Diagnostics: ArviZ summaries (`r_hat`, `ess_bulk`, divergences).
- Calibration: extend likelihood with matched experiment terms (see `calibration/`).

Prefer Bayesian when sample size per geo is moderate, priors are defensible, and posterior uncertainty is required for decisioning.
