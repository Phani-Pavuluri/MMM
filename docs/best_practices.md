# Best practices and caveats

- Default **`semi_log` + geometric adstock + Hill** balances flexibility and stability; **`log_log`** is supported for legacy workflows — avoid double-compression with aggressive saturations (warnings emitted).
- **Partial pooling** stabilizes sparse geos; verify pooling mode matches business heterogeneity.
- **Identifiability**: collinear channels and short histories inflate posterior width / BO instability — use calibration lifts and conservative priors/penalties.
- **Bayesian vs Ridge+BO**: use Bayesian when uncertainty is a first-class deliverable; use Ridge+BO for rapid exploration of transform hyperparameters.
- **Calibration** aligns observational fit with experimental evidence; always record match rates and dropped experiments.
