# Experiment calibration

Experiments are `ExperimentObservation` records (CSV or JSON list) with lift and optional `lift_se`.

Matching respects configured levels (`geo`, `time_window`, `channel`, `device`, `product`) and silently drops incompatible rows.

- **Ridge+BO**: calibration mismatch enters the composite objective (weighted MSE scaled by SE).
- **Bayesian**: integrate as additional Gaussian likelihoods on matched lift summaries (extend in `pymc_trainer` when experiment rows align to model states).

Always document which uncertainty source (experiment SE vs posterior width) is active in a given report.
