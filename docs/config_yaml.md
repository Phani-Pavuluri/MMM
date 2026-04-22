# YAML configuration

Canonical keys mirror `MMMConfig` in `mmm/config/schema.py`. Important sections:

- `framework`: `ridge_bo` | `bayesian`
- `model_form`: `semi_log` (default) | `log_log`
- `pooling`: `none` | `full` | `partial`
- `data`: paths, column names, channel list
- `transforms`: `adstock` (`geometric`|`weibull`), `saturation` (`hill`|`log`|`logistic`), optional param dicts
- `cv`: `mode` (`auto`|`rolling`|`expanding`), `n_splits`, `min_train_weeks`, `horizon_weeks`, `gap_weeks`
- `ridge_bo` / `bayesian`: backend-specific knobs
- `objective`: composite weights for Ridge+BO
- `calibration`: optional experiments path + match levels
- `artifacts`: `local` store root or `mlflow` experiment name

Every training run should persist `resolved_config.yaml` next to metrics and diagnostics.

## Extensions (`extensions:`)

Optional block on `MMMConfig` for identifiability, governance scorecard, optimization gates, falsification, feature-engine preview (trend/Fourier/holiday flags), and estimand alignment. Defaults are non-breaking. Training writes `extension_report.json` via the artifact store when `MMMTrainer.run()` completes.
