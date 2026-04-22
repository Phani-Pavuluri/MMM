# MMM — Weekly Geo-Level Marketing Mix Modeling

Production-oriented Python library for weekly geo-level MMM with:

- **Bayesian MMM** (PyMC default; CmdStanPy optional)
- **Ridge + Bayesian Optimization** (Optuna default) with a documented composite objective
- **Transforms**: geometric / Weibull adstock; Hill / log / logistic saturation; plugin hooks
- **Pooling**: none, full, partial (hierarchical) — configurable per framework
- **Calibration**: GeoX / CLS-style experiments with multi-level matching and likelihood / objective integration
- **Decisioning**: decomposition, response curves, uncertainty (P10/P50/P90), budget optimization, scenarios
- **Artifacts**: pluggable stores (local default; MLflow optional, lazy-loaded)

See `docs/` for architecture, YAML reference, and framework guides. Quickstart:

```bash
pip install -e ".[bayesian,bo,dev]"
mmm train --config examples/minimal_train.yaml
```

Python API:

```python
from mmm.api import MMMTrainer
from mmm.config.load import load_config

trainer = MMMTrainer.from_yaml("examples/minimal_train.yaml")
result = trainer.run()
```
