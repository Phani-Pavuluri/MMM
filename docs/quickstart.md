# Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bayesian,bo,dev]"
mmm train --config examples/minimal_train.yaml
```

Python:

```python
from mmm.api import MMMTrainer
trainer = MMMTrainer.from_yaml("examples/minimal_train.yaml")
trainer.run()
```

Bayesian extras: `pip install mmm[bayesian]`. BO extras: `pip install mmm[bo]` (falls back to grid search if Optuna missing).
