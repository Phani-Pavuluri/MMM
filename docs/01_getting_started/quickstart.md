# Quickstart

Requires **Python 3.11+** (3.11 or 3.12 recommended).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
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

After training, use **`mmm decide simulate`** / **`mmm decide optimize-budget`** for full-panel planning — see [../03_planning/planning_howto.md](../03_planning/planning_howto.md).
