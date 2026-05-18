# Python API

Primary entrypoints:

- `MMMTrainer`, `MMMTrainer.from_yaml(path)`
- `RidgeBOMMMTrainer`, `BayesianMMMTrainer`
- `DatasetBuilder`, `DataLoader`
- `CalibrationEngine`
- `DecompositionEngine`, `BudgetOptimizer`, `ReportBuilder`

Configs are `pydantic` models (`MMMConfig`) and round-trip to YAML via `dump_resolved_config` / `load_config`.

## Decision planning API

Full-panel planning (same gates and artifacts as `mmm decide …`). **Walkthrough:** [../03_planning/planning_howto.md](../03_planning/planning_howto.md).

| Function | Purpose |
|----------|---------|
| `mmm.decision.api.run_decision_simulation` | Δμ for a candidate media plan (+ optional control overlays) |
| `mmm.decision.api.run_decision_optimization` | SLSQP on media only; optional fixed non-media via `scenario` |
| `mmm.planning.scenario.planning_scenario_from_yaml` | Load typed `PlanningScenario` |
| `mmm.planning.scenario.planning_scenario_from_dict` | Build scenario from a dict (legacy keys OK) |

Outputs include `planning_assumptions`, `scenario_lineage`, and `control_scenario_policy` on decision bundles and top-level optimize payloads.

### Simulate

```python
from pathlib import Path

from mmm.decision.api import run_decision_simulation

payload = run_decision_simulation(
    config=Path("configs/prod.yaml"),
    scenario=Path("scenarios/candidate.yaml"),
    extension_report=Path("artifacts/extension_report.json"),
    out=Path("decisions/sim.json"),
)
assert payload["planning_assumptions"]["media_assumption"] in (
    "constant",
    "geo_channel",
    "piecewise_path",
)
print(payload["simulation"]["delta_mu"])
```

### Optimize (media only)

```python
from pathlib import Path

from mmm.decision.api import run_decision_optimization

# Without scenario: non-media = observed historical panel
payload = run_decision_optimization(
    config=Path("configs/prod.yaml"),
    extension_report=Path("artifacts/extension_report.json"),
    out=Path("decisions/opt.json"),
)

# With scenario: fixed control overlays on every optimizer evaluation
payload = run_decision_optimization(
    config=Path("configs/prod.yaml"),
    extension_report=Path("artifacts/extension_report.json"),
    scenario=Path("scenarios/promo_fixed.yaml"),
    out=Path("decisions/opt_promo.json"),
)
bundle = payload["decision_bundle"]
print(bundle["planning_assumptions"])
print(bundle.get("scenario_lineage", {}).get("plan_overlay_spec_sha256"))
```

### Scenario helpers

```python
from mmm.planning.scenario import planning_scenario_from_dict, planning_scenario_from_yaml

ps = planning_scenario_from_yaml("scenarios/promo_fixed.yaml")
lineage = ps.lineage_payload()  # scenario_hash, overlay SHA-256

ps2 = planning_scenario_from_dict(
    {
        "scenario_id": "inline",
        "media": {"candidate_spend": {"tv": 1e6}},
    }
)
```

Lower-level (tests, custom tooling): `mmm.planning.decision_simulate.simulate`, `mmm.optimization.budget.simulation_optimizer.optimize_budget_via_simulation` with `OptimizeNonMediaContext` — prefer `mmm.decision.service` / `mmm.decision.api` for prod policy enforcement.
