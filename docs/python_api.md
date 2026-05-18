# Python API

Primary entrypoints:

- `MMMTrainer`, `MMMTrainer.from_yaml(path)`
- `RidgeBOMMMTrainer`, `BayesianMMMTrainer`
- `DatasetBuilder`, `DataLoader`
- `CalibrationEngine`
- `DecompositionEngine`, `BudgetOptimizer`, `ReportBuilder`

Configs are `pydantic` models (`MMMConfig`) and round-trip to YAML via `dump_resolved_config` / `load_config`.

## Decision planning API

- `mmm.decision.api.run_decision_simulation` — full-panel Δμ (`simulate_decision`); pass `PlanningScenario` YAML via `scenario` path.
- `mmm.decision.api.run_decision_optimization` — media-only SLSQP optimize; optional `scenario` for **fixed** non-media control overlays.
- `mmm.planning.scenario.PlanningScenario` — typed contract (`scenario_id`, `media`, `controls`, hashes).
- Outputs include `planning_assumptions` (`controls_assumption`, `media_assumption`, `world_assumption`) and `scenario_lineage` on decision bundles.
