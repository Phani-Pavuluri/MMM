# Python API

Primary entrypoints:

- `MMMTrainer`, `MMMTrainer.from_yaml(path)`
- `RidgeBOMMMTrainer`, `BayesianMMMTrainer`
- `DatasetBuilder`, `DataLoader`
- `CalibrationEngine`
- `DecompositionEngine`, `BudgetOptimizer`, `ReportBuilder`

Configs are `pydantic` models (`MMMConfig`) and round-trip to YAML via `dump_resolved_config` / `load_config`.
