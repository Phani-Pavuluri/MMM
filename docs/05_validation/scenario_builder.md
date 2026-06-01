# ScenarioBuilder (Phase 3B MVP)

**Status:** Implemented — `mmm/validation/synthetic/scenario_builder.py`  
**Version:** `scenario_builder_v1.0.0`

## Purpose

Compose **deterministic** `world_truth.json` documents from declarative scenario specifications. ScenarioBuilder is the v1.x composition layer between fixed archetype generators (Phase 3A) and large-scale sweeps (Phase 5).

**Not in scope:** Monte Carlo sampling, certification runners, train/decide workflows, Bayesian worlds, or stochastic DGP panel simulation.

## Architecture

```text
ScenarioSpec
      │
      ▼
build_world_truth()  ──► compose_archetype_truth()  (generators.py)
      │
      ▼
write_scenario_world()  ──► world_truth.json only
      │
      ▼
materialize_world()  ──► panel / replay / checksums / metadata
```

| Layer | Writes |
|-------|--------|
| ScenarioBuilder | `world_truth.json` only |
| Materializer | All derived bundle artifacts |

## ScenarioSpec fields

| Field | Values | Effect (truth sections) |
|-------|--------|-------------------------|
| `world_id` | string | `metadata.world_id`; bundle directory name |
| `family` | `baseline`, `replay` | Archetype; replay enables experiment units |
| `seed` | int ≥ 0 | Determinism; coefficients, windows |
| `n_geos` | int ≥ 1 | `geo_truth` |
| `n_periods` | int ≥ 4 | `time_truth` |
| `channels` | list[string] | `media_truth`, `transform_truth`, `coefficient_truth` |
| `noise_level` | `low`, `medium`, `high` | `outcome_truth.observation_noise_level` |
| `correlation_level` | `low`, `medium`, `severe` | `media_truth.spend_process_spec`; severe → artifact warning |
| `seasonality` | `none`, `mild`, `strong` | `time_truth.seasonality_declared` |
| `drift` | bool | `drift_truth` changepoints + coefficient_drift |
| `experiment_quality` | `none`, `weak`, `medium`, `high` | `experiment_truth.units` (replay); lift SE tier |
| `privacy_loss` | bool | `drift_truth.privacy_shifts` |
| `missingness` | `none`, `mild` | `artifact_truth.expected_warnings` |

## Determinism contract

Identical `ScenarioSpec` (including field order in `channels` tuple) → identical canonical `world_truth.json`. Changing one field updates only the truth sections that field owns (e.g. `n_geos` → `geo_truth` and dependent experiment geo lists).

## MVP limitations

- Materialized panels remain **constant spend/KPI** (materializer smoke path) — scenario axes are declared in truth, not fully simulated in Parquet.
- Drift and collinearity are **authored** in `drift_truth` / `artifact_truth.expected_warnings`, not injected into complex time series yet.
- Lattice sweep MVP (Phase 5A) — see [lattice_sweep.md](lattice_sweep.md); committed smoke worlds remain for CI.

## Committed smoke worlds

| World | Spec constant |
|-------|----------------|
| `validation/worlds/WORLD-005-scenario-low-noise/` | `WORLD_005_LOW_NOISE` |
| `validation/worlds/WORLD-006-scenario-high-collinearity/` | `WORLD_006_HIGH_COLLINEARITY` |
| `validation/worlds/WORLD-007-scenario-replay-drift/` | `WORLD_007_REPLAY_DRIFT` |

## API

```python
from pathlib import Path
from mmm.validation.synthetic.scenario_builder import (
    WORLD_005_LOW_NOISE,
    build_world_truth,
    write_scenario_world,
)
from mmm.validation.synthetic import materialize_world, validate_bundle

write_scenario_world(Path("validation/worlds/WORLD-005-scenario-low-noise"), WORLD_005_LOW_NOISE)
materialize_world("validation/worlds/WORLD-005-scenario-low-noise", overwrite=True)
assert validate_bundle("validation/worlds/WORLD-005-scenario-low-noise", max_level=3).passed
```

## Related

- [world_materialization.md](world_materialization.md) §10  
- [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) §10 Phase 3B  
- [generators.py](../../mmm/validation/synthetic/generators.py) — `compose_archetype_truth`
