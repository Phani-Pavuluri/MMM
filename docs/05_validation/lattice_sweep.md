# Lattice sweep (Phase 5A MVP)

**Status:** Implemented — `mmm/validation/synthetic/lattice_sweep.py`  
**Version:** `lattice_sweep_v1.0.0`  
**Report artifact:** `validation/reports/lattice_sweep_mvp_report.json`

## Purpose

Run a **fixed, deterministic grid** of ScenarioBuilder worlds through materialization, structural certification, and scorecard aggregation to measure **reliability by scenario axis** (noise, collinearity, replay, drift, experiment quality).

This is **not** Monte Carlo sampling. It is a controlled lattice that shows where capability is still structural vs behavioral.

## Pipeline

```text
mvp_lattice_specs()  →  ScenarioSpec (12 worlds)
        │
        ▼
write_scenario_world()  →  world_truth.json
        │
        ▼
materialize_world()  →  panel / replay / checksums
        │
        ▼
run_world_certification(include_recovery=False)
        │
        ▼
build_scorecard_from_reports(mode=lattice_structural)
        │
        ▼
lattice_sweep_mvp_report.json
```

## MVP lattice (12 worlds)

| Family | Cells |
|--------|--------|
| `baseline` (6) | noise × correlation × drift: low/low/off, low/severe/off, high/low/off, high/severe/off, low/low/on, high/severe/on |
| `replay` (6) | same pattern with `experiment_quality=medium` |

Axes encoded in `world_id`:

```text
L5A-{family}-noise-{low|high}-corr-{low|severe}-drift-{on|off}-eq-{none|medium}
```

Worlds are written under `validation/worlds/lattice/<world_id>/` (generated, not hand-edited).

## Scoring

- **Structural focus:** CERT-4A checks and contract compatibility.
- **Deferred VAL-* rows:** skipped without penalty (`lattice_structural` scorecard mode).
- **Per-axis summary:** certification pass/fail rates grouped by `noise_level`, `correlation_level`, `drift`, `family`, `experiment_quality`.

## Report fields

| Field | Description |
|-------|-------------|
| `sweep_id` | `lattice_sweep_mvp` |
| `sweep_version` | `lattice_sweep_v1.0.0` |
| `lattice_axes` | Axis value domains |
| `per_world_status` | truth / materialize / certify outcome per world |
| `per_axis_summary` | Reliability by axis |
| `failure_taxonomy` | structural / replay / contract / recovery / governance / skipped |
| `skipped_validation_summary` | Deferred VAL rows by reason |
| `limitations` | MVP scope boundaries |
| `recommended_followups` | Phase 5B+ |

## API

```python
from pathlib import Path
from mmm.validation.synthetic.lattice_sweep import (
    mvp_lattice_specs,
    run_lattice_sweep,
    write_lattice_sweep_report,
)

report = run_lattice_sweep(Path("."))
write_lattice_sweep_report(Path("."))
```

## Limitations (MVP)

- Smoke materializer panels (constant KPI) — axes declared in truth, not full DGP simulation.
- No behavioral recovery scoring on lattice worlds (VAL-001–006 deferred).
- Small fixed grid — not statistically representative.
- Does not gate production releases.

## Related

- [scenario_builder.md](scenario_builder.md)  
- [reliability_scorecard.md](reliability_scorecard.md)  
- [certification_runner.md](certification_runner.md)  
- [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) §11 Phase 5A

## Next phase

**Phase 5B** ✅ — [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md). **Phase 5C** — [exact recovery investigation](exact_recovery_investigation.md).
