# Behavioral lattice sweep (Phase 5B MVP)

**Status:** Implemented — `mmm/validation/synthetic/behavioral_lattice_sweep.py`  
**Version:** `behavioral_lattice_sweep_v1.0.0`  
**Report artifact:** `validation/reports/behavioral_lattice_sweep_mvp_report.json`

## Purpose

Extend the Phase 5A lattice machinery to **behavioral recovery** on a small fixed grid of **rich DGP** worlds. Measures recovery metrics across controlled scenario axes — not Monte Carlo sampling.

## Pipeline

```text
mvp_behavioral_lattice_specs()
        │
        ▼
build_behavioral_world_truth()  (WORLD-008–012 patterns)
        │
        ▼
materialize_dgp_world()  ← never smoke materializer
        │
        ▼
run_world_certification(include_recovery=True)
        │
        ▼
build_scorecard_from_reports + per-axis recovery summaries
        │
        ▼
behavioral_lattice_sweep_mvp_report.json
```

## MVP grid (10 worlds)

| world_type | noise | correlation | drift | replay |
|------------|-------|-------------|-------|--------|
| exact_recovery | zero, low (×2), low+severe | off | off |
| optimizer | zero, low | off | off |
| replay | zero, low | off | on |
| drift | zero, low | on | off |
| identifiability | low | severe | off | off |

World IDs: `L5B-{world_type}-noise-{zero|low}-corr-{low|severe}-drift-{on|off}-replay-{on|off}`

Bundles: `validation/worlds/behavioral-lattice/<world_id>/`

## Behavioral vs structural scoring

| Score | Meaning |
|-------|---------|
| **behavioral_score** | Mean recovery-metric score on in-scope behavioral checks only |
| **structural_score** | CERT-4A / contract aggregate (`lattice_structural` mode) |
| **coverage_ratio** | Fraction of in-scope behavioral metrics that received a scored outcome |

Skipped expected checks (e.g. `REC-4B5-ID-COEF`, `REC-4B5-DRIFT-COEF`) do not penalize behavioral score.

## Recovery metrics (per world)

- `coefficient_recovery_status`
- `delta_mu_recovery_status`
- `optimizer_recovery_status`
- `replay_recovery_status`
- `drift_behavior_status` (partial when VAL-012 registry row is partial — INV-055)
- `identifiability_behavior_status`
- `contract_compatibility_status`

## Rules

- **Rich DGP only** for behavioral cells (`behavioral_mode=behavioral`).
- Unsupported axis combos → `behavioral_mode=unsupported` (no fake recovery pass).
- Drift worlds: coefficient recovery skipped; drift behavior scored.
- Identifiability worlds: coefficient recovery skipped; governance/identifiability scored.
- Failures preserved in report `failures` / `partials` / `skips`.

## API

```python
from pathlib import Path
from mmm.validation.synthetic.behavioral_lattice_sweep import (
    mvp_behavioral_lattice_specs,
    run_behavioral_lattice_sweep,
    write_behavioral_lattice_sweep_report,
)

write_behavioral_lattice_sweep_report(Path("."))
```

## Limitations

- Small fixed grid — not statistically representative.
- `TBD_v1_runtime` tolerances — not production gates.
- VAL-012 remains partial without dedicated `drift_detection_runner`.
- No causal incrementality claims.

## Related

- [lattice_sweep.md](lattice_sweep.md) — Phase 5A structural lattice  
- [reliability_scorecard.md](reliability_scorecard.md)  
- [dgp_materialization.md](dgp_materialization.md)

## Next phase

**Phase 5C** ✅ — [exact recovery investigation](exact_recovery_investigation.md) (INV-056). **5D** ✅ — [reliability threshold governance](reliability_threshold_governance.md). **5E** drift runner (next), **5F** Monte Carlo.
