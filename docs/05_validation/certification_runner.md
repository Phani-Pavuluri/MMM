# Phase 4A — Synthetic world certification runner

Structural certification for materialized `GroundTruthWorld` bundles. This layer proves **bundle integrity**, **semantic compatibility**, **replay compatibility**, **governance compatibility**, and **platform-contract preservation** — not modeling recovery.

## Pipeline

```
ScenarioBuilder / generators
  → world_truth.json
  → materializer
  → bundle artifacts (panel, replay_units, checksums, …)
  → certification_runner.run_world_certification()
  → synthetic_world_certification_report.json
```

## Modules

| Module | Role |
|--------|------|
| `mmm/validation/synthetic/certification_registry.py` | CERT-4A-* check catalog; VAL-001–014 deferred rows with explicit skip reasons |
| `mmm/validation/synthetic/certification_runner.py` | Orchestration, truth/contract checks, report emission |

## What Phase 4A proves

- L1–L3 bundle validator passes (`CERT-4A-001`)
- On-disk checksums match recorded digests (`CERT-4A-002`)
- `replay_units.json` loads via `units_io` when experiment units exist (`CERT-4A-003`)
- Canonical transform families in truth (`CERT-4A-004`)
- Bundle `metadata.json` aligns with `world_truth.metadata` (`CERT-4A-005`)
- `artifact_truth.expected_warnings` well-formed (`CERT-4A-006`)
- `decision_truth` scenario structure (`CERT-4A-007`)
- Replay payload shape (`CERT-4A-008`)
- **DecisionSurface** semantics: `semi_log` + full-panel replay transform (`CERT-4A-009`)
- **Estimand** declarations (`CERT-4A-010`)
- **CalibrationSignal** required fields (`CERT-4A-011`)
- **TrustReport** gate expectations in truth (`CERT-4A-012`)
- **Release-gate** governance enums (`CERT-4A-013`)

## What Phase 4A does NOT prove

- Full coefficient/adstock/Hill recovery (except WORLD-008 coef path and WORLD-009 optimizer path)
- Replay calibration recovery (Phase 4B-4)
- Causal validity or experiment effectiveness
- Uncertainty calibration or drift detection accuracy
- Production allocation correctness
- Runtime train/decide artifact tiers

Deferred registry rows (`VAL-001`–`VAL-014`) appear in the report as **`skipped`** with reasons such as `requires_rich_dgp_worlds`, `requires_train_decide_execution`, or `requires_thresholds`. They are **never passed** and **never silently omitted**.

## Certification artifact

**Filename:** `synthetic_world_certification_report.json` (written into the bundle directory).

| Field | Meaning |
|-------|---------|
| `world_id`, `world_version`, `world_contract_version` | Identity from truth metadata |
| `generator_version`, `materialization_version` | Provenance |
| `certification_runner_version` | Runner semver tag (`cert_runner_v1.0.0`) |
| `executed_validations` | CERT-4A checks that ran (not skipped) |
| `skipped_validations` | `{check_id, skip_reason, message}` for deferred/not-applicable |
| `failed_validations` | CERT-4A (or load) failures |
| `validation_results` | Full per-check rows |
| `overall_status` | `pass` \| `fail` \| `error` (deferred skips do not flip to pass) |
| `contract_compatibility` | Aggregate pass/fail over platform contract checks |
| `decision_surface_compatibility`, `replay_compatibility`, `trust_semantics_compatibility` | Rollups |

## Usage

```python
from pathlib import Path
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.materializer import materialize_world

bundle = Path("validation/worlds/WORLD-005-scenario-low-noise")
materialize_world(bundle, overwrite=True)
result = run_world_certification(bundle)
assert result.passed
```

## Roadmap

| Phase | Scope |
|-------|--------|
| **4A** (this doc) | Structural / contract certification on bundles |
| **4B-1** | Rich DGP panel materialization (`dgp_materializer.py`, WORLD-008) — see [dgp_materialization.md](dgp_materialization.md) |
| **4B-2** | Train/decide recovery on WORLD-008 (`recovery_certification.py`) |
| **4B-3** | Optimizer recovery world (`WORLD-009`) + VAL-005 via `recovery_certification` |
| **4B-4** | Replay calibration recovery world (`WORLD-010`) + VAL-006 via `recovery_certification` |
| **4B-5** | Drift (`WORLD-011`) + identifiability (`WORLD-012`) reliability recovery |
| **4C** | ReliabilityScorecard MVP — [reliability_scorecard.md](reliability_scorecard.md) |
| **5A** ✅ | Small ScenarioBuilder lattice sweep — [lattice_sweep.md](lattice_sweep.md) |
| **5B** ✅ | Rich DGP behavioral lattice — [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md) |
| **5C** (next) | Exact recovery investigation — [exact_recovery_investigation.md](exact_recovery_investigation.md) |
| **5D** | Drift detection runner (VAL-012) + drift scorecard integration |
| **5D** | Threshold governance (DR-04) + scorecard metric classes ✅ |
| **5E** | Drift runner (VAL-012) + TrustReport downgrade semantics |

See [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md), [validation_registry.md](validation_registry.md), and [reliability_scorecard.md](reliability_scorecard.md) (Phase 4C aggregate over WORLD-008–012).
