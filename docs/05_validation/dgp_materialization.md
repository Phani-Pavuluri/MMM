# Phase 4B-1 — Rich DGP world materialization

Deterministic panel generation where observed KPI is **computed from** `world_truth` using the canonical Ridge stack:

- **semi_log** outcome: `log(y) = intercept + Σ β_c · hill_c(adstock_c(spend)) + ε`
- **geometric** adstock per channel
- **Hill** saturation per channel

This phase proves the **rich panel is generated correctly from truth**. It does **not** prove that the MMM estimator recovers coefficients (Phase 4B-2).

## Module

`mmm/validation/synthetic/dgp_materializer.py` — `materialize_dgp_world()`, `compute_dgp_series()`.

| Constant | Value |
|----------|--------|
| `DGP_MATERIALIZATION_VERSION` | `dgp_materialize_v1.0.0` |

## Canonical world

`validation/worlds/WORLD-008-exact-recovery/` — 3 geos, 3 channels, 16 weekly periods, `observation_noise_std: 0`.

## Bundle artifacts

| File | Role |
|------|------|
| `world_truth.json` | **Immutable** authoritative truth (never written by materializer) |
| `panel.parquet` | Derived KPI + channel spend for train/decide |
| `dgp_diagnostics.parquet` | **Derived** long-form diagnostics (not truth) |
| `dgp_diagnostics_manifest.json` | Marks diagnostics as non-authoritative + formula doc |
| `metadata.json` | `dgp_materialization: true`, version `dgp_materialize_v1.0.0` |
| `checksums.json` | Includes `dgp_diagnostics_sha256` |
| `decision_truth.json` | Optional scenario index (no duplicate β) |

### Diagnostics columns

Per `(geo, week, channel)`: `raw_spend`, `adstocked_spend`, `saturated_feature`; per `(geo, week)`: `linear_predictor`, `log_kpi`, `generated_kpi`, `noise_epsilon`. Column `derived_artifact: true` on all rows.

## Usage

```python
from pathlib import Path
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.validator import validate_bundle

bundle = Path("validation/worlds/WORLD-008-exact-recovery")
materialize_dgp_world(bundle, overwrite=True)
assert validate_bundle(bundle, max_level=3).passed
```

## Relationship to smoke materializer

| | `materialize_world` (2A) | `materialize_dgp_world` (4B-1) |
|--|---------------------------|----------------------------------|
| KPI | Constant `base_level_mean` | Equation-backed from coef + transforms |
| Diagnostics | None | `dgp_diagnostics.parquet` |
| Version | `materialize_v1.0.0` | `dgp_materialize_v1.0.0` |

## Phase 4B-2 — Recovery certification ✅

`mmm/validation/synthetic/recovery_certification.py` trains Ridge BO on the DGP panel and checks:

| Check | Kind |
|-------|------|
| REC-4B2-001 | Coefficient recovery vs `coefficient_truth` (TBD_v1_runtime) |
| REC-4B2-002/003 | Fitted adstock/Hill vs shared truth hyperparameters |
| REC-4B2-004 | Exact geometric + Hill formula on panel spend |
| REC-4B2-005 | Δμ vs analytic `decision_truth` (placeholder `0.0` replaced at runtime) |
| REC-4B2-006 | Optimizer recovery **skipped** (`requires_optimizer_truth_thresholds`) |
| REC-4B2-007 | Train↔decide panel fingerprint match |
| REC-4B2-008 | `simulate_decision` contract (research path) |

Invoked automatically from `run_world_certification()` on `WORLD-008-exact-recovery`. Fixture: `train_config.yaml` in the world bundle.

## Phase 4B-3 — Optimizer recovery world ✅

`validation/worlds/WORLD-009-optimizer-recovery/` — two channels (`high_return`, `low_return`), zero noise, fixed total budget **40**.

| Design choice | Rationale |
|---------------|-----------|
| `spend_process_spec.kind: channel_modulated` | Week-varying spend per channel so Ridge can identify separate coefficients |
| Grid-search `decision_truth` | `true_optimal_budget`, `expected_allocation_band`, `true_optimal_delta_mu` recorded under truth coef simulate |
| `mmm/validation/synthetic/optimizer_truth.py` | `grid_search_true_optimum`, `validate_optimizer_surface`, `build_world_009_truth` |

Recovery (`recovery_certification.py`) trains Ridge with **truth-pinned** adstock/Hill, then runs production `optimize_budget_via_simulation` and executes **VAL-005** (not skipped).

## Phase 4B-4 — Replay calibration recovery world ✅

`validation/worlds/WORLD-010-replay-recovery/` — one treated channel (`search`), 3 geos (experiment on G0/G1), 16 weeks with pre-window impulse for adstock carryover, zero noise.

| Design choice | Rationale |
|---------------|-----------|
| `spend_process_spec.kind: pre_impulse_constant` | Pre-window spend impulse preserves adstock into experiment window |
| Experiment spend shock | `spend_shock.observed_multiplier` on treated geos inside experiment window |
| `counterfactual_spend_multiplier` | Full-panel cf path: obs spend × multiplier inside estimand mask only |
| `mmm/validation/synthetic/replay_truth.py` | Authoritative true lift via `implied_lift_from_counterfactual` + truth coefs |
| `replay_units.json` | Materialized from `experiment_truth` via `build_replay_units_payload` |

Recovery trains Ridge with truth-pinned transforms, builds full-panel replay frames, compares fitted replay-implied lift to `experiment_truth`, and executes **VAL-006**.

## Phase 4B-5 — Drift and identifiability recovery ✅

| World | Focus |
|-------|--------|
| `WORLD-011-drift-recovery` | Coefficient changepoint in KPI (`drift_truth`); pre-period train + post-period degradation |
| `WORLD-012-identifiability-recovery` | `collinear_block` spend; VIF/collinearity warning; coef recovery skipped |

Materializer additions: `_effective_betas_at_week_index`, `collinear_block` spend with week jitter, `effective_beta` in diagnostics.

`mmm/validation/synthetic/reliability_truth.py` — world builders. Recovery: `REC-4B5-DRIFT` (partial VAL-012), `REC-4B5-ID` (partial VAL-013/014).

**Limitation:** VAL-012 full `drift_detection_runner` not landed — see INV-055.

## Next phase

**Phase 4C** — ReliabilityScorecard MVP over WORLD-008–WORLD-012.
