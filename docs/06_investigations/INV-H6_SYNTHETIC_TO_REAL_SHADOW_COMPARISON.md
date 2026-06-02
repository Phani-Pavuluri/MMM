# INV-H6: Synthetic-to-real shadow comparison

**Lane:** Bayes-H6 (research only)  
**Date:** 2026-06-01  
**Status:** Initial pilot evidence

## Purpose

Summarize what production-shaped **synthetic** H6 evidence says, compare with **real-panel** H5 shadow behavior (H5m/H5o/H5q/H5r), and list what remains blocked before any promotion discussion.

## H6 synthetic evidence (pilot scale)

Pilot worlds (`WORLD-H6-PILOT-*`) use 20 DMAs × 52 weeks × 7 media channels with:

- Known μ/τ/β and geo baselines
- Adstock + Hill saturation generative path
- Collinearity block (`display`, `ctv`, …)
- Sparse tail geos for `radio` / `local_flyer`
- National `tv` with weak geo noise
- Vertical controls (retail / CPG / auto profiles)

**Ridge (H6c):** Runs in `RunEnvironment.RESEARCH` with controls in schema. Reports prediction RMSE, geo-fold stability, collinearity sensitivity, coarse coef-vs-μ sign recovery.

**Bayes-H5 (H6c):** Same panel via `run_sandbox_fit` + H5 model spec. Reports convergence, β/μ recovery vs truth, posterior interval behavior, H5 diagnostic warnings (transform mismatch expected).

**H6d confounding:** Compare `full_controls` vs `omitted_controls` vs `media_correlated_controls`. Omitted required controls emit **forbidden claims** — do not attribute omitted control lift to media.

## Real-panel shadow comparison

| Panel | Shadow ID | Collinearity | Policy | Outcome | H6 synthetic analogue |
|-------|-----------|--------------|--------|---------|------------------------|
| `examples_mmm_sample_panel_v1` | H5m | High | drop tv | Converged | Collinearity block + high corr warning |
| `examples_mmm_benchmark_geo_panel_v1` | H5o | Low | keep all | Converged | Full-controls retail pilot |
| `examples_mmm_triangulation_geo_panel_v1` | H5q | Moderate | keep all | **Failed convergence** (14 div) | Sparse `radio` tail geos |
| Same triangulation | H5r | Moderate | drop_sparse_channels | Converged (diagnostic) | Sparse channel + policy drop |

### Alignment

- **Sparse channels:** H6 sparse tail geos mimic H5q failure mode; H5r `drop_sparse_channels` is the governed real-panel remedy — H6 does not auto-drop; benchmark harness records collinearity/sparsity flags for Ridge and H5 warnings for Bayes.
- **Collinearity:** H5m and H6 collinearity blocks both trigger high channel correlation diagnostics; Ridge BO tuning may still fit — Bayes-H5 warns on mismatch.
- **Controls:** Real panels use client-specific controls; H6b vertical profiles approximate required/optional sets. Omitted-control stress (H6d) has no direct H5m/o/q/r replay — **gap remains** for real omitted-control audits.

### Divergence

- **Scale:** Real panels are client-sized; H6 pilot is intentionally smaller for CI. Production-scale H6 (`200×130`) is for offline runs only until runtime budgeted.
- **Transforms:** H6 generative uses adstock+saturation; H5 sandbox semi_log on raw media — same limitation as H5 validation worlds.
- **Calibration stubs:** H5q used calibration conflict probes; H6 pilot worlds do not embed calibration signals (future H6 extension).

## What synthetic evidence supports

1. Ridge and H5 can be benchmarked on identical known-truth panels with governed flags.
2. Omitted-control stress produces explicit forbidden claims for downstream reporting.
3. Sparse/geo-tail channel configurations reproduce **classes** of real-panel fragility seen in H5q.

## What remains blocked (no promotion)

1. **H5q without sparse policy** — failed convergence on real triangulation panel.
2. **Transform alignment** — Bayes-H5 MVP vs adstock+saturation generative truth.
3. **Real omitted-control governance** — H6d synthetic only; not replayed on client panels.
4. **Optimizer / DecisionSurface** — out of H6 scope by design.
5. **Production Bayes / recommendations / Ridge replacement** — explicitly blocked in all H6 artifacts.

## Recommended next steps (research)

1. Run `build_h6_benchmark_artifact(WORLD-H6-PILOT-RETAIL-FULL-CONTROLS)` at `production` scale offline; archive under `docs/05_validation/archives/`.
2. Add calibration-stub variant to H6 for H5q-style conflict probes.
3. Wire H6 confounding comparison into continuous validation (research job only).

## Artifacts

- ADR: `docs/05_validation/bayes_h6_synthetic_lane_adr.md`
- Code: `mmm/research/h6_synthetic/`
- Tests: `tests/research/test_h6_synthetic_lane.py`
