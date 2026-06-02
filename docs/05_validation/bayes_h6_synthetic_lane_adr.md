# ADR: Bayes-H6 production-shaped synthetic validation lane

**Status:** Accepted (research lane)  
**Date:** 2026-06-01  
**Scope:** H6a–H6e synthetic validation; no production promotion

## Context

Bayes-H5 shadow workflow (H5m–H5r) established governed real-panel diagnostics. Existing recovery worlds (H4/H5) are useful for transform and pooling diagnostics but are not production-shaped: they use few geos/weeks and omit vertical-specific controls.

## Decision

Add a dedicated **H6 synthetic validation lane** under `mmm/research/h6_synthetic/` that:

1. **H6a** — Generates production-shaped DMA panels (~200 geos × 104–156 weeks at `production` scale; `pilot` scale for CI).
2. **H6b** — Encodes vertical control profiles (retail, CPG, auto) with known control effects and stress variants.
3. **H6c** — Benchmarks **production Ridge** (research environment) and **Bayes-H5 sandbox** on the same worlds.
4. **H6d** — Runs confounding / omitted-control stress comparisons with explicit forbidden claims.
5. **H6e** — Documents synthetic-to-real shadow comparison and promotion blockers.
6. **H6f** — Ridge vs H5 benchmark matrix + control-confounding summary artifacts.

## Boundaries (non-negotiable)

| Capability | H6 lane |
|------------|---------|
| Production Ridge baseline | Yes (research `RunEnvironment`) |
| Bayes-H5 sandbox | Yes (explicit `enable_h5_sandbox`) |
| Production Bayes | **No** |
| Optimizer / DecisionSurface | **No** |
| Recommendations | **No** |
| Ridge replacement | **No** |

All artifacts carry `production_flags.approved_for_prod=false` and `outputs_are_diagnostic_only=true`.

## World IDs (pilot registry)

| World ID | Vertical | Stress |
|----------|----------|--------|
| `WORLD-H6-PILOT-RETAIL-FULL-CONTROLS` | retail | full_controls |
| `WORLD-H6-PILOT-RETAIL-OMITTED-CONTROLS` | retail | omitted_controls |
| `WORLD-H6-PILOT-RETAIL-MEDIA-CORRELATED-CONTROLS` | retail | media_correlated_controls |
| `WORLD-H6-PILOT-CPG-FULL-CONTROLS` | cpg | full_controls |
| `WORLD-H6-PILOT-AUTO-OMITTED-CONTROLS` | auto | omitted_controls |

`production` scale (200 DMA × 130 weeks) is available via `build_production_shaped_world(scale="production")` for offline artifact generation; CI uses `pilot` scale.

## Generative truth

Known truth is recorded in `materialize_h6_truth_artifact()`:

- μ/τ/β channel hierarchy
- geo baselines (`true_alpha_g`)
- adstock/saturation (`transform_truth`)
- collinearity blocks, sparse/local channels, national TV weak variation
- seasonality and shocks
- control effects by vertical profile
- noise (`noise_sigma`)

H5 sandbox fits raw standardized media (semi_log MVP); generative path uses adstock+saturation — **transform mismatch is expected** and flagged in diagnostics (aligned with H5 validation worlds).

## Promotion blockers (H6e summary)

Synthetic evidence **does not** override real-panel shadow outcomes. Before any promotion discussion:

1. H5q/H5r triangulation panel: sparse-channel convergence remains fragile without explicit policy.
2. H6 transform mismatch: Bayes-H5 on raw media vs adstock+saturation generative — recovery metrics are diagnostic only.
3. Omitted-control worlds: forbidden claims apply; Ridge/H5 must not be used to assert incrementality without required controls.
4. No optimizer/DecisionSurface validation in H6 — budget decisioning untested.
5. Production Bayes, recommendations, and Ridge replacement remain **blocked**.

See `docs/06_investigations/INV-H6_SYNTHETIC_TO_REAL_SHADOW_COMPARISON.md`.

## H6f — benchmark matrix (2026-06-01)

`build_h6f_benchmark_matrix()` and `build_h6f_control_confounding_summary()` in `mmm/research/h6_synthetic/benchmark_matrix.py`.

Archives:

- `docs/05_validation/archives/BAYES_H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX_20260601.json`
- `docs/05_validation/archives/BAYES_H6F_CONTROL_CONFOUNDING_SUMMARY_20260601.json`

Investigation: `docs/06_investigations/INV-H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX.md`

### H5r / H6 recommender guidance (shadow policy)

1. **Extreme `near_zero_share` (≥ 0.99):** prefer `drop_sparse_channels` or `do_not_run` over `keep_all` unless explicitly justified with convergence evidence (H5r triangulation: keep-all failed; sparse drop converged).
2. **Dropped sparse channels:** emit forbidden separate-effect claims per dropped channel.
3. **Calibration stubs:** diagnostic only — do not automatically recover sparse-channel effects or override sparse-drop policy.

Implemented in `h5_shadow_policy_recommender.py` (`CHANNEL_DROP_SPARSE`).

## H7 — Ridge production diagnostics (2026-06-01)

H6f recommended hardening the **production Ridge baseline** before further H5 transform work.

- Contract: [ridge_production_diagnostics_contract.md](ridge_production_diagnostics_contract.md)
- Module: `mmm/diagnostics/ridge_diagnostics.py`
- Vertical profiles (shared): `mmm/config/vertical_control_profiles.py`
- Example artifact: `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json`

## References

- `mmm/research/h6_synthetic/production_shapes.py`
- `mmm/research/h6_synthetic/vertical_controls.py`
- `mmm/research/h6_synthetic/benchmark_harness.py`
- `mmm/research/h6_synthetic/benchmark_matrix.py`
- H5 shadow gate: `docs/audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md`
