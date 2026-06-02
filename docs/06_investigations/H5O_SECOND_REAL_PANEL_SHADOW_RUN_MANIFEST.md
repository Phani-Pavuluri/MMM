# H5O — Second Real-Panel Shadow Run Manifest

**Milestone:** Bayes-H5o  
**Status:** Authorized research-only second real-panel shadow  
**Date:** 2026-06-01  
**Prerequisites:** H5n complete (`bf98798`)

---

## Panel selection

| Criterion | Choice |
|-----------|--------|
| **Panel** | `examples/benchmark_geo_panel_v1.csv` |
| **panel_id** | `examples_mmm_benchmark_geo_panel_v1` |
| **Why this panel** | Second in-repo frozen benchmark — **not** the H5g–H5m `sample_panel.csv` pilot; tests recommender + frozen-policy workflow on different collinearity geometry |
| **Why safe** | Public, version-controlled CSV; no client confidentiality; no live planning attachment |
| **Why not sample_panel** | Already used for H5g–H5m; H5o requires a distinct panel to test generalization |
| **Why not H4/H5 worlds** | Recovery/stress worlds are synthetic DGP materializations (H5e exclusions) |
| **DGP note** | Rows generated with deterministic `SyntheticGeoPanelSpec` (seed 7) and frozen in git — historical-style geo×week MMM layout, not a client extract |

---

## Lineage

| Field | Value |
|-------|--------|
| **dataset_snapshot_id** | `mmm-examples-benchmark-geo-panel-frozen-2022-v1` |
| **source path** | `examples/benchmark_geo_panel_v1.csv` |
| **content hash** | Recorded in shadow artifact `data_snapshot_hash` |
| **Date range** | 2022-01-03 through 2022-12-26 (weekly, per geo) |
| **Row count** | 208 (4 geos × 52 weeks) |
| **Geo grain** | `geo_id` (G0–G3) |

---

## Panel schema

| Role | Column |
|------|--------|
| Geo | `geo_id` |
| Week | `week_start_date` |
| Outcome | `revenue` |
| Media | `search`, `social`, `tv` |
| Controls | *(none)* |

---

## Diagnostics summary (pre-run)

| Diagnostic | Value |
|------------|--------|
| max \|ρ\| (media) | ~0.061 |
| Collinear groups | none at threshold 0.95 |
| Sparsity | no hard zeros |
| Recommender | **keep_all_channels** + σ-floor hierarchy |

---

## Comparisons

| Comparison | Available |
|------------|-----------|
| **Ridge** | Not run for H5o (optional per protocol; illustrative benchmark panel) |
| **GeoX/CLS** | No |
| **Prior panel (H5m)** | Contrasts drop-tv (high collinearity) vs keep-all (low collinearity) |

---

## Caveats

- Benchmark panel is illustrative — not client production sign-off.
- Keep-all recommendation forbids channel-level budget/causal decision use under weak-ID monitoring.
- `evidence_promotion_allowed` on recommendation is false until shadow run proves convergence.

---

## Privacy / confidentiality

**Public in-repo benchmark.** No production secrets or private client artifacts in committed files.

---

## Authorization boundary

- Research lane only (`run_environment=research`, `enable_h5_sandbox=true`).
- Recommender must not return `do_not_run` before shadow execution (enforced).

---

## Production boundary

- `approved_for_prod=false`, `prod_decisioning_allowed=false`
- No optimizer, DecisionSurface, budget recommendations, prod TrustReport, or Ridge replacement
- σ-floor geometry is **not** a production default

---

## Artifacts

| Artifact | Path |
|----------|------|
| Recommendation | [BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_…](../05_validation/archives/BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) |
| Frozen policy | [h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json](h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json) |
| Shadow run | [BAYES_H5O_SHADOW_RUN_…](../05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) |
| Investigation | [INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md](INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md) |
