# H5Q — Third Real-Panel Shadow Run Manifest

**Milestone:** Bayes-H5q  
**Status:** Authorized research-only third real-panel shadow  
**Date:** 2026-06-01  
**Prerequisites:** H5p audit @ `0b918f0` ([AUDIT-H5P](../audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md))

---

## Panel selection

| Criterion | Choice |
|-----------|--------|
| **Panel** | `examples/triangulation_geo_panel_v1.csv` |
| **panel_id** | `examples_mmm_triangulation_geo_panel_v1` |
| **Distinct from H5m** | Not high collinearity drop-tv pilot (3 geos, 3 channels) |
| **Distinct from H5o** | Not low-collinearity keep-all benchmark (4 geos, 3 channels) |
| **New conditions tested** | **8 geos**, **4 media channels**, **moderate collinearity** (max \|ρ\| ≈ 0.94), **sparse radio**, **calibration stub** for GeoX/CLS triangulation |
| **Why safe** | Public frozen CSV in git; deterministic DGP; no client confidentiality |
| **DGP note** | Research triangulation panel — not H4 recovery world, not client extract |

---

## Lineage

| Field | Value |
|-------|--------|
| **dataset_snapshot_id** | `mmm-examples-triangulation-geo-panel-frozen-2022-v1` |
| **source path** | `examples/triangulation_geo_panel_v1.csv` |
| **Date range** | 2022-01-03 through 2022-12-26 (weekly) |
| **Row count** | 416 (8 geos × 52 weeks) |
| **Geo grain** | `geo_id` (G0–G7) |

---

## Panel schema

| Role | Column |
|------|--------|
| Geo | `geo_id` |
| Week | `week_start_date` |
| Outcome | `revenue` |
| Media | `search`, `social`, `display`, `radio` |
| Controls | *(none)* |

Schema file: [h5q_triangulation_panel_schema.json](h5q_triangulation_panel_schema.json)

---

## Diagnostics (pre-run)

| Diagnostic | Value |
|------------|--------|
| max \|ρ\| | ~0.943 (< 0.95 threshold) |
| High collinearity flag | false (borderline moderate) |
| Radio sparsity | ~99% near-zero spend |
| Calibration stub | [h5q_triangulation_calibration_stub.json](h5q_triangulation_calibration_stub.json) |

---

## Comparisons

| Comparison | Available |
|------------|-----------|
| **Ridge** | Not executed in H5q (optional per protocol) |
| **GeoX/CLS** | **Stub** on policy (`geox_cls_comparison.available=true`; likelihood not integrated) |

---

## Caveats

- Illustrative triangulation panel — not production sign-off.
- Keep-all recommended despite sparse radio; single-channel diagnostic listed as alternative.
- Shadow replay may fail convergence with 4 channels × 8 geos — record honestly per AUDIT-H5P.

---

## Privacy / confidentiality

**Public in-repo.** No production secrets in committed artifacts.

---

## Authorization / production boundary

- Research lane only; recommender-first; stop on `do_not_run`.
- Production Bayes, optimizer, DecisionSurface, recommendations: **blocked**.

---

## Artifacts

| Artifact | Path |
|----------|------|
| Recommendation | [BAYES_H5Q_SHADOW_POLICY_RECOMMENDATION_…](../05_validation/archives/BAYES_H5Q_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json) |
| Frozen policy | [h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy.json](h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy.json) |
| Shadow run | [BAYES_H5Q_SHADOW_RUN_…](../05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json) |
| Investigation | [INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md](INV-H5Q_THIRD_REAL_PANEL_SHADOW_RUN.md) |
