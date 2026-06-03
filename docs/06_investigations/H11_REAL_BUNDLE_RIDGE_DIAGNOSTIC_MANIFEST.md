# H11 — Real-Bundle Ridge Diagnostic Manifest

**Milestone:** H11  
**Status:** Active (diagnostic hardening)  
**Date:** 2026-06-01  
**Prerequisite:** MIP-C1 @ `df54dd1`, H10 @ `3ed159d`

---

## Bundle: `MMM-BENCHMARK-GEO-PANEL-V1`

| Field | Value |
|-------|--------|
| **bundle_id** | `MMM-BENCHMARK-GEO-PANEL-V1` |
| **panel_id** | `examples_mmm_benchmark_geo_panel_v1` |
| **dataset_snapshot_id** | `mmm-examples-benchmark-geo-panel-frozen-2022-v1` |
| **panel / source path** | `examples/benchmark_geo_panel_v1.csv` |
| **panel_content_sha256** | `406d217a92fd0b8a59ffd1ad34d2da2c237fbb7db43af68742ff56571ae7f126` |
| **vertical** | `retail` (illustrative — panel schema has **no** control columns) |
| **vertical assumption** | Retail profile applied for control-completeness stress; not a claim that panel is retail production data |
| **date range** | 2022-01-03 → 2022-12-26 (weekly) |
| **geo grain** | `geo_id` (G0–G3) |
| **geo count** | 4 |
| **row count** | 208 |
| **outcome column** | `revenue` |
| **media columns** | `search`, `social`, `tv` |
| **control columns** | *(none in schema)* |
| **transform metadata availability** | Ridge BO search — `metadata_complete=true` when fit completes; contract requires warning when missing |
| **calibration evidence availability** | `false` — no replay/CalibrationSignal on this run |
| **CalibrationSignal context** | **Absent** (explicit via `evidence_attachment_lineage`) |
| **privacy / confidentiality** | **Public in-repo illustrative** (same lane as H5o second real-panel shadow) |
| **Ridge config** | Research env; geometric adstock + hill; `n_trials=4` (H11 runner) |

### Expected diagnostic risks

| Risk | Rationale |
|------|-----------|
| **Omitted required retail controls** | `promo_flag`, `holiday`, `unemployment_index` absent from panel |
| **Forbidden attribution / budget claims** | H9 policy + H7 forbidden-claims when `omitted_control_risk` |
| **Weak identification** | Possible high cross-media correlation on small geo count |
| **No MIP-C1 attachment** | Default path — C2 may wire ETL later |
| **Not client production sign-off** | Illustrative benchmark panel only |

### Artifacts (H11)

| Artifact | Path |
|----------|------|
| Full diagnostic archive (coef redacted) | [H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json](../05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) |
| Operator summary Markdown | [H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md](../05_validation/archives/H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md) |

### Runner

- Module: `mmm/diagnostics/ridge_real_bundle_hardening.py`
- Entry: `run_h11_benchmark_bundle_archive()` or `run_real_bundle_ridge_diagnostics(BENCHMARK_BUNDLE_SPEC)`

---

## Deferred bundles (not in H11 v1)

| bundle_id | Panel | Notes |
|-----------|-------|-------|
| `MMM-TRIANGULATION-GEO-PANEL-V1` | `examples/triangulation_geo_panel_v1.csv` | Sparse `radio`; calibration stub in H5q — C2 + follow-on H11b |
| `MMM-SAMPLE-PANEL-V1` | `examples/sample_panel.csv` | H5g first real-panel; 3 geos only |

---

## Production boundaries (unchanged)

- Ridge fitting behavior unchanged (research config only on this lane).  
- Optimizer / DecisionSurface / recommendations not emitted.  
- MIP-C1 remains context-only; no live GeoX/CLS ETL in H11.  
- Bayes-H5 research-only.  
- Diagnostics are not hard gates.
