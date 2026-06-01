# H5G — First Real-Panel Shadow Run Manifest

**Milestone:** Bayes-H5g  
**Status:** Authorized research-only first real-panel shadow  
**Date:** 2026-06-01  
**Harness:** `mmm/research/bayes_h3_sandbox/h5_shadow_runner.py` (commit `280781f`+)

---

## Panel selection rationale

| Criterion | Choice |
|-----------|--------|
| **Panel** | `examples/sample_panel.csv` — public in-repo historical-style MMM panel used by prod train template |
| **Why safe** | No client confidentiality; not tied to live planning; stable CSV in version control |
| **Why not worlds** | `validation/worlds/*` panels are synthetic DGP materializations (H5e exclusions) |
| **Weeks / geos** | 3 geos (G0–G2), 41 weekly rows each (~123 rows) |
| **Caveat** | Illustrative panel — evidence is **research lane first real-panel**, not client production sign-off |

---

## Lineage

| Field | Value |
|-------|--------|
| **panel_id** | `examples_mmm_sample_panel_v1` |
| **dataset_snapshot_id** | `mmm-examples-sample-panel-frozen-2022` |
| **source path** | `examples/sample_panel.csv` |
| **content hash** | Recorded in shadow artifact `data_snapshot_hash` |
| **Prod template snapshot** | `prod-template-v1` (`examples/prod_train_template.yaml` `data.data_version_id`) |

---

## Panel schema

| Role | Column |
|------|--------|
| Geo | `geo_id` |
| Week | `week_start_date` |
| Outcome | `revenue` |
| Media | `search`, `social`, `tv` |
| Controls | *(none)* |

**Date range:** 2022-01-03 through 2022-10-10 (weekly, per geo)  
**Geo grain:** `geo_id` (G0, G1, G2)

---

## Transform config

File: [h5g_sample_panel_transform_config.json](h5g_sample_panel_transform_config.json)

- H5 sandbox: **identity** on all media channels (`transform_mismatch_mode=aligned`)
- Prod Ridge template: geometric adstock + hill (`examples/prod_train_template.yaml`) — **not** applied in H5 shadow path; documented comparison gap only

---

## Calibration signals

| Field | Value |
|-------|--------|
| **Stub slots** | Empty for first run |
| **likelihood_integrated** | `false` |
| **GeoX/CLS** | Not available on this panel |

---

## Ridge comparison

| Field | Value |
|-------|--------|
| **Availability** | Yes — same panel path and schema as `examples/minimal_train.yaml` / prod template |
| **Mode** | Diagnostic-only (`decision_grade=false`, `used_for_optimizer=false`) |
| **Run** | Research Ridge BO with reduced trials (post-shadow diagnostic script) |

---

## GeoX / CLS comparison

| Field | Value |
|-------|--------|
| **available** | `false` |
| **note** | No experiment evidence fixtures attached to `examples/sample_panel.csv` |

---

## Known caveats

1. Illustrative panel — not a client production dataset.
2. H5 uses identity media transforms; Ridge template uses adstock/saturation — contrasts are **not** apples-to-apples on transforms.
3. Fast MCMC profile (200/200/2 chains) — convergence may be noisy; not extended MCMC.
4. Partial pooling with 3 geos — weak hierarchy signal expected.
5. Output is **not decision grade** and must not feed optimizer or prod TrustReport.

---

## Authorization boundary

| Allowed | Blocked |
|---------|---------|
| Research sandbox `run_sandbox_fit` | Production pipelines |
| Archive JSON under `docs/05_validation/archives/` | `approved_for_prod`, `prod_decisioning_allowed`, hard gates |
| H5d `trust_report_candidate_diagnostics` | Prod TrustReport wiring, DecisionSurface, recommendations |
| Diagnostic Ridge contrast | Optimizer input from H5 posterior |

**Production Bayes remains blocked.** Ridge production path unchanged.
