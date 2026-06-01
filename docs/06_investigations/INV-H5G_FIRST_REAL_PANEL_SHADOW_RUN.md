# INV-H5G — First Real-Panel Bayes-H5 Shadow Run

**Investigation ID:** INV-H5G  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5f harness (`280781f`)  
**Manifest:** [H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md](H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md)  
**Artifact:** [BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)

---

## 1. Did the run complete?

**Yes.** The H5f harness executed `run_sandbox_fit` on `examples/sample_panel.csv` with `artifact_type=real_panel_shadow_artifact`, fast MCMC (200/200/2 chains), and wrote the archive JSON. PyMC sampling finished in ~8s wall time with sampler warnings (divergences, low ESS, high R-hat).

---

## 2. What warnings fired?

| Source | Codes / signals |
|--------|-----------------|
| **trust_report_candidate_diagnostics** | `h5:production:block`, `h5:transform_mismatch:adstock` |
| **h5_transform_diagnostics** | `transform_mismatch_detected=true` (generative label `shadow_panel` vs identity channel transforms — harness labeling, not client FE) |
| **PyMC** | 9 divergences; `rhat_max=2.09`; `ess_bulk_min=3` |

`h5:recovery_candidate:stable_research_only` did **not** fire because mismatch was detected.

---

## 3. Were diagnostics useful or noisy?

**Mixed.**

- **Useful:** Convergence and pooling blocks populated; `mu_channel_mean` / `tau_channel_mean` per channel; production block flag present; artifact validates against H5e schema.
- **Noisy:** Transform mismatch warning on a real panel with declared `aligned` mode — driven by `h5_generative_transform=shadow_panel` stub, not client transform policy. Should be refined before multi-panel campaigns.
- **Not useful yet:** No integrated calibration likelihood; trust mapping is candidate-only.

---

## 4. How did H5 compare to Ridge?

Diagnostic Ridge BO (research, 6 trials, geometric adstock + hill, same 123 rows):

| Metric | Ridge (in-sample) |
|--------|---------------------|
| MAE | ~75.1 |
| WMAPE | ~13.6% |

H5 sandbox (identity media transforms, hierarchical partial pooling):

| Channel | `mu_channel_mean` (log-scale sandbox) |
|---------|--------------------------------------|
| search | +0.09 |
| social | +0.33 |
| tv | **−0.31** |

**Interpretation:** Not apples-to-apples — Ridge uses adstock/saturation; H5 path used identity transforms per H5g manifest. Negative TV posterior mean with poor MCMC convergence is **not** decision-grade and should not be compared to Ridge coefficients directly. Ridge in-sample fit is a **sanity baseline** only (`decision_grade=false`).

---

## 5. How did H5 compare to GeoX/CLS?

**Not available.** `geox_cls_comparison.available=false`; no experiment evidence on `examples/sample_panel.csv`.

---

## 6. Did any output look implausible?

| Observation | Assessment |
|-------------|------------|
| `rhat_max=2.09`, `ess_bulk_min=3` | Unacceptable for promotion; expected under fast MCMC + 3 geos |
| Negative `mu_channel_mean` for `tv` | Plausible sampling noise / weak ID; **do not** interpret as business insight |
| `sigma_mean` very small (~0.002) | Suggests overconfident likelihood scale on this panel — review sandbox likelihood scaling |
| 9 NUTS divergences | Re-run with extended MCMC before trusting posteriors |

---

## 7. Was runtime acceptable?

**Yes for research smoke.** ~8s sampling + negligible I/O. Extended MCMC (H5c profile) would be required before treating posteriors as stable evidence.

---

## 8. What should be fixed before more panels?

1. **Transform mismatch labeling** on real panels when `transform_mismatch_mode=aligned` (avoid false `h5:transform_mismatch:adstock` from `shadow_panel` generative stub).
2. **Default panel_schema** documentation in H5f CLI help (added `panel_schema` in transform JSON for this run).
3. **Extended MCMC option** for real-panel archives (not only fast profile).
4. **Automated Ridge contrast** in harness (optional diagnostic hook) with parsed week columns.
5. **Client panels** with immutable snapshot IDs, GeoX stubs, and documented transform alignment to prod FE.
6. **In-sample H5 predictive metric** for symmetric Ridge contrast (not implemented in sandbox v1).

---

## 9. Production boundary

| Item | Status |
|------|--------|
| Production Bayes | **Blocked** |
| Prod TrustReport wiring | **Not done** |
| Optimizer / DecisionSurface | **Absent** from artifact |
| `approved_for_prod` / `prod_decisioning_allowed` | **false** |
| Ridge production path | **Unchanged** |

This run is **first real-panel research evidence** on an in-repo illustrative panel — not client production sign-off or decision grade output.
