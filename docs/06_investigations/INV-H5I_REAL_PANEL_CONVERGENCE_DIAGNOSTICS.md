# INV-H5I — Bayes-H5 Real-Panel Convergence Diagnostics

**Investigation ID:** INV-H5I  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5h shadow hardening (`a42e2ba`)  
**Panel:** `examples/sample_panel.csv` (`examples_mmm_sample_panel_v1`)  
**Artifacts:**
- [BAYES_H5I_CONVERGENCE_DIAGNOSTICS_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5I_CONVERGENCE_DIAGNOSTICS_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)
- [BAYES_H5I_CONVERGENCE_EXPERIMENT_MATRIX_20260601.json](../05_validation/archives/BAYES_H5I_CONVERGENCE_EXPERIMENT_MATRIX_20260601.json)  
**Code:** `mmm/research/bayes_h3_sandbox/h5_convergence_diagnostics.py`

---

## 1. Executive summary

Convergence failure on the first real-panel shadow fit is **not** caused by transform-label semantics (fixed in H5h). Static diagnostics point to **media collinearity + hierarchical over-parameterization + tight residual scale (sigma)**. Controlled matrix experiments on the **same panel only** show:

- **Sampler-only** (`target_accept=0.99`) and **prior tightening** do not clear the evidence bar.
- **Scaling prescale** (z-score media + z-score log outcome) removes divergences but **R-hat remains high** (1.46) — geometry improved, not solved.
- **Single-channel (search)** lowers R-hat to 1.41 but divergences persist.

**Do not run more real panels** until a variant reaches `converged_diagnostic_only` on this pilot panel.

---

## 2. Root-cause read (what kind of failure?)

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| Panel size (123 rows) | 3 geos × 41 weeks — adequate for Ridge; marginal for 3×3 geo-channel partial pooling | **Contributing** |
| Outcome scale | Log revenue std ≈ 0.16 — sane | **Unlikely primary** |
| Media scale | Levels 180–700; identity transform standardizes per channel | **OK after transform** |
| Collinearity | max \|ρ\| ≈ **0.99** (social–tv) | **Primary** |
| Sparsity / zeros | No zero spend | **Ruled out** |
| Transform config | Identity aligned — not mismatch | **Ruled out** |
| Sampler settings alone | Extended MCMC worsened divergences (H5h); target_accept 0.99 did not fix R-hat | **Secondary** |
| Priors too loose | Tighter tau/mu helped slightly; still failed | **Secondary** |
| Model geometry | Worst R-hat on **z_beta**, **tau_channel**, **mu_channel** — hierarchical media coefficients | **Primary** |

**Conclusion:** Failure is predominantly **weak identification / funnel geometry** from **collinear media** in a **small-geo partial-pooling** model, with **very small posterior sigma** (~0.0018) on log scale compounding sampler stress. Not a panel-ingest bug.

---

## 3. Parameter-level diagnostics (H5I baseline replay)

From `H5I-BASELINE-FAST-REPLAY` (fast 200/200/2):

| Family | Worst parameters (R-hat) |
|--------|--------------------------|
| beta_offset (`z_beta`) | up to **2.37** |
| tau | up to **2.09** |
| mu_channel | up to **2.07** |
| intercept (`alpha_geo`) | up to **1.87** |
| sigma | stable vs offsets |

Divergences (9) co-occur with **tau/beta** blocks — consistent with hierarchical funnel pathology, not intercept-only.

H5g/H5h reference: fast R-hat 2.09; extended R-hat 1.69 but **74 divergences** — longer sampling traded one metric for worse geometry exploration.

---

## 4. Feature / scale checks

| Check | Result |
|-------|--------|
| Missing data | None |
| Outcome range | 402–700 (toy upward trend) |
| Media CV | 0.13–0.25 |
| Post-transform scale | Identity → ~0 mean, ~1 std per channel |
| Collinearity | search–social 0.96, search–tv 0.91, social–tv **0.99** |

Media are **not sparse** but are **near-linearly dependent** — partial pooling cannot separate three collinear channels with only 3 geo random effects per channel.

---

## 5. Controlled experiment matrix (same panel only)

| variant_id | changed_factor | rhat_max | divergences | convergence_status |
|------------|----------------|----------|-------------|-------------------|
| H5I-REF-H5G-FAST | reference | 2.09 | — | failed |
| H5I-REF-H5H-EXTENDED | reference | 1.69 | 74 | failed |
| H5I-BASELINE-FAST-REPLAY | baseline | 2.37 | 9 | failed |
| H5I-SCALING-ZSCORE-PRESCALE | scaling | **1.46** | **0** | failed |
| H5I-SAMPLER-TARGET-099 | sampler | 2.51 | 0 | failed |
| H5I-PRIOR-TIGHT-TAU | prior tau | 2.50 | 4 | failed |
| H5I-PRIOR-TIGHT-TAU-MU | prior tau+mu | 1.98 | 2 | failed |
| H5I-SINGLE-CHANNEL-SEARCH | 1 channel | **1.41** | 6 | failed |

**Best probes:** scaling prescale (no divergences, R-hat 1.46) and single-channel (R-hat 1.41). Neither meets `rhat_max <= 1.05` and `divergence_count == 0` together for promotion.

---

## 6. Minimal changes to try next (ordered)

1. **Collinearity-aware shadow config** — drop to one channel or PCA/orthogonal composite for research shadow only (extend H5I single-channel with extended MCMC once).
2. **Scaling prescale + extended MCMC** — combine `media_prescale`/`outcome_prescale` with H5c sampler profile; verify R-hat and ESS jointly.
3. **Non-centered parameterization** for `z_beta` / tau (code change in sandbox model — research only).
4. **Pooled media coefficients** (no geo×channel offsets) ablation to confirm geo hierarchy is the stressor.
5. **Do not** promote Bayes or batch panels until `converged_diagnostic_only` on pilot.

---

## 7. Production boundary

Production Bayes **remains blocked**. These diagnostics and experiments are report-only; `evidence_promotion_allowed=false` for all matrix rows.

---

## 8. How to reproduce

```bash
# Static diagnostics + experiment matrix (PyMC required for fits)
poetry run python -m mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics

poetry run pytest tests/research/test_bayes_h5_convergence_diagnostics.py -m "not slow"
```
