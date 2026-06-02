# INV-H5J — Bayes-H5 Collinearity-Aware Shadow Config and Geometry Ablations

**Investigation ID:** INV-H5J  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5i convergence diagnostics (`c309592`)  
**Artifact:** [BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json](../05_validation/archives/BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json)  
**Code:** `h5_real_panel_preprocessing.py`, `h5j_geometry_ablation_runner.py`, shadow runner `channel_policy` integration

---

## 1. Purpose

Test **explicit, governed** collinearity-aware shadow configurations on the **same** sample panel (`examples/sample_panel.csv`) before any additional real-panel runs. No silent channel dropping or merging.

---

## 2. Collinearity facts (static)

| Metric | Value |
|--------|--------|
| max \|ρ\| off-diagonal | **0.989** (social–tv) |
| Collinear group @ 0.95 | search, social, tv (one connected group) |
| Geos | 3 × 41 weeks |

---

## 3. Does explicit collinearity handling improve convergence?

**Partially — not enough for evidence promotion.**

| variant_id | Channel policy | rhat_max | divergences | convergence_status |
|------------|----------------|----------|-------------|-------------------|
| H5J-A-BASELINE-REPLAY | all channels, fast | 2.37 | 9 | failed |
| H5J-B-PRESCALE-EXTENDED | all + prescale | 1.07 | 21 | failed |
| H5J-C-SINGLE-SEARCH-PRESCALE-EXTENDED | search only | **1.01** | 38 | failed |
| H5J-D-DROP-COLLINEAR-PRESCALE-EXTENDED | drop redundant (tv) | **1.02** | **4** | **weak_convergence** |
| H5J-E-COMPOSITE-SOCIAL-TV-PRESCALE-EXTENDED | PC1(social,tv)+search | 1.04 | 7 | failed |
| H5J-F-POOLED-CHANNEL-EFFECTS | — | — | — | not implemented |

**Best probe:** `H5J-D` (drop collinear + prescale + extended) — first **weak_convergence** on pilot.  
**None** achieved `converged_diagnostic_only` (requires rhat ≤ 1.05 **and** zero divergences).

---

## 4. Does prescale + extended MCMC clear divergences?

**No** when all three channels remain (H5J-B: rhat 1.07 but **21** divergences).  
Prescale helps R-hat vs baseline but **does not** jointly clear divergences + R-hat on full collinear set.

---

## 5. Does single-channel or dropped-channel clear R-hat?

- **Single search (H5J-C):** R-hat **1.01** (near threshold) but **38** divergences → still **failed**.
- **Drop collinear (H5J-D):** keeps search + social, drops **tv** (highest redundancy in group). R-hat **1.02**, divergences **4** → **weak_convergence** only.

Dropping one collinear channel helps more than keeping all three with prescale alone.

---

## 6. Does composite media help?

**H5J-E** (PC1 of social+tv, keep search): rhat 1.04, 7 divergences → **failed**.  
Slightly better than baseline but worse than drop-collinear on divergences.

---

## 7. Is hierarchy still problematic after collinearity reduction?

**Yes, but reduced.** H5J-D worst parameters remain `tau_channel`, `z_beta`, `mu_channel` — hierarchical media blocks still stress NUTS, yet status improves to weak_convergence.  
Full elimination of hierarchy stress likely needs **non-centered parameterization** or **pooled-channel ablation** (F not implemented).

---

## 8. Minimum shadow-run eligibility before more real panels

A panel may proceed to additional real-panel shadow runs only when a pilot configuration satisfies **all**:

1. Explicit `channel_policy` documented in transform config (no silent edits).
2. `convergence_status` = **`converged_diagnostic_only`** (rhat_max ≤ 1.05, divergence_count = 0), or documented executive acceptance of `weak_convergence` for research-only archives.
3. `evidence_promotion_allowed` = true (derived from above).
4. Production flags remain false; no optimizer / DecisionSurface / prod TrustReport.

**Current pilot:** **not eligible** — best is weak_convergence (H5J-D).

---

## 9. channel_policy modes (explicit)

| mode | Behavior |
|------|----------|
| `keep_all_channels` | No channel change |
| `single_channel` | Requires `channel` |
| `drop_collinear_channels` | Requires `max_abs_corr_threshold`; records dropped channels + reason; fail closed if zero media left |
| `composite_media_channel` | Requires `source_channels`, `method`, `output_channel`; ablation/diagnostic only |
| `pooled_channel_effects` | **Not implemented** (variant F documented) |

---

## 10. Production boundary

Production Bayes **remains blocked**. Ablation outputs are report-only; `h5:evidence:blocked` unless converged_diagnostic_only.

---

## Reproduce

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5j_geometry_ablation_runner
poetry run pytest tests/research/test_bayes_h5_real_panel_preprocessing.py tests/research/test_bayes_h5j_geometry_ablation_runner.py -m "not slow"
```
