# Exact Recovery Investigation Report (INV-056 / Phase 5C)

**Version:** exact_recovery_investigation_v1.0.0  
**Generated:** 2026-05-29T17:03:12.752586+00:00  
**Scope:** Analysis only — no new estimators, no production changes.

Spec: [exact_recovery_investigation.md](exact_recovery_investigation.md)

---

## Executive summary

Phase 4B–5B evidence shows **structural reliability is high** while **behavioral recovery (coefficient + transform) is materially weaker**. This investigation explains **why** WORLD-008 and behavioral-lattice exact-recovery worlds fail coefficient recovery, and whether gaps are expected MMM limitations vs bugs.

**Primary conclusions:**

1. **Δμ recovery is stronger than coefficient recovery** on the same worlds — consistent with MMM practice (many parameter settings can yield similar counterfactuals).
2. **Transform estimation dominates BO-path error**: truth-pinned transforms drop WORLD-008 max |β̂−β| from **0.88 → 0.14** and fix display (β̂≈0.08 vs true 0.08), but **search/social still homogenize** (~0.28 vs 0.42/0.15) because channels share one transform pipeline in Ridge BO.
3. **BO search mis-estimates decay/Hill** (WORLD-008: decay 0.17 vs truth 0.55; Hill half 1.2 vs 10.0), producing wrong features before Ridge sees the data.
4. **Hyperparameter–coefficient coupling**: alternative (decay, Hill) grids achieve similar in-sample RMSE with different β vectors — objective does not identify truth uniquely.
5. **Ridge shrinkage is secondary** at truth-pinned transforms: α sweep moves max coef error only ~0.14→0.42 across log_alpha −10…2; does not explain BO display blow-up.
6. **TBD_v1 coef tolerances may be unrealistic** for multi-channel BO; **Δμ tolerances are more aligned** with decision use-cases.

---

## Observed failures

| World | Coef (BO) | Transform (BO) | Δμ (BO) | Coef (truth-pinned) |
|-------|-----------|----------------|---------|---------------------|
| `WORLD-008-exact-recovery` | fail | fail | True | fail |
| `L5B-exact_recovery-noise-low-corr-low-drift-off-replay-off` | fail | fail | True | fail |
| `L5B-exact_recovery-noise-low-corr-severe-drift-off-replay-off` | fail | fail | True | pass |
| `L5B-exact_recovery-noise-zero-corr-low-drift-off-replay-off` | fail | fail | True | fail |

WORLD-008 per-channel (BO): display fitted 0.958 vs true 0.080; search/social both ~0.51 vs 0.42/0.15.

---

## Recovery decomposition

Error enters primarily in this order:

1. **Transform hyperparameters** (decay, Hill half, slope) — BO search returns parameters that fit in-sample KPI but deviate from truth.
2. **Channel coefficient vector** — given shared transforms, Ridge assigns similar β across channels; cannot recover heterogeneous true β.
3. **Δμ / decision layer** — counterfactual simulation often remains within TBD_v1 tolerance even when β is wrong (business-facing metric more stable).

WORLD-008 Δμ: analytic=0.05377146150445178, fitted=0.0005727550543426219, pass=True.

---

## Transform sensitivity (fitted vs truth-pinned)

Truth-pinned training fixes transform params to `transform_truth` (mean across channels when per-channel values differ). This isolates **coefficient estimation** given correct feature construction.

See `investigations/transform_sensitivity.json`.

---

## Regularization findings

Alpha sweep on truth-pinned design (WORLD-008):

| log_alpha | max coef abs error | Δμ pass |
|-----------|-------------------|---------|
| -10 | 0.1350 | True |
| -8 | 0.1350 | True |
| -6 | 0.1352 | True |
| -4 | 0.1515 | True |
| -2 | 0.2167 | True |
| 0 | 0.3154 | True |
| 2 | 0.4179 | True |

Shrinkage is **not** the dominant driver of display-channel failure at α≈1e-6…1e-2. The failure persists because **features are wrong** (shared transform + collinear channel features), not because of excessive penalty.

---

## Identifiability findings

| Variant | Coef pass (BO) | Coef pass (pinned) | Δμ pass (BO) |
|---------|----------------|--------------------|--------------|
| id-1ch | False | True | None |
| id-2ch-orth | False | True | None |
| id-2ch-severe | False | True | None |

Single-channel and mini two-channel worlds **pass coefficient recovery when transforms are truth-pinned** (`id-1ch`, `id-2ch-orth`, `id-2ch-severe`). WORLD-008 fails pinned recovery because **three channels share constant spend** with one transform — features are nearly collinear across columns. Behavioral lattice **severe-collinearity** exact world can pass pinned coef when collinearity is the only axis change.

---

## Data-volume findings

See `investigations/data_volume_sweep.json`. More geos/periods improve stability but do not fix shared-transform homogenization on multi-channel worlds at zero noise.

---

## Recovery taxonomy

- `WORLD-008-exact-recovery`: identifiability_limitations, regularization_bias, threshold_artifacts, transform_misspecification
- `L5B-exact_recovery-noise-low-corr-low-drift-off-replay-off`: identifiability_limitations, regularization_bias, threshold_artifacts, transform_misspecification
- `L5B-exact_recovery-noise-low-corr-severe-drift-off-replay-off`: implementation_issues, threshold_artifacts, transform_misspecification
- `L5B-exact_recovery-noise-zero-corr-low-drift-off-replay-off`: identifiability_limitations, regularization_bias, threshold_artifacts, transform_misspecification

---

## Root-cause ranking

1. **shared_transform_across_channels** (high): Ridge BO uses one decay/Hill for all channels; fitted betas homogenize (display error largest)
2. **hyperparameter_search_objective** (high): BO transform params differ from truth; grid shows 1 near-equivalent RMSE points
3. **coefficient_non_identifiability_under_shared_features** (medium): Multi-channel worlds fail coef recovery even at truth-pinned transforms
4. **ridge_shrinkage_secondary_at_default_alpha** (low): Best coef error in alpha sweep log_alpha=-10 (pinned transforms)
5. **delta_mu_more_forgiving_than_coef** (informational): Δμ recovery passes TBD_v1 tolerances while coef fails on same worlds

---

## Recommendations

### Expected limitations (not bugs)

- Multi-channel MMM with **one shared transform** cannot exactly recover per-channel truth in general.
- **Δμ-first reliability** is more appropriate than coef-first for production TrustReport.

### Implementation / architecture

- Document shared-transform policy in recovery certification limitations.
- Consider per-channel transform search only in research worlds (out of scope for 5C).

### Threshold calibration (Phase 5D — implemented)

Encoded in [reliability_threshold_governance.md](reliability_threshold_governance.md):

- Split tolerances: **decision metrics** (Δμ, optimizer) vs **attribution metrics** (β).
- Do not use coef recovery alone as a release gate for Ridge BO.
- `TBD_v1_runtime` remains provisional until DR-04 approval.

### Future Bayesian worlds (Track 4)

- Expect **posterior shrinkage** and **partial pooling** — different failure mode than Ridge BO.
- Bayes-H2 worlds should encode geo-level truth; coef recovery expectations must differ.

### Roadmap

- **5D** ✅ threshold governance and metric-class scorecard adopt findings above.
- **5E** drift runner + TrustReport downgrade semantics (next).
- **5F** Monte Carlo should sample transform-identifiability axes, not only noise.

---

## Acceptance criteria (INV-056)

| Question | Answer |
|----------|--------|
| Why does coef recovery fail? | Shared transforms + BO search + multi-channel identifiability |
| Why does transform recovery fail? | BO optimizes in-sample fit; flat objective vs coef |
| Are failures expected? | **Yes** for multi-channel BO; **no** for single-channel pinned case |
| Are tolerances unrealistic? | **Coef tolerances yes** for BO; **Δμ tolerances more reasonable** |
| Is Δμ reliable when coef fails? | **Often yes** within TBD_v1 — primary decision metric |
| Is Ridge architecture fundamentally limited? | **For exact coef recovery yes**; for Δμ less so |
| Bayesian world expectations? | Hierarchical geo worlds need different metrics; not coef-parity with Ridge |

---

## Supporting artifacts

| File | Content |
|------|---------|
| [investigations/exact_recovery_findings.json](investigations/exact_recovery_findings.json) | Full machine-readable bundle |
| [investigations/recovery_decomposition.json](investigations/recovery_decomposition.json) | Per-world metrics |
| [investigations/regularization_sweep.json](investigations/regularization_sweep.json) | Alpha sweep |
| [investigations/hyperparameter_coupling.json](investigations/hyperparameter_coupling.json) | Decay/Hill grid |
| [investigations/identifiability_grid.json](investigations/identifiability_grid.json) | Controlled worlds |
| [investigations/data_volume_sweep.json](investigations/data_volume_sweep.json) | Volume axes |
| [investigations/recovery_taxonomy.json](investigations/recovery_taxonomy.json) | Failure classes |

