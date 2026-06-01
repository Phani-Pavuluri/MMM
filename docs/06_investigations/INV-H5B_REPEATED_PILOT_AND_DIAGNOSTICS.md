# INV-H5B — Bayes-H5 Repeated Pilot and Diagnostic Polish

**Investigation ID:** INV-H5B  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5a (`d8d0413` implementation, `8fcedf0` full fast pilot)  
**Artifact:** [BAYES_H5B_REPEATED_PILOT_20260601.json](../05_validation/archives/BAYES_H5B_REPEATED_PILOT_20260601.json)  
**ADR:** [bayes_h5_model_spec_improvement_adr.md](../05_validation/bayes_h5_model_spec_improvement_adr.md)

---

## 1. H5a issue found

H5a fast pilot showed correct recovery patterns but **noisy diagnostics**:

- `transforms_aligned()` did not treat `linear`, `correlated`, or `weak_signal` generative kinds as aligned with `identity` fit.
- Fit packaging set `transform_mismatch_detected=true` on sparse/correlated/weak worlds.
- `compute_h5_diagnostic_warnings` emitted `h5:unexpected_transform_mismatch` on non-transform-probe worlds.
- Duplicate `h4c:transform_mismatch` warnings appeared on H5 intentional-mismatch worlds.

**Classification:** Diagnostic / reporting bug — **not** a model-spec defect.

---

## 2. Diagnostic fix (H5b)

| Change | Location |
|--------|----------|
| `GENERATIVE_KINDS_IDENTITY_FIT` + expanded `transforms_aligned()` | `h5_transforms.py` |
| `compute_transform_mismatch_detected()` | `h5_transforms.py`, `model.py` |
| Skip `h4c:transform_mismatch` when `h5_classification` set | `recovery_runner.py` |
| Weak-signal role tag; suppress false unexpected mismatch on weak-ID | `recovery_runner.py` |
| Regression tests | `tests/research/test_bayes_h5_diagnostics.py` |

**Post-fix behavior:**

- Aligned adstock/saturation worlds: **no** transform-mismatch warnings.
- Intentional mismatch worlds: **`h5:transform_mismatch`** at **100%** rate across seeds.
- Correlated: **collinearity** warnings only (no transform-mismatch).
- Weak-signal: **`h5:weak_identification`** at **100%** rate.
- Sparse-recovery: **no** transform-mismatch noise; report-only recovery metrics.

---

## 3. Repeated pilot design

| Field | Value |
|-------|--------|
| **Runner** | `run_h5_repeated_pilot(seeds=[4400, 4401, 4402], fast_mcmc=True)` |
| **Sampler** | PyMC NUTS; 200 tune / 200 draw / 2 chains / `target_accept=0.92` |
| **Worlds** | All 7 `WORLD-BAYES-H5-*` validation worlds |
| **Runs** | 21 (7 worlds × 3 seeds) |
| **H4c baselines** | Loaded from `BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json` |

---

## 4. Per-world results (aggregate `beta_gc_mae` mean)

| World | H5 mean | H4c baseline | Δ vs H4c | Mismatch warn rate | Other diagnostics |
|-------|---------|--------------|----------|-------------------|-------------------|
| ADSTOCK-ALIGNED | 0.264 | 0.279 (adstocked) | **−0.015** | 0% | — |
| SATURATION-ALIGNED | 0.114 | 0.171 (saturation) | **−0.057** | 0% | — |
| ADSTOCK-MISMATCH | 0.278 | 0.279 | ≈0 | **100%** | intentional probe |
| SATURATION-MISMATCH | 0.100 | 0.171 | −0.071 | **100%** | still warns (by design) |
| CORRELATED-CHANNELS | 0.268 | 0.268 | ≈0 | 0% | collinearity 100% |
| WEAK-SIGNAL | 0.283 | 0.284 | ≈0 | 0% | weak-ID 100% |
| SPARSE-RECOVERY | 0.269 | 0.269 | ≈0 | 0% | report-only sparse metrics |

**Stability (aligned worlds):**

- ADSTOCK-ALIGNED: std ≈ 0.0012 across seeds; improved vs H4c on **all 3** seeds.
- SATURATION-ALIGNED: std ≈ 0.0005 across seeds; improved vs H4c on **all 3** seeds.

---

## 5. Comparison to H4c baselines

| Question | Answer |
|----------|--------|
| Saturation-aligned improvement stable? | **Yes** — large Δ (−0.057 MAE) with very low cross-seed variance. |
| Adstock-aligned improvement stable or noise? | **Stable modest gain** — Δ ≈ −0.015 on all seeds; std &lt; 0.002. |
| Mismatch worlds consistently warn? | **Yes** — 100% `h5:transform_mismatch` rate. |
| Weak-ID worlds consistently warn? | **Yes** — collinearity (correlated) and weak-signal tags at 100%. |
| Sparse-recovery acceptable? | **Yes** — MAE in line with H4c sparse-recovery; no false transform warnings. |
| Diagnostics noisy after fix? | **No** — `unexpected_transform_mismatch` rate **0%** on all worlds. |

---

## 6. Interpretation

1. **H5 transform registry achieves its primary goal** on transform-probe worlds: aligned fits beat H4c MVP mismatch baselines (especially saturation); intentional mismatch worlds still fail loudly.
2. **Adstock gain is real but small** on toy panels — sufficient to continue research, not sufficient for promotion.
3. **Weak-ID and sparse behavior unchanged in spirit** from H4c — still warn/restricted, not hard-fail.
4. **No implementation defects** blocking H5 spec; optional future work: extended MCMC, prior grid, TrustReport field wiring.

---

## 7. Recommended next step

| Step | Notes |
|------|--------|
| **H5c (optional)** | Extended MCMC repeated pilot if convergence diagnostics need tightening |
| **TrustReport mapping** | Wire `h5_transform_diagnostics` into diagnostic TrustReport stub |
| **Production** | **Remain blocked** — no DecisionSurface, optimizer, or INV-071 hard gates |

---

## 8. Production impact

**None.** `approved_for_prod=false`, `prod_decisioning_allowed=false`, `hard_gate=false`, Ridge production path unchanged.
