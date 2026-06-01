# INV-H5C — Bayes-H5 Extended MCMC Confirmation

**Investigation ID:** INV-H5C  
**Status:** **Complete (research lane)**  
**Date:** 2026-06-01  
**Prerequisites:** Bayes-H5b (`a768626`) — diagnostic polish + fast repeated pilot  
**Artifacts:**
- [BAYES_H5B_REPEATED_PILOT_20260601.json](../05_validation/archives/BAYES_H5B_REPEATED_PILOT_20260601.json)
- [BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json](../05_validation/archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json)

---

## 1. H5b summary (baseline for confirmation)

| Finding | H5b fast MCMC (3 seeds) |
|---------|-------------------------|
| SATURATION-ALIGNED | `beta_gc_mae` mean **0.114** vs H4c **0.171** |
| ADSTOCK-ALIGNED | mean **0.264** vs H4c **0.279** |
| Mismatch worlds | `h5:transform_mismatch` **100%** |
| Weak-ID | collinearity + weak-signal tags **100%** |
| False mismatch noise | `unexpected_transform_mismatch` **0%** |

---

## 2. H5c extended design

| Field | Value |
|-------|--------|
| **Runner** | `run_h5_repeated_pilot(seeds=[4400,4401,4402], extended_mcmc=True)` |
| **Sampler** | draws=600, tune=600, chains=4, target_accept=0.95 |
| **Worlds** | All 7 `WORLD-BAYES-H5-*` |
| **Runs** | 21 (7 × 3 seeds) |
| **Reference** | H5b artifact for `comparison_to_h5b_fast_pilot` |

---

## 3. Results table (aggregate `beta_gc_mae` mean)

| World | H5c extended | H5b fast | Δ vs H5b | H4c baseline | Improved vs H4c |
|-------|--------------|----------|----------|--------------|-----------------|
| ADSTOCK-ALIGNED | 0.263 | 0.264 | −0.001 | 0.279 | Yes |
| SATURATION-ALIGNED | 0.114 | 0.114 | −0.0003 | 0.171 | Yes |
| ADSTOCK-MISMATCH | 0.278 | 0.278 | −0.0002 | 0.279 | Yes |
| SATURATION-MISMATCH | 0.100 | 0.100 | −0.0002 | 0.171 | Yes |
| CORRELATED-CHANNELS | 0.267 | 0.268 | −0.001 | 0.268 | Yes |
| WEAK-SIGNAL | 0.280 | 0.283 | −0.003 | 0.284 | Yes |
| SPARSE-RECOVERY | 0.269 | 0.269 | −0.0004 | 0.269 | Yes |

**Diagnostic rates (H5c):** mismatch worlds 100% transform-mismatch warnings; unexpected-mismatch 0%; weak-ID/collinearity 100% on probe worlds.

---

## 4. Comparison to H5b

| Question | H5c answer |
|----------|------------|
| Saturation-aligned improvement hold? | **Yes** — mean 0.114, Δ vs H5b &lt; 0.001 |
| Adstock-aligned improvement hold? | **Yes** — mean 0.263, Δ vs H5b &lt; 0.002 |
| Mismatch warnings clean? | **Yes** — 100% on both mismatch worlds |
| Weak-ID warnings clean? | **Yes** — unchanged behavior |
| Sparse recovery report-only? | **Yes** — no promotion flags; MAE ≈ H4c |
| Material change under extended MCMC? | **No** — all \|Δ vs H5b\| &lt; 0.05 |

---

## 5. Comparison to H4c

H5c **confirms** H5b: transform-aligned H5 spec continues to outperform H4c MVP mismatch baselines on adstock/saturation probe worlds. Extended sampling did not reverse any H4c comparison direction.

---

## 6. Accepted / rejected conclusions

### Accepted (research evidence only)

1. **H5 `bayes_h5_sandbox_spec_v1` transform registry** is accepted as **stable research evidence** for:
   - Saturation-aligned recovery improvement vs H4c.
   - Modest adstock-aligned improvement vs H4c.
   - Reliable mismatch / weak-ID diagnostic separation.

2. H5b diagnostic polish (`transforms_aligned` for linear/correlated/weak-signal) is **confirmed** under extended MCMC.

### Rejected / not authorized

- Production Bayesian MMM promotion.
- Optimizer / DecisionSurface / recommendation emission.
- INV-071 hard gates or `approved_for_prod`.

---

## 7. Remaining limitations

- Toy panels only; not national-scale calibration.
- Some NUTS divergences on extended runs (monitor; not used as hard fail).
- TrustReport field wiring and real-panel pilots remain future work.
- Ridge production path unchanged.

---

## 8. Recommended next step

| Option | Notes |
|--------|--------|
| **TrustReport wiring** | Map `h5_transform_diagnostics` into diagnostic TrustReport stub |
| **Real-panel shadow** | Separate authorization; not part of H5c |
| **Production** | **Remain blocked** |

---

## 9. Production impact

**None.** `hard_gate=false`, `approved_for_prod=false`, `prod_decisioning_allowed=false`.
