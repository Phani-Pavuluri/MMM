# INV-H5R — Bayes-H5 Sparse-Channel Remedy Replay

**Investigation ID:** INV-H5R  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Prerequisites:** H5q @ `597d989`  
**Panel:** `examples_mmm_triangulation_geo_panel_v1` (same panel as H5q — no new panel)  
**Policy:** [h5r_examples_mmm_triangulation_geo_panel_v1_sparse_radio_policy.json](h5r_examples_mmm_triangulation_geo_panel_v1_sparse_radio_policy.json)  
**Comparison:** [BAYES_H5R_REMEDY_COMPARISON_…](../05_validation/archives/BAYES_H5R_REMEDY_COMPARISON_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json)

## Purpose

Test whether a **governed sparse-channel remedy** (explicit drop of near-zero `radio`) restores MCMC convergence on the H5q triangulation panel without changing hierarchy-faithful geometry.

## Did dropping sparse radio restore convergence?

**Yes.**

| Run | channel_policy | convergence_status | rhat_max | divergences | evidence_promotion_allowed |
|-----|----------------|-------------------|----------|-------------|---------------------------|
| **H5q** | keep_all (4 channels) | `failed_convergence` | 1.01 | **14** | false |
| **H5r** | drop_sparse_channels (drop radio) | `converged_diagnostic_only` | 1.01 | **0** | true (research only) |

Shadow artifact: [BAYES_H5R_SHADOW_RUN_…](../05_validation/archives/BAYES_H5R_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_SPARSE_RADIO_20260601.json)

## Did R-hat / divergences improve relative to H5q?

- **Divergences:** 14 → **0**  
- **R-hat max:** 1.01 → 1.01 (both at diagnostic threshold)  
- **Convergence class:** failed → `converged_diagnostic_only`

The remedy directly addressed the H5q failure mode (weak ID from ~99% near-zero radio under 8×4 hierarchy).

## Usable research shadow evidence (AUDIT-H5P U1–U9)?

**Yes for H5r** — meets converged diagnostic criteria on this panel with explicit sparse-drop policy, lineage, forbidden claims, and calibration stub carried as diagnostic-only metadata.

**H5q** remains a valid **workflow** validation (honest failure) but not converged evidence.

## Interpretation changes after dropping radio

- **Forbidden:** no separate **radio** effect claim.  
- Retained channels: search, social, display — partial-pooling geo×channel coefficients only.  
- Moderate social–display correlation (~0.94) still applies; no clean isolated channel claims without external calibration.  
- Calibration stub references retained channels only (`radio` excluded from experiment ref list in policy).

## Are calibration stubs still only diagnostic?

**Yes.**

- `likelihood_integrated=false` on policy and artifact.  
- GeoX/CLS stub is a **triangulation cross-check** only — not production calibration or likelihood integration.

## Should recommender prefer `drop_sparse_channels` for near-zero media?

**Recommendation for H5n+ enhancement:** when `near_zero_share` ≥ threshold (e.g. 0.99), recommender should:

1. Add forbidden separate-coefficient claim (already does).  
2. Offer **`drop_sparse_channels`** as primary or co-primary remedy when keep-all + sparse channel detected — not only `single_channel_diagnostic`.  
3. Record `sparse_drop_reason` explicitly — distinct from collinearity drop.

H5q listed single-channel diagnostic as alternative but recommended keep-all; H5r shows **governed drop** is the effective remedy for this panel.

## What remains blocked?

- Production Bayes, optimizer, DecisionSurface, budget recommendations, prod TrustReport, Ridge replacement.  
- No new panel batching.  
- `evidence_promotion_allowed=true` on H5r is **research eligibility only**.

## Production boundary

**Blocked.** All production flags false on policy, shadow artifact, and comparison JSON.

## Commands

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --policy-path docs/06_investigations/h5r_examples_mmm_triangulation_geo_panel_v1_sparse_radio_policy.json \
  --output-path docs/05_validation/archives/BAYES_H5R_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_SPARSE_RADIO_20260601.json
```

## Conclusion

H5r confirms the H5q failure was remediable via **explicit sparse-channel drop** on the same panel. The governance loop correctly allowed H5q to fail honestly and H5r to test the named remedy before any fourth panel.
