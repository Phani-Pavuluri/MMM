# INV-H5Q — Bayes-H5 Third Real-Panel Shadow Run

**Investigation ID:** INV-H5Q  
**Status:** complete (research lane) — workflow pass; convergence **failed** honestly  
**Date:** 2026-06-01  
**Prerequisites:** H5p @ `0b918f0`  
**Manifest:** [H5Q_THIRD_REAL_PANEL_SHADOW_RUN_MANIFEST.md](H5Q_THIRD_REAL_PANEL_SHADOW_RUN_MANIFEST.md)

## Why was this panel selected?

`examples_mmm_triangulation_geo_panel_v1` tests conditions **not covered** by prior panels:

| Prior | Condition |
|-------|-----------|
| **H5m** (sample) | High collinearity → governed **drop tv**; 3 geos; converged |
| **H5o** (benchmark) | Low collinearity → **keep all**; 4 geos; converged |

**H5q** adds: **8 geos**, **4 channels** (including sparse **radio**), **moderate** social–display correlation (~0.94, below 0.95 gate), and a **GeoX/CLS calibration stub** to exercise the triangulation bridge in recommender + policy metadata.

## What new diagnostic condition did it test?

1. **Scale:** more geos and channels than H5m/H5o.  
2. **Sparsity:** radio ~99% near-zero — forbidden separate coefficient claim; single-channel diagnostic alternative.  
3. **Moderate collinearity:** borderline \|ρ\| without crossing 0.95 keep-all block.  
4. **Calibration availability:** `calibration_evidence_available=true` → external-calibration allowed alternative on recommendation.

## What did the recommender diagnose?

Artifact: [BAYES_H5Q_SHADOW_POLICY_RECOMMENDATION_…](../05_validation/archives/BAYES_H5Q_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json)

| Field | Value |
|-------|--------|
| max \|ρ\| | 0.943 |
| high_collinearity | false |
| Radio sparsity | near_zero_share ≈ 0.99 |
| **status** | `recommended` (runnable — not `do_not_run`) |
| Primary | **keep_all_channels** + σ-floor hierarchy |

Alternatives: single-channel diagnostic on **radio**; external calibration (stub available).  
Blocked: ablation promotion.

## What policy did it recommend?

Frozen: [h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy.json](h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy.json)

| Field | Value |
|-------|--------|
| policy_id | `bayes_h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy_v1` |
| channel_policy | keep_all_channels (all four media columns) |
| geometry | NC hierarchy, learned τ, sigma_floor=0.05 |
| calibration | stub + geox_cls_comparison.available=true |
| sampler | extended MCMC 600/600/4 |

## Were channels dropped, kept, composited, or blocked?

- **Kept:** search, social, display, radio (explicit keep-all).  
- **Not dropped or composited.**  
- **Sparse radio:** separate coefficient claim blocked via forbidden-claims / sparsity rules; single-channel diagnostic listed as alternative only.

## What claims became forbidden?

- No production Bayes, optimizer, DecisionSurface, budget recommendations, Ridge replacement.  
- Channel `radio` near-zero variation — block separate coefficient claim.  
- Keep-all weak-ID — no channel-level budget/causal decision use.

## Did frozen policy validation pass?

**Yes.** `validate_shadow_policy` passed including calibration stub fields and production flags false.

## Did shadow replay converge?

**No** (recorded honestly):

| Metric | Value |
|--------|--------|
| convergence_status | `failed_convergence` |
| rhat_max | 1.01 |
| divergences | 14 |
| evidence_promotion_allowed | **false** |

Artifact: [BAYES_H5Q_SHADOW_RUN_…](../05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json)

PyMC warned overflow/divergences — likely weak identification from sparse radio + 8×4 hierarchy load. **Do not** treat as evidence-ready per [AUDIT-H5P](../audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md) criterion U4.

## How does it compare to H5m / H5o?

| | H5m | H5o | H5q |
|--|-----|-----|-----|
| Geos | 3 | 4 | **8** |
| Channels | 3 (drop tv) | 3 (keep all) | **4** (keep all) |
| Collinearity | high | low | **moderate** |
| Calibration | none | none | **stub** |
| Convergence | converged_diagnostic_only | converged_diagnostic_only | **failed_convergence** |

H5q proves the **workflow** generalizes (recommender → freeze → replay + calibration metadata). It does **not** add a third converged panel.

## Usable research shadow evidence (H5p)?

**Partial:**

- ✅ Workflow, lineage, policy match, forbidden claims, calibration stub on artifact.  
- ❌ Not usable as **converged** shadow evidence (failed convergence, 14 divergences).

## What should happen next?

1. **Do not batch** panels.  
2. Before H5r+: consider governed **drop radio** or single-channel diagnostic policy and re-replay; or reduce hierarchy load.  
3. Prefer next panel with **Ridge diagnostic** arm when wiring is available.  
4. Re-read [AUDIT-H5P](../audits/AUDIT-H5P_BAYES_H5_SHADOW_WORKFLOW_GATE.md) before authorizing another panel.

## Production boundary

**Blocked.** All production flags false. No optimizer / DecisionSurface / recommendations / prod TrustReport / Ridge replacement.

## Commands

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_policy_recommender \
  --panel-path examples/triangulation_geo_panel_v1.csv \
  --panel-id examples_mmm_triangulation_geo_panel_v1 \
  --dataset-snapshot-id mmm-examples-triangulation-geo-panel-frozen-2022-v1 \
  --panel-schema-path docs/06_investigations/h5q_triangulation_panel_schema.json \
  --calibration-evidence-available \
  --output-path docs/05_validation/archives/BAYES_H5Q_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json

poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --policy-path docs/06_investigations/h5q_examples_mmm_triangulation_geo_panel_v1_shadow_policy.json \
  --output-path docs/05_validation/archives/BAYES_H5Q_SHADOW_RUN_EXAMPLES_MMM_TRIANGULATION_GEO_PANEL_V1_20260601.json
```
