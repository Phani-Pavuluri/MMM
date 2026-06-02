# INV-H6F: Ridge vs H5 synthetic benchmark matrix

**Lane:** Bayes-H6f (research only)  
**Date:** 2026-06-01  
**Prerequisite:** H5r (`c77e68a`), H6a–H6e synthetic lane

## Purpose

Produce a single governed artifact summarizing **Ridge (production baseline path)** and optional **Bayes-H5 (research)** behavior across all H6 pilot synthetic worlds. Separates:

- model issue
- transform-specification issue
- control-omission issue
- identification issue

## Worlds tested

| World ID | Vertical | Control variant |
|----------|----------|-----------------|
| `WORLD-H6-PILOT-RETAIL-FULL-CONTROLS` | retail | full_controls |
| `WORLD-H6-PILOT-RETAIL-OMITTED-CONTROLS` | retail | omitted_controls |
| `WORLD-H6-PILOT-RETAIL-MEDIA-CORRELATED-CONTROLS` | retail | media_correlated_controls |
| `WORLD-H6-PILOT-CPG-FULL-CONTROLS` | cpg | full_controls |
| `WORLD-H6-PILOT-AUTO-OMITTED-CONTROLS` | auto | omitted_controls |

**Scale:** pilot (20 geos × 52 weeks × 7 channels).  
**H5 optional runs:** retail full + retail media-correlated (slow; not required for CI).

## Ridge findings (pilot matrix)

- Ridge runs in `RunEnvironment.RESEARCH` with vertical controls in schema and Ridge BO transform tuning (adstock/Hill).
- Metrics per world: RMSE, WMAPE, geo-fold RMSE stability, coef/lift sign recovery vs known μ/β, collinearity max |ρ|, sparse-channel diagnostics.
- **Omitted-control worlds** emit forbidden claims; Ridge coef recovery typically degrades vs full-controls baseline when required controls are absent from the panel.
- **Media-correlated controls** stress false attribution risk to national TV when a required control co-moves with `tv` spend.
- Ridge does **not** emit optimizer, DecisionSurface, or recommendations.

## H5 findings (when run)

- H5 sandbox fits raw semi_log media; generative truth uses adstock+saturation → **transform mismatch warnings expected**.
- `evidence_promotion_allowed` remains **false** on all H6f rows.
- Convergence (divergences, rhat_max) and β/μ recovery metrics are diagnostic only — not promotion gates.
- H5q/H5r lesson applies: sparse tail channels can fail under keep-all; governed `drop_sparse_channels` is the real-panel remedy (see recommender update).

## Control omission findings

- `omitted_controls` variants drop **required** vertical controls from the panel while truth still includes their effects.
- Forbidden claims:
  - `do_not_attribute_omitted_control_lift_to_media`
  - `do_not_publish_incrementality_without_required_controls`
- **Vertical controls are necessary** for credible MMM evidence on production-shaped panels.

## Transform mismatch findings

| Layer | Transform family |
|-------|------------------|
| H6 generative truth | adstock + Hill |
| Ridge H6 benchmark | Ridge BO tuned adstock/Hill (closer to truth) |
| H5 sandbox | raw semi_log on standardized media |

H6f quantifies **transform-specification damage** separately from model/pooling damage. Ridge should recover better than H5 on coef/μ metrics when transforms align; remaining gap indicates model/pooling/identification limits.

## Sparse / collinearity findings

- Digital collinearity block (`display`, `ctv`) yields elevated max |ρ| on pilot panels.
- Sparse channels (`radio`, `local_flyer`) show high `near_zero_share` in tail geos — aligns with H5q keep-all failure and H5r sparse-drop remedy.

## Synthetic-to-real comparison (H5m / H5o / H5q / H5r)

| Real shadow | Synthetic H6 analogue |
|-------------|----------------------|
| H5m collinear drop tv | Collinearity block + high ρ warnings |
| H5o keep-all converged | Retail/CPG full-controls pilot |
| H5q keep-all 14 div | Sparse tail + identification stress |
| H5r drop_sparse radio converged | Recommender `drop_sparse_channels` over keep_all |

## What this proves

1. A **matrix lane** exists to compare Ridge and H5 on identical production-shaped known-truth worlds.
2. Control omission and media–control confounding produce governed **forbidden claims** and measurable Ridge degradation.
3. Transform mismatch is **recorded**, not silent — supports separating specification vs model issues.

## What it does not prove

- Real client panel incrementality or budget optimality
- Production Bayes readiness
- That synthetic success overrides H5q-style real-panel convergence failures

## Production boundary

All H6f artifacts: `approved_for_prod=false`, `prod_decisioning_allowed=false`, no optimizer/DecisionSurface/recommendations. Ridge remains production baseline; Bayes-H5 research-only.

## Artifacts

- `docs/05_validation/archives/BAYES_H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX_20260601.json`
- `docs/05_validation/archives/BAYES_H6F_CONTROL_CONFOUNDING_SUMMARY_20260601.json`
- Recommender guidance: `docs/05_validation/bayes_h6_synthetic_lane_adr.md` §H6f / H5r

## Recommended next step

**H7 Ridge production diagnostic hardening** — see [ridge production diagnostics contract](../05_validation/ridge_production_diagnostics_contract.md) and `mmm/diagnostics/ridge_diagnostics.py`.

**Harden Ridge production diagnostics first** (in progress via H7) (transform alignment reporting, sparse-channel flags, control completeness checks) — Ridge remains the decision path while H5 shadow matures.
