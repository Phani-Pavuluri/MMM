# INV-H5K — Bayes-H5 Non-Centered / Geometry Stabilization

**Investigation ID:** INV-H5K  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Source:** H5j @ commit `655853f`  
**Artifact:** [BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json](../05_validation/archives/BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json)  
**Panel:** `examples/sample_panel.csv` (`examples_mmm_sample_panel_v1`) only — no additional real panels.

## Context

H5j best variant (`H5J-D`): `drop_collinear_channels` @ 0.95 (drops **tv**, keeps **search** + **social**), prescaled log outcome + z-scored media, extended MCMC → **rhat_max ≈ 1.02**, **4 divergences**, **`weak_convergence`** — not `converged_diagnostic_only`.

H5k adds **explicit** sandbox geometry configuration (`parameterization`, `likelihood_scale_policy`, `hierarchy_policy`) and runs stabilization variants on the same governed channel baseline.

## Variants (summary)

| Variant | Parameterization | Hierarchy | Channel policy | Notes |
|---------|------------------|-----------|----------------|-------|
| H5K-A | centered | full geo×channel | drop collinear | Centered beta ablation (funnel-prone) |
| H5K-B | non_centered | full | drop collinear | Explicit NC label + H5j-D replay |
| H5K-C | non_centered | full | drop collinear | `target_accept=0.99` |
| H5K-D | non_centered | full | single search | Localization probe |
| H5K-E | non_centered | pooled channel | drop collinear | No geo×channel partial pooling |
| H5K-F | non_centered | fixed τ=0.2 | drop collinear | Removes τ funnel |

| Variant | rhat_max | divergences | status |
|---------|----------|-------------|--------|
| H5K-A centered replay | 1.02 | 3 | weak_convergence |
| H5K-B explicit non-centered | 1.02 | 4 | weak_convergence |
| H5K-C target_accept=0.99 | 1.06 | **0** | weak_convergence (R-hat) |
| H5K-D single search | 1.01 | 38 | failed_convergence |
| H5K-E pooled channel | **1.00** | **0** | **converged_diagnostic_only** |
| H5K-F fixed τ=0.2 | **1.00** | **0** | **converged_diagnostic_only** |

`any_variant_converged_diagnostic_only=true` in the artifact.

## Questions

### Did non-centered parameterization reduce divergences?

**No meaningful improvement** on the H5j-D channel baseline. Legacy H5 already used `z_beta`; H5K-B (explicit label) matches H5j-D (4 divergences). H5K-A **centered** `beta` actually had **fewer** divergences (3) but the same weak R-hat — not a promotion win.

### Did it preserve R-hat?

**Yes** for H5K-A/B (~1.02). H5K-C cleared divergences but **R-hat regressed to 1.06** → still not promotable.

### Did any variant reach `converged_diagnostic_only`?

**Yes — two ablations**, not the full partial-pooling spec:

- **H5K-E** (`pooled_channel_effects_ablation`): shared `beta_channel` across geos — diagnostics pass; **geo-specific media hierarchy removed**.
- **H5K-F** (`fixed_tau_ablation`, τ=0.2): removes τ funnel — diagnostics pass; **τ no longer learned**.

**Production Bayes remains blocked** (`approved_for_prod=false`). These are research eligibility signals on the pilot panel only, not production promotion.

### If no (on full hierarchy), what remains the blocker?

For **full geo×channel partial pooling** under drop-collinear + prescale: **τ / μ_channel / z_beta funnel** and tight log-scale `sigma`. Collinearity handling alone is insufficient.

### Is pooled / fixed-τ ablation needed?

**Yes — decisive on this panel.** Pooled (E) and fixed-τ (F) are the first variants to clear the H5h evidence bar. Next step for a **hierarchy-faithful** pass: non-centered τ / σ_floor while keeping partial pooling, or pilot re-run documenting E/F as diagnostic-only ablations.

### Default H5 sandbox parameterization?

**Remain experimental.** Implicit legacy default (non-centered + full hierarchy) is unchanged unless `h5_geometry_config` is set on the H5 gated path. Do **not** silently default to pooled or fixed-τ.

## Production

- `hard_gate=false`
- `production_promotion=false`
- `approved_for_prod=false`
- `prod_decisioning_allowed=false`
- No optimizer, DecisionSurface, or recommendations
