# INV-H5L — Bayes-H5 Hierarchy-Faithful Geometry Refinement

**Investigation ID:** INV-H5L  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Source:** H5k @ `454ce94`  
**Artifact:** [BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json](../05_validation/archives/BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json)  
**Panel:** `examples/sample_panel.csv` only.

## Purpose

H5k showed `converged_diagnostic_only` only under **ablation** specs (pooled channels, fixed τ). H5l tests **hierarchy-faithful** stabilization: learned geo×channel partial pooling and learned τ, with explicit `tau_parameterization`, `sigma_policy`, `beta_prior_policy`, and `hierarchy_strength_policy`.

## Channel baseline (H5j-D)

- `drop_collinear_channels` @ 0.95 → drops **tv**, keeps **search** + **social**
- Prescaled log outcome + z-scored media
- Extended MCMC (600/600/4)

## Results

| Variant | Faithful | rhat_max | div | Status | Promotable |
|---------|----------|----------|-----|--------|------------|
| H5L-A replay | yes | 1.03 | 3 | weak | no |
| **H5L-B sigma_floor** | **yes** | **1.01** | **0** | **converged_diagnostic_only** | **yes (research)** |
| H5L-C strong τ prior | yes | 1.01 | 4 | weak | no |
| H5L-D strong β prior | yes | 1.01 | 7 | failed | no |
| **H5L-E σ_floor + strong τ** | **yes** | **1.01** | **0** | **converged_diagnostic_only** | **yes (research)** |
| H5L-F log_tau | yes | 1.05 | 2 | weak | no |
| H5L-G NC log_tau | yes | 1.02 | 2 | weak | no |
| H5L-H weak τ + σ reg | yes | 1.03 | 8 | failed | no |
| H5L-I channel-scaled β | yes | 1.03 | 3 | weak | no |
| H5L-J fixed-τ benchmark | no | 1.01 | 0 | converged | **no (ablation)** |
| H5L-K pooled benchmark | no | 1.01 | 0 | converged | **no (ablation)** |

**Best hierarchy-faithful:** `H5L-B-NC-SIGMA-FLOOR` (`sigma_policy=sigma_floor`, `sigma_floor=0.05`, full geo×channel NC, learned τ).

**Benchmark rows:** `H5L-J`, `H5L-K` — diagnostics pass but `evidence_promotion_allowed=false`.

## Questions

### Did any hierarchy-faithful variant reach `converged_diagnostic_only`?

**Yes.** `H5L-B` and `H5L-E` — `any_hierarchy_faithful_converged_diagnostic_only=true` in the artifact.

### If yes, what exact config?

**H5L-B (best):** `parameterization=non_centered`, `hierarchy_policy=full_geo_channel_hierarchy`, `tau_parameterization=current`, `sigma_policy=sigma_floor` (`sigma_floor=0.05`), `beta_prior_policy=current_default`, `hierarchy_strength_policy=learned_tau`, H5j-D channel policy + prescale + extended MCMC.

**H5L-E:** same plus `hierarchy_strength_policy=strongly_regularized_tau`.

### If no, what remains the blocker?

For variants **without** σ floor: τ / `z_beta` / `mu_channel` funnel and tight log-scale σ remain the stressors. **σ floor** on the likelihood was sufficient to clear divergences while keeping full partial pooling on this pilot.

### Are pooled/fixed-τ ablations still only diagnostic?

**Yes.** They are replayed as benchmarks (`H5L-J`, `H5L-K`) and explicitly excluded from `evidence_promotion_allowed`.

### Should H5 real-panel eligibility require minimum geo count / collinearity constraints?

**Yes, recommended.** Full partial pooling on 3 geos with correlated channels is marginal; eligibility should require explicit `channel_policy` and documented geometry, and consider minimum geo count (e.g. ≥5) before shadow batching.

### Should the sample panel be deemed too small/collinear for full hierarchy?

**For production-grade full hierarchy: treat as stress-test only.** It is useful for research probes but not sufficient alone to approve prod Bayes.

### Next safe step

1. **Frozen pilot re-run** with documented policy: drop-collinear + prescale + `sigma_floor=0.05` + full hierarchy NC — **research eligibility only**, not prod Bayes.  
2. Validate on synthetic worlds that σ floor does not bias recovery.  
3. Do **not** batch additional real panels until re-run confirms.  
4. Never default to pooled/fixed-τ or silent σ floor.

### Production

**Blocked** — `approved_for_prod=false`, `prod_decisioning_allowed=false`.
