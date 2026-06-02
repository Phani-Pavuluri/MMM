# INV-H5M — Bayes-H5 Frozen Shadow-Policy Replay

**Investigation ID:** INV-H5M  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Source:** H5L @ `7845909` (H5L-B best hierarchy-faithful config)  
**Frozen policy:** [h5m_sample_panel_shadow_policy.json](h5m_sample_panel_shadow_policy.json)  
**Replay artifact:** [BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)

## Purpose

Validate that H5L-B is **reproducible and governed** via a frozen policy JSON replayed through the standard H5 shadow runner — not an ad hoc geometry runner invocation.

## Frozen policy summary

| Field | Value |
|-------|--------|
| `policy_id` | `bayes_h5m_sample_panel_shadow_policy_v1` |
| Channel | `drop_collinear_channels` @ 0.95 — explicit drop **social**, keep **search** + **tv** (H5L-B heuristic) |
| Geometry | NC full hierarchy, learned τ, `sigma_floor=0.05` |
| Prescale | z-score media + z-score log outcome |
| Sampler | extended 600/600/4, `target_accept=0.95` |

## Questions

### Did the frozen policy reproduce H5L-B?

**Partially — same governed config, sampling drift on divergences.**

| Metric | H5L-B | H5M replay |
|--------|-------|------------|
| rhat_max | 1.01 | 1.01 |
| divergences | 0 | **3** |
| status | converged_diagnostic_only | **weak_convergence** |

Channel/geometry/sampler policies match; MCMC stochasticity left 3 divergences this run. Policy replay is still valid for governance (frozen JSON → shadow runner) but does not guarantee identical diagnostics every seed.

### Were channel/geometry/sampler policies recorded?

Yes — `channel_policy_declared`, `channel_policy_applied`, `geometry_config_applied`, `sampler_profile_applied` on the artifact envelope. Applied drop: **social**; kept: **search**, **tv** (collinear heuristic, same as H5L-B).

### Did convergence remain acceptable?

**Weak_convergence** on replay (`rhat_max=1.01`, 3 divergences). `evidence_promotion_allowed=false` until a re-run clears divergences.

### Schema-compliant outputs?

Artifact uses `real_panel_shadow_artifact` + H5E shadow-run record; no optimizer / DecisionSurface / recommendations.

### Still research-only?

**Yes.** `approved_for_prod=false`, `prod_decisioning_allowed=false`, `hard_gate=false`. σ floor is **not** a production default.

### Before a second real panel?

1. Confirm replay matches H5L-B diagnostics.  
2. Optional synthetic-world check that σ floor does not bias recovery.  
3. Document minimum geo count / collinearity gates in shadow protocol.  
4. Do **not** treat ablation benchmarks (pooled/fixed-τ) as promotion evidence.

### Production

**Blocked.**
