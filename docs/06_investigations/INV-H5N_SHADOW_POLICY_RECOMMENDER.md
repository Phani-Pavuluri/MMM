# INV-H5N — Bayes-H5 Shadow-Policy Recommender (Planned)

**Investigation ID:** INV-H5N  
**Status:** planned — blocked on H5m  
**Roadmap:** Bayes-H5n  
**ADR:** [bayes_h5_model_spec_improvement_adr.md § H5n](../05_validation/bayes_h5_model_spec_improvement_adr.md#h5n-shadow-policy-recommender-planned)

## Purpose

Build a **research-only** recommender that converts real-panel diagnostics into **allowed H5 shadow-run policy suggestions** — the bridge between H5i–H5l findings and governed policies like [h5m_sample_panel_shadow_policy.json](h5m_sample_panel_shadow_policy.json).

**Does not:** run production Bayes; wire prod TrustReport; connect posterior to optimizer; emit DecisionSurface or budget recommendations; replace Ridge; promote ablation-only specs (pooled, fixed-τ) as evidence-ready.

## Planned inputs

- Panel diagnostics
- Collinearity groups
- Sparsity diagnostics
- Convergence diagnostics
- Prior H5j / H5l ablation results
- Optional business / channel metadata
- Optional calibration evidence availability

## Planned outputs

- `recommended_channel_policy`
- `recommended_geometry_config`
- Sampler profile suggestion
- Allowed alternatives
- Blocked options
- Interpretation changes
- Forbidden claims
- Evidence status
- `production_flags` false

## Governance

- No silent channel dropping or compositing
- No separate channel-effect claims after collapse/drop
- All recommendations include rationale + forbidden claims
- Recommends **shadow policies only** — not business or production decisions

## Sequencing

| Milestone | Gate |
|-----------|------|
| H5m | Frozen policy replay validates governed config path |
| **H5n** | Recommender artifacts available |
| H5o+ | Second real panel only after H5n — collinearity via explicit policy, not silent fixes |

## Production

**Blocked.**
