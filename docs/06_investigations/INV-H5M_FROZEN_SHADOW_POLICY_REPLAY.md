# INV-H5M — Bayes-H5 Frozen Shadow-Policy Replay

**Investigation ID:** INV-H5M  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Source:** H5L @ `7845909` (H5L-B hierarchy-faithful config)  
**Frozen policy:** [h5m_sample_panel_shadow_policy.json](h5m_sample_panel_shadow_policy.json)  
**Replay artifact:** [BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)

## Purpose

Prove the successful H5L-B configuration is **reproducible from governed frozen policy JSON** via the standard H5 shadow runner (`--policy-path`), not ad hoc geometry-runner logic.

## Source H5L-B configuration

| Field | Value |
|-------|--------|
| Panel | `examples/sample_panel.csv` |
| panel_id | `examples_mmm_sample_panel_v1` |
| dataset_snapshot_id | `mmm-examples-sample-panel-frozen-2022` |
| Channels (frozen) | drop **tv**, keep **search** + **social** |
| Geometry | NC full hierarchy, learned τ, `sigma_floor=0.05` |
| Prescale | z-score media + z-score log outcome |
| Sampler | extended 600/600/4, `target_accept=0.95` |
| H5L-B result | rhat≈1.01, 0 div, `converged_diagnostic_only` |

**Note:** The H5L geometry runner used a collinear **heuristic** that dropped `social` on the same panel. H5m encodes the **governed drop-tv** variant with explicit `dropped_channels` / `kept_channels` (no silent dropping).

## Replay command

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_runner \
  --policy-path docs/06_investigations/h5m_sample_panel_shadow_policy.json \
  --output-path docs/05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json
```

## Replay result (latest run)

| Metric | H5L-B (reference) | H5m replay |
|--------|-------------------|------------|
| rhat_max | 1.01 | **1.01** |
| divergences | 0 | **0** |
| convergence_status | converged_diagnostic_only | **converged_diagnostic_only** |
| channels | heuristic dropped social | **explicit drop tv** |
| evidence_promotion_allowed | true (research) | **true** (research only) |

| Check | Status |
|-------|--------|
| policy_id on artifact | `bayes_h5m_sample_panel_shadow_policy_v1` |
| channel_policy_applied | drop tv, keep search+social |
| h5_geometry_config_applied | sigma_floor=0.05, NC full hierarchy |
| production flags | all false |

If MCMC sampling drifts, the artifact records it honestly and sets `evidence_promotion_allowed=false`.

## Did H5L-B reproduce?

Compare replay diagnostics to H5L-B. Same governed **intent** (hierarchy-faithful + σ floor); channel set may differ from heuristic H5L run (drop-tv vs drop-social). Convergence should be `converged_diagnostic_only` when sampling cooperates.

## Interpretation changes (drop tv)

- No separate **tv** effect claim after tv is dropped.
- **Social** may absorb shared social–tv movement; no clean social-only causal claim without external calibration.
- Forbidden claims are listed on the policy and artifact.

## Production boundary

- `hard_gate=false`, `production_promotion=false`, `approved_for_prod=false`, `prod_decisioning_allowed=false`
- No optimizer, DecisionSurface, budget recommendations, prod TrustReport, or Ridge replacement
- σ floor is **not** a production default

## H5n recommender

**H5n may proceed.** Replay converged under frozen drop-tv policy with full governance metadata on the artifact.

## Conclusion

- Frozen policy validates and drives the shadow runner without duplicate CLI args.
- Replay reached `converged_diagnostic_only` (rhat 1.01, 0 divergences) on this run.
- Production Bayes remains blocked; `evidence_promotion_allowed` is research eligibility only.

## Before a second real panel

1. Successful H5m replay with `converged_diagnostic_only` under frozen policy  
2. H5n recommendation artifact for the next panel  
3. Explicit collinearity policy + forbidden claims — not silent channel fixes
