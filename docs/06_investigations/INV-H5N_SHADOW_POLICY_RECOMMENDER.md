# INV-H5N — Bayes-H5 Shadow-Policy Recommender

**Investigation ID:** INV-H5N  
**Status:** complete (research lane)  
**Date:** 2026-06-01  
**Roadmap:** Bayes-H5n  
**ADR:** [bayes_h5_model_spec_improvement_adr.md § H5n](../05_validation/bayes_h5_model_spec_improvement_adr.md#h5n-shadow-policy-recommender-inv-h5n)  
**Module:** `mmm/research/bayes_h3_sandbox/h5_shadow_policy_recommender.py`  
**Sample artifact:** [BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](../05_validation/archives/BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json)

## Purpose

Convert real-panel **diagnostics** and prior H5 experiment evidence into **governed shadow-policy recommendations** — explicit channel + geometry + sampler suggestions with rationale, allowed alternatives, blocked options, interpretation changes, and forbidden claims.

The recommender recommends **shadow-run policies for research**, not business or production decisions.

**Does not:** run additional real panels; promote H5 to production; wire prod TrustReport; connect posterior to optimizer; emit DecisionSurface or budget recommendations; enable hard gates; replace Ridge; silently drop or merge channels; label ablation-only geometry as promotable.

## Source evidence (H5j / H5l / H5m)

| Milestone | Role |
|-----------|------|
| **H5j** | Collinearity ablations — social–tv \|ρ\| ≈ 0.99; keep-all weak; drop/composite probes |
| **H5l** | Hierarchy-faithful **H5L-B** (`sigma_floor=0.05`, NC, learned τ) converged; runner heuristic dropped social; ablation pooled/fixed-τ benchmarks only |
| **H5m** | Frozen policy `bayes_h5m_sample_panel_shadow_policy_v1` — **governed explicit drop tv**, keep search + social; replay `converged_diagnostic_only` (rhat 1.01, 0 div) |

**Governance nuance:** H5L runner heuristic dropped social; H5m policy drops **tv** with explicit `dropped_channels` / `kept_channels`. H5n recommends the **H5m governed pattern** (explicit, policy-owned drop).

## Recommender rules (summary)

### Collinearity

- `max_abs_corr < threshold` → `keep_all_channels_with_weak_id_warning` allowed when diagnostics clean.
- High correlation + governed drop converged → **`drop_collinear_channels`** (explicit lists, `no_silent_dropping=true`).
- High correlation + strategically inseparable channels → **`composite_media_channel`** as allowed alternative.
- High correlation + business-critical separation → **`external_calibration_required`**.
- High correlation + no converged governed remedy → **`do_not_run_h5_shadow_until_panel_fixed`**.

### Sparsity

- Near-zero variation → block separate coefficient claim; suggest single-channel diagnostic or governed suppress.

### Convergence

- Keep-all failed + governed drop converged → recommend governed drop.
- Only pooled/fixed-τ ablations converged → **`ablation_benchmark_only`**, `evidence_promotion_allowed=false`.
- `evidence_promotion_allowed=true` only for hierarchy-faithful `converged_diagnostic_only` configs.

### Interpretation

- Dropped channel → forbid separate effect claim.
- Correlated retained channels → forbid clean isolated channel claim without calibration.
- Composite → combined-media-block effect only.
- Keep-all under weak ID → no channel-level decision use.

### Production

- All production flags **false**; no optimizer / DecisionSurface / recommendations.

## Sample-panel recommendation

**Recommended:** align with [h5m_sample_panel_shadow_policy.json](h5m_sample_panel_shadow_policy.json)

| Field | Value |
|-------|--------|
| Channel | drop **tv**, keep **search** + **social** |
| Geometry | NC full hierarchy, learned τ, `sigma_floor=0.05` |
| Prescale | z-score media + z-score log outcome |
| Sampler | extended MCMC 600/600/4, `target_accept=0.95` |
| Rationale | social–tv collinearity + H5m governed replay convergence |
| Frozen policy id | `bayes_h5m_sample_panel_shadow_policy_v1` |

### Allowed alternatives

- **Composite social/tv (PC1):** allowed alternative, not selected over governed drop.
- **External calibration:** if business metadata requires separable claims.

### Blocked options

- **Keep all:** blocked for evidence (weak ID / keep-all convergence failure under collinearity).
- **Pooled / fixed-τ:** ablation benchmark only — `do_not_use_ablation_for_promotion`.

### Interpretation changes

- No separate **tv** effect after drop.
- **Social** may absorb shared social–tv movement — no clean social-only causal claim without calibration.
- **Search** / **social** partial-pooling coefficients only; correlated pair caveats apply.

### Forbidden claims

- No separate tv effect after explicit drop.
- No clean isolated social-only effect without external calibration.
- No production Bayes, optimizer, DecisionSurface, budget recommendations, or Ridge replacement.
- Shadow recommendations are not business decisions.

### Evidence status

- `evidence_promotion_allowed=true` (research eligibility only)
- `convergence_status=converged_diagnostic_only` (from H5m replay)
- Production Bayes **blocked**

## Generate artifact

```bash
poetry run python -m mmm.research.bayes_h3_sandbox.h5_shadow_policy_recommender
```

## Production boundary

- `approved_for_prod=false`, `prod_decisioning_allowed=false`, `hard_gate=false`
- `excluded_fields`: DecisionSurface, optimizer, recommendations, budget_recommendation

## Requirements before H5o (second real panel)

1. H5m frozen policy replay at `converged_diagnostic_only` under governed channel policy.
2. **H5n recommendation artifact** for the target panel (this investigation).
3. Shadow run uses **recommender-assisted or frozen policy** — no silent channel fixes.
4. Forbidden claims and interpretation changes recorded on policy + artifact.

## Conclusion

- H5n emits a governed recommendation artifact for the sample panel pointing at the H5m frozen policy.
- Alternatives and blocked options are explicit; forbidden claims are recorded.
- Production Bayes remains blocked; **H5o** may proceed with recommender-assisted shadow policy on a second real panel.
