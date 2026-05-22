# Experiment evidence platform (Phase 1)

Phase 1 adds **opt-in**, **diagnostic-first** contracts for experiment lifecycle evidence. Default Ridge replay calibration behavior is unchanged unless you enable the new flags.

## Contracts

- **`ExperimentEvidence`** — canonical payload (lift, SE, geo scope, KPI, lineage, approval).
- **`ExperimentEvidenceRegistry`** — register, validate, query, mark accepted/rejected/expired. Does **not** run experiments or auto-calibrate.
- **`ExperimentCompatibilityResolver`** — maps evidence to MMM panel scope → `ReplayCompatibilityDecision`.
- **`CounterfactualShockPlanner`** — ranks allocation methods for replay bridges (never DMA truth from national deltas).
- **`EvidenceQualityScore`** — `evidence_weight` and `quality_tier` for future weighted replay / Bayesian likelihood.

## Configuration (all opt-in, defaults off)

```yaml
calibration:
  evidence_registry_path: path/to/evidence.json
  compatibility_resolver_enabled: true
  evidence_weighting_enabled: true    # PR 2: Ridge BO weighted replay (requires replay_mode below)
  replay_mode: evidence_registry      # legacy | evidence_registry
  use_replay_calibration: true        # required for BO replay term
  hierarchical_regularization_enabled: false  # deprecated; use hierarchy.enabled (see hierarchical_borrowing.md)
  model_geo_granularity: dma
  channel_mapping: { platform_tv: tv }
```

## Artifacts

When enabled, `extension_report` may include:

- `experiment_compatibility_report`
- `evidence_weighting_report`
- `counterfactual_shock_plan`
- `experiment_evidence_registry_coverage`
- `governance_unsupported_claims`

## Validation loop (continuous + decision)

After experiments are registered and model runs are archived locally:

- **[continuous_validation.md](continuous_validation.md)** — compare **prior run predicted lift** vs **new experiment lift** (`continuous_validation_report`).
- **[decision_validation.md](decision_validation.md)** — compare **prior recommendations** vs **subsequent experiment lift** (`decision_validation_report`).

Both are **opt-in**, **diagnostic-only**, and require operator-maintained local registries (`registry_dir`, `decision_registry_dir`). They do not auto-retrain, promote runs, or change budgets.

## Guardrails

- Registry stores and validates only.
- Aggregate national/user experiments cannot support DMA subgeo claims.
- Allocated shocks are computational bridges only.
- Bayesian prod decisioning and production monetary CIs remain blocked per existing policy.

## Ridge weighted replay (PR 2)

When **all** of the following are set:

- `use_replay_calibration: true`
- `replay_mode: evidence_registry`
- `evidence_weighting_enabled: true`
- `evidence_registry_path` + `compatibility_resolver_enabled: true`

Ridge+BO uses **weighted** replay loss in the composite objective:

```
weighted_error_i = w_i * ((mmm_lift_i - experiment_lift_i) / se_i)^2
weighted_replay_loss = sum(weighted_error_i) / sum(w_i)
```

`legacy` mode (default) is unchanged: `replay_units_path` + unweighted mean standardized error.

### Evidence usage rules (objective)

| Status | In BO loss? |
|--------|-------------|
| `compatible` / `aggregate_only` / `allocation_required` (with acceptable shock) | Yes, if quality weight > 0 |
| `diagnostic_only` / `rejected` | No (report only) |
| Missing SE (prod) | Rejected |
| Missing SE (research) | Conservative SE + warning |
| `expired` approval | Rejected |
| Stale freshness | Downweighted, may still contribute |

### Production gate (PR 2.5)

Evidence-registry replay is supported in **prod** only when `evidence_weighted_replay_summary` passes `assert_evidence_registry_replay_production_ready`:

- `n_evidence_units_used >= 1`
- At least one **high** or **medium** quality unit in the objective
- No **rejected** / **diagnostic_only** units in the objective set
- **Positive `lift_se`** on every used unit unless `calibration.allow_missing_se_in_prod_evidence_replay: true`
- **Aggregate-only** units must declare `supports_subgeo_claims=false` (no DMA/subgeo claims from national experiments)
- **Allocated shocks** must declare `allocation_role=computational_bridge_only`
- Channels with loaded registry evidence cannot be **critically rejected** without a used unit

Weighted experiment evidence is **calibration evidence** for the Ridge BO objective — not causal proof and not a substitute for designed geo experiments.

### Limitations

- **Aggregate-only** national/user-level evidence on a DMA panel: aggregate replay only; `supports_subgeo_claims=false`.
- **Allocated shocks** are computational bridges only — not experimental DMA truth.
- No automatic experiment execution, retraining, or budget changes.

## Bayesian experiment likelihood (PR 3)

Opt-in PyMC term when `bayesian.use_experiment_likelihood: true`:

```
experiment_lift_obs ~ Normal(mu=mmm_implied_lift, sigma=adjusted_SE)
```

- **Research-only** (`exp_likelihood_research_only` must remain true).
- **Prod decisioning remains blocked** for Bayesian.
- Report: `bayesian_experiment_likelihood_report` on `extension_report` (tier: research).

## PR phasing

1. Contracts + diagnostic reports
2. Ridge weighted replay integration
3. Bayesian experiment likelihood (research-only)
4. Hierarchical borrowing
5. Uncertainty + robust optimization (research)
6. Continuous + decision validation
