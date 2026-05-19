# Feature separability governance

The **feature separability** extension is a **diagnostic and governance layer only**. It does not change training, feature engineering, model fitting, budget optimization, planning outputs, or attribution math.

## Coefficient stability interpretation

Bootstrap refits report **raw ridge coefficients** for transparency, but **instability flags** use **standardized effects** (`coefficient × feature column std` on the media design matrix). Raw coefficient variance alone is scale-sensitive and is **not** the primary separability signal.

**Primary:** within-group **contribution share** stability across bootstrap draws (`|coef| × mean feature`, normalized within the group).

**Secondary:** standardized effect sign-flip rate and CV.

The separability score weights contribution stability highest, then correlation/VIF, then standardized effects, then calibration evidence.

## Why prediction stability differs from attribution stability

A model can predict weekly revenue reasonably well while **splitting credit** between related media columns (for example `Meta_prospecting` and `Meta_retargeting`) in ways that are not stable or identifiable.

| Concern | What it measures |
|--------|------------------|
| **Prediction / fit** | How well the panel-level target is reproduced on held-out rows |
| **Attribution / separability** | Whether each split column has a **distinct, stable** incremental story |

High spend correlation, collinearity (VIF), sign flips across bootstrap refits, and contribution share swapping are signals that **split-level claims** may be misleading even when overall fit looks acceptable.

## Extension output

After training, `extension_report.feature_separability_report` includes:

- Per **feature group** analysis (`feature_group`, `member_columns`)
- `pairwise_correlations`, VIF reuse from identifiability
- Coefficient stability (bootstrap refits, same ridge alpha as the fit)
- Contribution share stability within the group
- Calibration evidence at split vs aggregate level
- Business importance (spend share, contribution share, optimization relevance)
- `separability_classification`: `high` | `medium` | `low`
- `recommended_action`: `keep_split` | `keep_with_caution` | `rollup_recommended` | `experiment_recommended`

Optional top-level sections:

- `rollup_recommendations`
- `experiment_recommendations`
- `unsupported_split_level_claims`
- `governance_summary` (release / optimization warnings — advisory)

## Group detection

By default, channels sharing the same **prefix before the first underscore** form a group when there are two or more members (for example `Meta_prospecting` + `Meta_retargeting` → group `Meta`).

Override with explicit groups in config:

```yaml
extensions:
  feature_separability:
    enabled: true
    feature_groups:
      Meta:
        - Meta_prospecting
        - Meta_retargeting
```

## Example: good separability

- Low pairwise correlation between split columns
- Stable coefficients (low sign-flip rate and coefficient CV)
- Stable within-group contribution shares across bootstrap draws
- Matched experiments at the **split** channel level

**Recommendation:** `keep_split` — safe to interpret and report separately with normal caveats.

## Example: poor separability

- Correlation above ~0.8 between splits
- Coefficient sign instability or high CV across bootstrap refits
- Contribution ranks swap across refits
- Only an aggregate-channel experiment (or none)

**Recommendation:** `rollup_recommended` for low business importance — roll up for reporting or modeling taxonomy review; **do not** treat splits as independent causal levers.

**Recommendation:** `experiment_recommended` only when separability is **low**, business importance is **high** (material spend or contribution share), **and** group spend share meets `experiment_min_group_spend_share` (default 3% of panel media). Tiny splits with low spend get `keep_with_caution` or `rollup_recommended` — not an automatic experiment mandate. Optimization gate approval alone does not trigger experiments.

## Rollup guidance

The library **never** merges columns automatically. Rollup means:

1. Change the **data contract** (fewer channel columns) in a new training run, or
2. Report and decide at a **parent** level while keeping splits in the panel for diagnostics only

Use `rollup_recommendations` in the extension report as an audit trail for release review.

## Experiment guidance

When `experiment_recommended` appears:

- Design lift tests that **isolate** the split (geo holdout, audience holdout, or platform split tests)
- Ensure `calibration.experiments_path` rows use the **same channel names** as the panel
- Re-run training after taxonomy changes; separability is evaluated on the **current** channel list

## Interpretation guidance

| Classification | Meaning |
|----------------|---------|
| `high` | Split-level interpretation is structurally supported |
| `medium` | Use splits with explicit uncertainty; avoid precise ROI at split level |
| `low` | Do not interpret splits independently; roll up or experiment |

`unsupported_split_level_claims` flags when low separability conflicts with split-level reporting or optimization interpretation — it does **not** block training.

## Configuration

See `extensions.feature_separability` in [../01_getting_started/config_yaml.md](../01_getting_started/config_yaml.md). Thresholds cover correlation bands, VIF bands, sign-flip rate, coefficient CV, and contribution share variance.

Bootstrap rounds reuse `extensions.identifiability` settings by default to avoid duplicate expensive tuning.

## Related docs

- [experiment_scheduler.md](experiment_scheduler.md) — prioritize experimentation budget from separability + calibration
- [diagnostics.md](diagnostics.md) — identifiability and collinearity
- [calibration.md](calibration.md) — experiment matching
- [../04_governance/prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) — decision-grade artifacts
