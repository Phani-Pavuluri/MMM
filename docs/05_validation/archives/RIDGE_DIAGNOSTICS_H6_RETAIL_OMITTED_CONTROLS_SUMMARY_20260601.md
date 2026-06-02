# Ridge production diagnostics

**Severity:** CRITICAL (`high`)

## Control completeness
- Vertical: `retail`
- Missing required controls: ['holiday', 'promo_flag', 'unemployment_index']
- Omitted control risk: **True**
- Media-correlated controls: False

## Sparse channels
- Extreme sparse (near_zero ≥ 0.99): none

## Collinearity
- Max |ρ|: 0.9722864384403749
- Weak identification risk: **True**
- Correlated groups: [['ctv', 'display']]

## Transform
- Metadata complete: **True**
- Selected params: `{'decay': 0.6559140851113047, 'hill_half': 2.4499189178222287, 'hill_slope': 1.551318381339807, 'log_alpha': 1.7125917412308524, 'transform_search_selected': True}`

## Stability
- Fold stability OK: False
- Geo-fold RMSE mean: 31229.862114309828
- Coefficient stability available: True

## Forbidden claims
- `collinear_group_ctv_display:forbid_isolated_channel_lift_claims`
- `no_budget_reallocation_claim_based_only_on_this_run`
- `no_channel_level_causal_claim_without_caveat`
- `no_clean_media_attribution_claim`
- `no_clean_separate_channel_effect_claim_without_external_calibration`

## Top warnings
- collinearity:weak_identification_risk:max_abs_corr=0.972
- control_completeness:missing_required:['holiday', 'promo_flag', 'unemployment_index']

## Production boundary
- Ridge remains production baseline: True
- Bayes-H5 research-only: True
- Diagnostics are not hard gates: True
- Optimizer enabled: False (must be false)
- DecisionSurface enabled: False (must be false)
- Recommendations enabled: False (must be false)

_Diagnostic only — not budget optimization or Bayes promotion._
