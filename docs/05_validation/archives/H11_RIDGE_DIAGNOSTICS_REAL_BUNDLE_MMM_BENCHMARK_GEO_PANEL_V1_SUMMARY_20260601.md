# Ridge production diagnostics

**Severity:** DIAGNOSTIC_ONLY (`diagnostic_only`)

## Output eligibility (H9)
- Allowed uses: ['model_fit_review', 'qa_regression_review', 'methodology_benchmark']
- Forbidden uses: ['budget_reallocation_claim', 'channel_level_causal_claim', 'clean_channel_attribution', 'clean_channel_lift_claim', 'clean_media_attribution', 'production_incrementality_claim']
- Human review required: **True**
- Diagnostic-only reason: Required vertical controls missing — diagnostic QA only.
- Classification triggers: ['claims:forbidden_claims_present', 'control:missing_required_vertical_controls']

## Control completeness
- Vertical: `retail`
- Missing required controls: ['holiday', 'promo_flag', 'unemployment_index']
- Omitted control risk: **True**
- Media-correlated controls: False

## Sparse channels
- Extreme sparse (near_zero ≥ 0.99): none

## Collinearity
- Max |ρ|: 0.060670394683298864
- Weak identification risk: **False**
- Correlated groups: none

## Transform
- Metadata complete: **True**
- Selected params: `{'decay': 0.36378537319927373, 'hill_half': 2.4287574992566916, 'hill_slope': 1.770223453733264, 'log_alpha': -0.27901266311609074, 'transform_search_selected': True}`

## Stability
- Fold stability OK: True
- Geo-fold RMSE mean: 17.718841930952195
- Coefficient stability available: True

## Forbidden claims
- `no_budget_reallocation_claim_based_only_on_this_run`
- `no_channel_level_causal_claim_without_caveat`
- `no_clean_media_attribution_claim`

## Top warnings
- control_completeness:missing_required:['holiday', 'promo_flag', 'unemployment_index']

## Calibration evidence (MIP-C1)
- **CalibrationSignal context:** not attached on this run (explicit).
- Collinearity replay flag: `False`

## Production boundary
- Ridge remains production baseline: True
- Bayes-H5 research-only: True
- Diagnostics are not hard gates: True
- Optimizer enabled: False (must be false)
- DecisionSurface enabled: False (must be false)
- Recommendations enabled: False (must be false)

_Diagnostic only — not budget optimization or Bayes promotion._
