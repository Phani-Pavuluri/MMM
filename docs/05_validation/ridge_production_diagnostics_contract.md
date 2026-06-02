# Ridge production diagnostics contract (H7)

**Status:** Accepted  
**Date:** 2026-06-01  
**Scope:** Production-safe Ridge path metadata only — not hard gates, not Bayes promotion

## Purpose

Emit **decision-safety diagnostics** on the production Ridge baseline path, aligned in spirit with H5 shadow diagnostics:

- transform reporting
- control completeness vs vertical profiles
- sparse-channel and collinearity risk
- fold / coefficient stability
- forbidden claims when evidence is insufficient

**Ridge remains the production baseline.** Bayes-H5 remains research-only. This contract does not change optimizer, DecisionSurface, or recommendation behavior.

## Report identity

| Field | Type | Description |
|-------|------|-------------|
| `report_version` | string | `mmm_ridge_production_diagnostics_v1` |
| `artifact_kind` | string | `RIDGE_PRODUCTION_DIAGNOSTICS` |
| `model_id` | string | e.g. `ridge_bo` |
| `run_id` | string | Run / data version id |
| `dataset_snapshot_id` | string | Frozen panel snapshot id |
| `run_environment` | string | `prod` or `research` |

## Transform (`transform_diagnostics`)

| Field | Description |
|-------|-------------|
| `transform_config` | Config adstock/saturation types |
| `selected_adstock_saturation` | BO-selected `decay`, `hill_half`, `hill_slope`, `log_alpha` |
| `raw_media_columns` | Input spend/exposure columns |
| `transformed_media_columns` | Post-transform feature names when available |
| `metadata_complete` | All selected transform params present |
| `warnings` | e.g. missing best_params metadata |

Missing transform metadata **must** produce a warning (not silent).

## Controls (`control_completeness`)

| Field | Description |
|-------|-------------|
| `vertical_id` | `retail`, `cpg`, `auto`, or null |
| `required_controls_by_vertical` | Profile required set |
| `optional_controls_by_vertical` | Profile optional set |
| `controls_present` | Columns in schema |
| `missing_controls` | Recommended but absent |
| `missing_required_controls` | Required vertical controls absent |
| `omitted_control_risk` | true when required controls missing |
| `media_correlated_controls` | Confounding stress flag |

When `omitted_control_risk` is true, **forbidden claims** must include:

- `no_clean_media_attribution_claim`
- `no_channel_level_causal_claim_without_caveat`
- `no_budget_reallocation_claim_based_only_on_this_run`

## Collinearity (`collinearity`)

| Field | Description |
|-------|-------------|
| `max_abs_correlation` | Max \|ρ\| across media |
| `weak_identification_risk` | true when max \|ρ\| ≥ 0.85 |
| `collinear_channel_groups` | Groups at \|ρ\| ≥ 0.95 |
| `calibration_evidence_available` | External calibration flag |

When weak ID without calibration: forbid clean separate channel-effect claims.

## Sparse channels (`sparse_channels`)

| Field | Description |
|-------|-------------|
| `near_zero_threshold` | Default 0.99 (H5r/H6f) |
| `by_channel` | `near_zero_share` per channel |
| `sparse_channel_extreme` | Channels above threshold |
| `silent_drop_occurred` | **Must remain false** in diagnostics |

Extreme sparse channels add forbidden claims: `no_separate_channel_effect_claim_for_{channel}`.

## Stability

| Block | Fields |
|-------|--------|
| `fold_stability` | `geo_fold_rmse`, `geo_fold_rmse_mean`, `fold_stability_ok` |
| `coefficient_stability` | `media_coef_by_channel`, `sign_by_channel` |
| `sign_plausibility` | `sign_match_rate_vs_known_mu` (when truth available) |
| `response_curve_plausibility` | decay/hill plausibility warnings |
| `lift_simulation_stability` | report-only; optimizer not invoked |

## Governance

| Field | Description |
|-------|-------------|
| `forbidden_claims` | Sorted list of blocked business claims |
| `diagnostic_severity` | `none` \| `low` \| `medium` \| `high` |
| `warnings` | Aggregated diagnostic warnings |
| `production_flags` | `approved_for_prod` always false on this report |
| `outputs_are_diagnostic_only` | true |

### Forbidden outputs (must not appear on report)

- `decision_surface`
- `optimizer_ready_curves`
- `budget_recommendation`
- `recommendation`
- `production_decision_surface`
- `optimizer_output`

## Implementation

- Module: `mmm/diagnostics/ridge_diagnostics.py`
- Vertical profiles: `mmm/config/vertical_control_profiles.py`
- H6 regression: `tests/diagnostics/test_ridge_production_diagnostics.py`
- Example artifact: `docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json`

## Related

- [Bayes-H6 synthetic lane ADR](bayes_h6_synthetic_lane_adr.md)
- [INV-H6F benchmark matrix](../06_investigations/INV-H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX.md)
