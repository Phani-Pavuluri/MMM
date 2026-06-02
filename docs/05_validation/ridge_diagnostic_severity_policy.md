# Ridge diagnostic severity policy (H9)

**Status:** Accepted  
**Date:** 2026-06-01  
**Scope:** Governed interpretation of Ridge diagnostic reports — **not** automatic hard gates

## Purpose

H8 made Ridge diagnostics visible to operators. H9 defines **consistent severity levels** and **output eligibility** so two analysts interpret the same run the same way.

Diagnostics **do not** change Ridge fitting, optimizer execution, or DecisionSurface generation. They govern **what humans may claim** from a run.

## Severity levels

| Level | Meaning |
|-------|---------|
| `clean` | No material diagnostic issues; routine production review. |
| `info` | Minor notes; standard coefficient and fit review allowed. |
| `warning` | Review recommended; some interpretive caveats apply. |
| `restricted_interpretation` | Channel- or aggregate-level claims need explicit caveats; no clean isolation. |
| `diagnostic_only` | Fit may be reviewed for QA; **not** for business attribution or budget claims. |
| `blocked_for_decision_use` | Report integrity failure; do not use for any decision narrative. |

## Output eligibility by level

### `clean`

| Field | Value |
|-------|-------|
| **Allowed uses** | `model_fit_review`, `coefficient_review`, `aggregate_performance_review`, `planning_input_with_standard_caveats` |
| **Forbidden uses** | (none beyond global forbidden claims list) |
| **Human review** | Optional |
| **Optimizer / DecisionSurface** | Unchanged — may consume run per existing prod gates (not blocked by this policy) |

**Example triggers:** No warnings; controls complete; no weak-ID or sparse extremes.

### `info`

| Field | Value |
|-------|-------|
| **Allowed uses** | Same as `clean` |
| **Forbidden uses** | `unsupported_strong_causal_claim` |
| **Human review** | Optional |
| **Optimizer / DecisionSurface** | Unchanged |

**Example triggers:** Single low-severity warning; optional controls missing only with no omitted-control risk.

### `warning`

| Field | Value |
|-------|-------|
| **Allowed uses** | `model_fit_review`, `coefficient_review`, `aggregate_performance_review` |
| **Forbidden uses** | `clean_channel_attribution`, `budget_reallocation_claim` without review sign-off |
| **Human review** | Recommended |
| **Optimizer / DecisionSurface** | Unchanged |

**Example triggers:** Missing optional vertical controls; incomplete transform metadata; fold stability marginal.

### `restricted_interpretation`

| Field | Value |
|-------|-------|
| **Allowed uses** | `model_fit_review`, `aggregate_diagnostic_review`, `collinearity_audit` |
| **Forbidden uses** | `clean_channel_attribution`, `clean_channel_lift_claim`, `budget_reallocation_claim`, `isolated_sparse_channel_claim` |
| **Human review** | **Required** |
| **Optimizer / DecisionSurface** | Unchanged — operator must not treat optimizer output as channel-proof |

**Example triggers:** High collinearity without calibration; extreme sparse channel (`near_zero_share ≥ 0.99`); media–control confounding; collinear groups without external calibration.

### `diagnostic_only`

| Field | Value |
|-------|-------|
| **Allowed uses** | `model_fit_review`, `qa_regression_review`, `methodology_benchmark` |
| **Forbidden uses** | `clean_media_attribution`, `channel_level_causal_claim`, `budget_reallocation_claim`, `production_incrementality_claim` |
| **Human review** | **Required** |
| **Optimizer / DecisionSurface** | Unchanged — **do not** promote results to decision-grade narrative |

**Example triggers:** Missing required vertical controls; severe geo-fold instability; diagnostics unavailable in research context.

### `blocked_for_decision_use`

| Field | Value |
|-------|-------|
| **Allowed uses** | `engineering_debug_only` |
| **Forbidden uses** | All business and planning claims |
| **Human review** | **Required** before any reuse |
| **Optimizer / DecisionSurface** | Unchanged mechanically — **operator must not run decide** on this narrative |

**Example triggers:** Forbidden production fields present on diagnostic artifact (`decision_surface`, `budget_recommendation`, etc.).

## Classification rules (implementation)

Implemented in `mmm/diagnostics/ridge_severity_policy.py` → `classify_ridge_diagnostic_severity(report)`:

1. Forbidden production artifact fields → `blocked_for_decision_use`
2. `status == unavailable` → `diagnostic_only`
3. Missing required vertical controls → `diagnostic_only`
4. Severe fold instability (`fold_stability_ok == false`) → `diagnostic_only`
5. Weak ID without calibration OR extreme sparse OR media-correlated controls → `restricted_interpretation`
6. Missing optional controls OR missing transform metadata → `warning` (if not already worse)
7. Residual warnings only → `info`
8. Otherwise → `clean`

Worst applicable level wins.

## Report fields (H9)

Each `ridge_production_diagnostics_report` includes:

```json
{
  "severity_policy_version": "mmm_ridge_severity_policy_v1",
  "severity": "restricted_interpretation",
  "output_eligibility": {
    "severity": "restricted_interpretation",
    "allowed_uses": ["model_fit_review", "aggregate_diagnostic_review"],
    "forbidden_uses": ["clean_channel_attribution", "budget_reallocation_claim"],
    "human_review_required": true,
    "diagnostic_only_reason": null,
    "classification_triggers": ["collinearity:weak_identification_risk"],
    "optimizer_decision_surface_unchanged": true,
    "diagnostics_are_not_hard_gates": true
  }
}
```

Legacy field `diagnostic_severity` (`none` / `low` / `medium` / `high`) is retained for backward compatibility and mapped from policy severity.

## Boundaries

- Bayes-H5 remains research-only.
- No new budget recommendations or DecisionSurface outputs from diagnostics.
- Policy is **interpretation governance**, not a substitute for `production_readiness_report` or decide-time gates.

## Related

- [ridge_production_diagnostics_contract.md](ridge_production_diagnostics_contract.md)
- H8: `mmm/diagnostics/ridge_diagnostic_summary.py`
