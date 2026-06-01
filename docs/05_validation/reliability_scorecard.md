# ReliabilityScorecard

Aggregate synthetic world certification reports into a **metric-class-aware** reliability view. Phase 4C introduced the MVP rollup; **Phase 5D** separates decision-grade, attribution-diagnostic, structural, and trust-modifier evidence per [reliability_threshold_governance.md](reliability_threshold_governance.md) and INV-056.

This is synthetic evidence aggregation — **not** a production release gate while thresholds remain `TBD_v1_runtime`.

## Artifact

**Filename:** `validation/synthetic_reliability_scorecard.json`  
**Module:** `mmm/validation/synthetic/reliability_scorecard.py`

## Input worlds (default)

| World | Primary evidence |
|-------|------------------|
| `WORLD-008-exact-recovery` | Coef / transform / Δμ recovery |
| `WORLD-009-optimizer-recovery` | Optimizer recovery (VAL-005) |
| `WORLD-010-replay-recovery` | Replay implied lift (VAL-006) |
| `WORLD-011-drift-recovery` | Drift degradation / readiness |
| `WORLD-012-identifiability-recovery` | Collinearity / VIF / readiness |

If `synthetic_world_certification_report.json` is missing under a bundle, the scorecard builder may **materialize** the DGP panel and **run** `run_world_certification(..., include_recovery=True)` first.

## Output fields

| Field | Meaning |
|-------|---------|
| `scorecard_version` | `reliability_scorecard_v1.2.0` |
| `world_ids` / `worlds_certified` | Requested vs successfully loaded |
| `capability_summary` | Per-capability entries, scores, failures, partials |
| `metric_class_by_capability` | Maps each capability group → governance class |
| `decision_reliability_score` | Mean of **decision_grade** capabilities (Δμ, optimizer, replay) |
| `attribution_diagnostic_score` | Mean of **diagnostic_attribution** (coef, transform) |
| `structural_reliability_score` | Mean of **structural** capabilities |
| `trust_modifier_status` | Object: `status`, `min_score`, `drift_severity_max`, `warnings`, failures/partials |
| `trust_report_interpretation` | Unified TrustReport rollup — [trust_report_semantics.md](trust_report_semantics.md) |
| `overall_evidence_score` | Mean of all scored capabilities (undifferentiated rollup) |
| `reliability_score` | **Deprecated alias** of `overall_evidence_score` |
| `interpretation_rules` | Machine-readable rules from governance doc |
| `status_counts` | pass / partial / fail / skipped / unsupported |
| `executed_validations` | Check IDs that ran (non-skipped) |
| `skipped_validations` | Skipped rows with reasons |
| `partial_validations` | e.g. `VAL-012` partial drift detection |
| `failed_validations` | `world_id:check_id` failures |
| `scored_capabilities` / `unscored_capabilities` | Coverage of capability groups |
| `coverage_ratio` | Scored in-scope checks / in-scope check slots |
| `release_readiness_interpretation` | **Conservative** synthetic-only label; prioritizes structural + decision + trust |
| `limitations` / `required_warnings` / `open_investigations` | Explicit non-goals |

## Capability groups

`structural_integrity`, `transform_consistency`, `coefficient_recovery`, `delta_mu_recovery`, `optimizer_recovery`, `replay_recovery`, `drift_behavior`, `identifiability_behavior`, `platform_contract_compatibility`, `artifact_integrity`, `governance_reaction`

## Scoring (MVP)

| Outcome | Score |
|---------|-------|
| pass | 1.0 |
| partial | 0.5 |
| fail | 0.0 |
| skipped / unsupported | not scored (`null`) |

**Expected skips** per world (e.g. `REC-4B5-DRIFT-COEF` on WORLD-011) are **not penalized** — excluded from the scored set.

`overall_evidence_score` = mean of **capability-level** means over capabilities with at least one scored in-scope check (`overall_evidence_score_method`: `mean_capability_score_v1`).

**Do not** use `overall_evidence_score` or `reliability_score` alone for release decisions. Inspect class scores per [reliability_threshold_governance.md §5](reliability_threshold_governance.md#5-reliabilityscorecard-interpretation).

## Interpretation rules (Phase 5D)

1. A model may be **decision-usable** with weak coefficient recovery if `decision_reliability_score` is high and `trust_modifier_status` is acceptable.
2. A model may be **attribution-unsafe** when `attribution_diagnostic_score` is low even if decision score is high.
3. Severe drift or identifiability may block via `trust_modifier_status: degraded` even when Δμ passes.
4. Curves and decomposition remain diagnostic unless an attribution certification profile exists.

## What it proves

- Structural certification (CERT-4A) holds on recovery worlds
- Targeted behavioral recovery evidence exists where each world is designed for it
- Drift/collinearity worlds surface **degraded** trust rather than false recovery passes
- Contract compatibility rollup is consistent across bundles

## What it does not prove

- Causal incrementality or real experiment validity
- Production release readiness
- Threshold-calibrated pass/fail at scale (`TBD_v1`)
- Monte Carlo robustness or large scenario lattices
- VAL-012 on non-drift worlds (deferred / not in scope)

## Usage

```python
from pathlib import Path
from mmm.validation.synthetic.reliability_scorecard import write_reliability_scorecard

path = write_reliability_scorecard(Path("validation/worlds").parents[1])
print(path)
```

## Lattice sweep integration (Phase 5A / 5B)

`build_scorecard_from_reports(..., mode="lattice_structural")` aggregates structural lattice worlds without penalizing deferred VAL-* rows. See [lattice_sweep.md](lattice_sweep.md).

`build_scorecard_from_reports(..., mode="recovery", world_scope_overrides=..., expected_skip_overrides=...)` supports behavioral lattice worlds with per-`world_type` capability scope and expected skips. Behavioral score is computed separately in [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md).

## Evolution

- **Phase 5C** ✅ — [exact recovery investigation](exact_recovery_investigation.md) (INV-056)
- **Phase 5D** ✅ — [reliability threshold governance](reliability_threshold_governance.md); metric-class scores (INV-059)
- **Phase 5E** ✅ — [drift_detection.md](drift_detection.md), [trust_report_semantics.md](trust_report_semantics.md)
- **Phase 5F** — Monte Carlo threshold calibration (INV-060)
- Versioned **approved** thresholds replacing `TBD_v1_runtime` (DR-04)
- Trend comparison across `materialization_version` / package semver
