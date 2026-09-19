# Monte Carlo Tier-1 recommendations (report-only)

**Status:** recommendations only — not approved for production gates. Thresholds remain provisional pending DR-04.

Batch: `tier1_batch_runner_v1.0.0` · tier `tier_1_calibration` · N=108 worlds · generated 2026-09-19T10:50:58.040919+00:00.

## Observed capability summary

| Capability | Class | n scored | Pass rate |
|------------|-------|----------|-----------|
| structural_integrity | structural | 108 | 0.759 |
| platform_contract_compatibility | structural | 108 | 0.88 |
| coefficient_recovery | diagnostic_attribution | 13 | 0.0 |
| transform_recovery | diagnostic_attribution | 13 | 0.0 |
| delta_mu_recovery | decision_grade | 13 | 1.0 |
| optimizer_recovery | decision_grade | 5 | 0.6 |
| replay_recovery | decision_grade | 5 | 1.0 |
| drift_recovery | trust_modifier | 3 | 1.0 |
| identifiability_recovery | trust_modifier | 2 | 1.0 |

## Threshold recommendations (report-only)

| Metric | VAL | Class | n | Pass rate | Suggested action | Confidence |
|--------|-----|-------|---|-----------|------------------|------------|
| VAL-001_coef | VAL-001 | diagnostic_attribution | 13 | 0.0 | report_only; must not gate releases | low |
| VAL-002_003_transform | VAL-002 | diagnostic_attribution | 13 | 0.0 | report_only; must not gate releases | low |
| VAL-004_delta_mu | VAL-004 | decision_grade | 13 | 1.0 | retain_provisional; needs DR-04 review | low |
| VAL-005_optimizer | VAL-005 | decision_grade | 5 | 0.6 | retain_provisional; needs DR-04 review | low |
| VAL-006_replay | VAL-006 | decision_grade | 5 | 1.0 | retain_provisional; needs DR-04 review | low |
| VAL-012_drift | VAL-012 | trust_modifier | 3 | 1.0 | calibrate severity bands from tier-1 distribution; DR-04 decides | low |

## Disclaimer

Report-only. No threshold is approved; thresholds remain provisional pending DR-04. No release-gate change.

## Limitations

- Small-panel deterministic worlds; not production-panel geometry
- Smoke stratum is structural-only by design (recovery honestly unscored)
- Behavioral stratum inherits WORLD-008-template DGP assumptions
- Anchors re-certified read-only from committed fixtures
