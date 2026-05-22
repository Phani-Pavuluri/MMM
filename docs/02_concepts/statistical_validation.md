# Statistical validation posture

This document states what automated tests and nightly CI **do** and **do not** prove.

## Replay holdout (optional)

When `calibration.use_replay_holdout_split=true`, Ridge BO uses **train** replay units in the hyperparameter objective only. **Holdout** replay units appear in `replay_holdout_validation` (train/holdout loss, unit counts, `sensitivity_warning`). This reduces calibration overfit risk during search but is **diagnostic** until holdout is required by policy.

Default behavior is unchanged when the flag is false. Prod replay evidence gates (`replay_calibration_active`) still apply.

## CV and adstock

Ridge BO builds media features on the full sorted panel (causal adstock within geo), then fits ridge on **training** rows per CV fold only (`apply_masks_for_fit`). Validation rows are scored but not included in the fold fit loss. See [cv_modes.md](cv_modes.md).

## Synthetic DGP tests

Tests under `tests/test_synthetic_dgp_recovery.py` and related modules check implementation sanity under controlled data:

- Semi-log coefficient sign / approximate scale (noiseless)
- Δμ direction under spend shifts
- Closed-form geometric adstock and Hill saturation
- Collinear channels → identifiability / separability signals
- Placebo null → small media coefficients

**Passing these tests does not prove causal validity, incrementality, or correct attribution on real client data.**

## Baseline beat waiver

`extensions.governance.require_beats_baselines_for_approval` defaults to **true**. Setting it to **false** waives the baseline beat requirement for scorecard approval and emits explicit warnings in governance JSON, `governance_summary`, model card, and prod CLI stderr. This is for exceptional/fixture use only—not a silent prod default.

## Nightly CI

The nightly workflow runs slow, Bayesian, Optuna/BO extended, uncertainty research, and large-panel tests. It improves **release confidence** and catches regressions; it is not a substitute for experiment design, holdout campaigns, or independent causal evidence.
