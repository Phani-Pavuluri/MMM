# Production safety checklist

Use this before enabling automated spend recommendations or publishing model-based targets.

## Data & fit

- [ ] Panel fingerprint matches training artifact (or documented exception).
- [ ] `run_environment: prod` set explicitly in config.
- [ ] **`model_form: semi_log`** for Ridge prod training and decisions (`log_log` is research-only until formally validated).
- [ ] Extension report / decision bundle do not indicate stale `model_form=log_log` lineage.
- [ ] No silent fallback from full-model to curve-only optimizer in prod.

## Governance

- [ ] `extension_report` governance / response diagnostics meet org thresholds (see `OptimizationSafetyGate`).
- [ ] Non-BAU baselines documented in optimization disclosure strings.

## Uncertainty propagation (PR 5A)

- [ ] Treat `uncertainty_propagation_report` as **research/diagnostic** only (`prod_decisioning_allowed: false`, `prod_monetary_ci_allowed: false`).
- [ ] Do not use source `magnitude_proxy` fields for prod budget decisions.

## Robust optimization research (PR 5B)

- [ ] Treat `robust_optimization_research` as **research/diagnostic** only — not a replacement for `mmm decide optimize-budget`.
- [ ] Confirm `recommended_prod_allocation` is null and `decision_safe` is false on the artifact.
- [ ] Do not publish robust frontier winners as approved prod spend without a separate validation phase.

## Continuous validation (diagnostic)

- [ ] Treat `continuous_validation_report` as **diagnostic only** — confirm `auto_retrain`, `auto_registry_promotion`, and `auto_budget_change` are false.
- [ ] Do not trigger retraining or registry promotion from `recommended_action` without human review.
- [ ] `not_evaluable` rows (missing SE, missing prior prediction) are expected when registries are incomplete.

## Decision validation (diagnostic)

- [ ] Treat `decision_validation_report` as **diagnostic only** — confirm `decision_safe` is false and `auto_optimizer_change` / `auto_budget_change` are false.
- [ ] Do not treat observational evidence as experiment validation (`not_evaluable_reason` observational).
- [ ] Do not claim causal proof of recommendation quality from this report alone.

## Uncertainty & posterior planning

- [ ] If using **P10/P50/P90** or **risk-aware** optimization: `posterior_diagnostics_ok` and `posterior_predictive_ok` are true in `bayesian_fit_meta`.
- [ ] `extensions.product.posterior_planning_mode: draws` in prod whenever `linear_coef_draws` are consumed.
- [ ] Coef draws are generated from a documented process (e.g. Bayesian export or approved bootstrap), not ad-hoc noise.

## Experiments & replay

- [ ] `experiment_id` is UUID and immutable in downstream stores.
- [ ] `experiment_readiness` returns `ready: true` before replay (approved, signed, calibration ref present) — **enforced in code** when `calibration.require_approved_experiment_registry` is true.
- [ ] Prod decide paths: `extension_report.calibration_summary.replay_calibration_active=true` **or** `experiment_matching.n_matched >= 1` (not `skipped`).
- [ ] Calibration artifact version pinned and traceable in metadata.
- [ ] **Legacy replay:** `replay_mode: legacy` + `replay_units_path` passes `assert_replay_production_ready` on units.
- [ ] **Evidence-registry replay:** `replay_mode: evidence_registry` + `extension_report.evidence_weighted_replay_summary` passes prod evidence gate (`n_evidence_units_used >= 1`, high/medium quality, no false subgeo claims).
- [ ] **Full-panel replay transform:** `replay_transform_mode` is `full_panel_transform_estimand_mask` (or `replay_uses_full_panel_transform: true`) on replay units and extension `calibration_summary` / `ridge_fit_summary` — not window-slice-only semantics.
- [ ] **Replay holdout disclosed:** `replay_holdout_available` is `true` when CV ran, or absence of `replay_holdout_loss` / gap fields is explicitly accepted (do not treat missing holdout as “no overfit”).
- [ ] **Severe replay generalization gap reviewed:** when `replay_generalization_gap_severity` is `severe`, read `replay_overfit_warning` and compare to predictive CV score before release.
- [ ] **`block_on_severe_replay_gap`:** enabled only when the org wants hard fail on severe gap (default `false` = advisory warning only).
- [ ] **Legacy replay deprecation:** review `legacy_replay_warnings` / `legacy_replay_upgrade_warnings` for `legacy_replay_deprecated_use_evidence_registry`; migrate new work to evidence-registry replay.
- [ ] Aggregate national/user experiments do not claim DMA lift; allocated shocks documented as `computational_bridge_only` only.
- [ ] **Bayesian experiment likelihood:** if `bayesian.use_experiment_likelihood`, treat as **research-only**; confirm `bayesian_experiment_likelihood_report.prod_decisioning_allowed` is false and prod decision APIs remain blocked.
- [ ] **Bayesian hierarchy:** if `bayesian.use_hierarchy`, treat as **research-only**; confirm `bayesian_hierarchy_report.prod_decisioning_allowed` is false. Do not use for prod budget optimization.

## Artifact tiers

- [ ] Training `extension_report` / nested bundle: **research** tier (diagnostics only).
- [ ] Budget decisions: fresh `mmm decide` JSON with `artifact_tier=decision` (see `extension_report.artifact_tier_disclosure` after train).

## Panel QA (training)

- [ ] In prod, `extensions.panel_qa.prod_block_severity` defaults to **block** unless `prod_block_waiver: true`.

## Economics metadata

- [ ] Surfaces exposed to business users carry a complete `economics_output_metadata` block: `economics_version`, `economics_contract_version`, `surface`, `uncertainty_mode`, `computation_mode` (`exact` / `approximate` / `unknown`), `baseline_type` (not `unspecified` for simulation paths), `decision_safe` (bool for full-model simulation), and KPI column fields.
- [ ] Call `validate_business_economics_metadata` (or rely on `build_decision_bundle` / extension runner) so missing keys fail closed in CI or release gates.
