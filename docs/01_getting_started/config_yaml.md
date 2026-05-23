# YAML configuration

Canonical keys mirror `MMMConfig` in `mmm/config/schema.py`. Important sections:

- `framework`: `ridge_bo` | `bayesian`
- `model_form`: `semi_log` (default, **prod canonical**) | `log_log` (**research-only**; forbidden when `run_environment=prod`)
- `pooling`: `none` | `full` | `partial`
- `data`: paths, column names, channel list (see [../02_concepts/control_templates.md](../02_concepts/control_templates.md) for illustrative control CSV scaffolds)
- `transforms`: production Ridge+BO / decide paths accept **`adstock: geometric`** and **`saturation: hill`** only (enforced by `canonical_transforms` + config validators). Other kinds (`weibull`, `log`, `logistic`) are registry stubs — not supported for training or full-panel simulation without a validated implementation.
- `cv`: `mode` (`auto`|`rolling`|`expanding`), `n_splits`, `min_train_weeks`, `horizon_weeks`, `gap_weeks`
- `ridge_bo` / `bayesian`: backend-specific knobs
- `objective`: composite weights for Ridge+BO
- `calibration`: experiments path, replay units, or evidence registry (see below)
- `artifacts`: `local` store root or `mlflow` experiment name

### Calibration / replay (`calibration:`)

**Legacy (default)** — unchanged prod path:

```yaml
calibration:
  use_replay_calibration: true
  replay_mode: legacy
  replay_units_path: path/to/replay_units.json
```

**Evidence-registry weighted replay (opt-in, Ridge only):**

```yaml
calibration:
  use_replay_calibration: true
  replay_mode: evidence_registry
  evidence_weighting_enabled: true
  compatibility_resolver_enabled: true
  evidence_registry_path: path/to/evidence.json
  model_geo_granularity: dma   # national | region | dma | geo | user
  channel_mapping: { platform_tv: tv }
```

`replay_mode: evidence_registry` requires `evidence_registry_path` and `compatibility_resolver_enabled: true`. Missing registry path or disabled resolver fails at config load.

Weighted BO loss: `sum(w_i * ((mmm_lift_i - lift_i)/se_i)^2) / sum(w_i)`. See [../02_concepts/experiment_evidence.md](../02_concepts/experiment_evidence.md).

**Prod gate:** evidence-registry replay requires `evidence_weighted_replay_summary` with `n_evidence_units_used >= 1`, acceptable quality tiers, and governance fields (`supports_subgeo_claims`, `allocation_role`). Optional `allow_missing_se_in_prod_evidence_replay: false` (default).

**Replay transform (legacy + evidence-registry):**

Both paths build **full-panel** observed/counterfactual spend frames, apply transforms on the sorted panel (preserving pre-window adstock), and evaluate implied lift only on the experiment **estimand mask**. Serialized units should set:

```yaml
replay_estimand:
  replay_transform_mode: full_panel_transform_estimand_mask
```

Window-slice frames in `replay_units.json` are upgraded at train time when `replay_estimand` is present. Units **without** `replay_estimand` emit warning `legacy_replay_deprecated_use_evidence_registry` and are skipped from replay loss (prefer evidence-registry replay for new work).

**BO replay generalization disclosure (advisory; objective unchanged):**

Ridge+BO trials record **train** replay loss (full-panel refit coef — this is what enters the BO objective) vs **holdout** replay loss (last time-series CV-fold coef, diagnostic only). The gap is **not** causal evidence; it flags possible optimism when replay fits better than out-of-fold prediction.

| Key | Default | Meaning |
|-----|---------|---------|
| `replay_generalization_gap_threshold` | `0.25` | `replay_generalization_gap_severity` becomes `severe` when `holdout_loss − train_loss` ≥ this value |
| `block_on_severe_replay_gap` | `false` | **`false` (default):** emit `replay_overfit_warning` only; `model_release` unchanged. **`true`:** severe gap adds `severe_replay_generalization_gap` to release invalidation (opt-in hard fail) |

```yaml
calibration:
  replay_generalization_gap_threshold: 0.25   # severe severity cutoff (moderate band: 0.1–threshold)
  block_on_severe_replay_gap: false           # default: warning only; true = opt-in hard fail on severe gap
```

Severity bands on `replay_generalization_gap` (`holdout_loss − train_loss`): `none` &lt; 0.1, `moderate` 0.1–threshold, `severe` ≥ threshold. When CV does not run or produces no folds, `replay_holdout_available: false` and gap fields are absent — **absence must be reviewed**, not treated as “no overfit.”

**Holdout replay (built-in):** Holdout replay uses the **last CV-fold** coefficients on the **same** replay units as train replay (diagnostic gap only). Optional **unit-list split** for BO train replay: `use_replay_holdout_split`, `replay_holdout_fraction`, `train_replay_units_path`, `holdout_replay_units_path` (see `CalibrationConfig`). Unit lists otherwise come from `replay_units_path` (legacy) or `evidence_registry_path` (evidence-registry).

Replay calibration does **not** prove causal validity — it checks internal consistency under stated estimands. See [../02_concepts/calibration.md](../02_concepts/calibration.md) and [../04_governance/artifact_schema.md](../04_governance/artifact_schema.md#extension-report-calibration--replay-disclosure).

**Replay refit mode (BO objective honesty):**

| Key | Default | Meaning |
|-----|---------|---------|
| `replay_refit_mode` | `full_panel_refit` | `full_panel_refit` (backward compatible), `fold_aligned`, or `holdout_only_diagnostic` |
| | | `full_panel_refit` emits optimism warning; `fold_aligned` fits train folds only; `holdout_only_diagnostic` excludes replay from BO objective |
| `full_panel_replay_refit_prod_waiver_path` | — | **Prod only:** required when `replay_refit_mode=full_panel_refit` and replay calibration is enabled. Prefer `fold_aligned` for production training. |

**Prod train ↔ decide fingerprint (fail-closed):**

```yaml
governance:
  allow_decision_fingerprint_mismatch: false
  decision_fingerprint_mismatch_waiver_path: null
```

Prod `mmm decide simulate|optimize-budget` compares `extension_report.data_fingerprint.sha256_combined` to the live panel fingerprint. Legacy artifacts fall back to `sha256_panel_keycols_sorted_csv` with warnings. Overrides require `allow_decision_fingerprint_mismatch: true` and a signed waiver JSON.

**Promotion workflow (optional prod gate):**

```yaml
governance:
  require_promoted_model_for_prod_decision: false
  promotion_registry_path: path/to/promotions.jsonl
```

See [../04_governance/promotion_workflow.md](../04_governance/promotion_workflow.md).

**Calibration freshness / drift readiness (warnings by default):**

```yaml
governance:
  calibration_max_age_days: 180
  coefficient_shift_threshold: 0.30
  replay_miss_threshold: 0.25
  require_review_on_drift: false   # true → blocks planning_allowed when drift thresholds exceeded
```

See [../04_governance/calibration_freshness.md](../04_governance/calibration_freshness.md).

**Operational trust extensions (diagnostic):**

```yaml
extensions:
  reproducibility_certification:
    enabled: false
  performance_certification:
    enabled: false
    include_medium_scenario: false
    include_large_scenario: false
```

**Bayesian experiment likelihood (research-only):**

```yaml
bayesian:
  use_experiment_likelihood: true
  experiment_registry_path: path/to/evidence.json
  experiment_likelihood_weight: 1.0
  exp_likelihood_research_only: true   # required; does not enable prod decisioning
```

Every training run should persist `resolved_config.yaml` next to metrics and diagnostics.

## Extensions (`extensions:`)

Optional block on `MMMConfig` for identifiability, governance scorecard, optimization gates, falsification, feature-engine preview (trend/Fourier/holiday flags), estimand alignment, and **planning policy**. Defaults are non-breaking. Training writes `extension_report.json` via the artifact store when `MMMTrainer.run()` completes.

### Planning policy (`extensions.planning_policy`)

Controls how **decision** paths (`mmm decide simulate`, `mmm decide optimize-budget`) treat non-media columns:

```yaml
extensions:
  planning_policy:
    promo_columns: [promo_flag, discount_depth]
    pricing_columns: [price_index]
    macro_columns: [gdp_growth]
    seasonality_columns: [holiday_week]
    strict_prod_requires_explicit_control_scenario: false
    name_heuristic_warnings: true
    store_full_control_overlays_in_artifacts: false
```

- **`promo_columns` / `pricing_columns` / `macro_columns` / `seasonality_columns`**: explicit lists; when present in `data.control_columns` and `controls_assumption=observed`, emits **warnings** (or **blocks** in prod if `strict_prod_requires_explicit_control_scenario: true`).
- **`name_heuristic_warnings`**: optional substring heuristics on control column names (warning-only unless strict prod is enabled).
- **`store_full_control_overlays_in_artifacts`**: when `true`, embed canonical overlay rows in `scenario_lineage` (default hashes only). See [../04_governance/artifact_schema.md](../04_governance/artifact_schema.md).

### Feature separability (`extensions.feature_separability`)

Diagnostic-only guidance when related channel columns (for example `Meta_prospecting` / `Meta_retargeting`) may not be reliably separable. **Does not merge columns or change training.**

```yaml
extensions:
  feature_separability:
    enabled: true
    auto_group_prefix: true
    feature_groups: {}   # optional explicit groups; overrides auto when non-empty
```

Output: `extension_report.feature_separability_report`. See [../02_concepts/feature_separability.md](../02_concepts/feature_separability.md).

### Experiment scheduler (`extensions.experiment_scheduler`)

Prioritizes **where to run geo/incrementality experiments** from post-fit diagnostics (no test design or execution). Requires `feature_separability_report`.

```yaml
extensions:
  experiment_scheduler:
    enabled: true
```

Output: `extension_report.experiment_scheduler_report`. See [../02_concepts/experiment_scheduler.md](../02_concepts/experiment_scheduler.md).

### Continuous / decision validation (diagnostic, default off)

```yaml
extensions:
  continuous_validation:
    enabled: false
    registry_dir: path/to/accepted_runs/
    lookback_days: 365
    require_experiment_se: false
  decision_validation:
    enabled: false
    decision_registry_dir: path/to/decisions/
    lookback_days: 180
```

See [../02_concepts/continuous_validation.md](../02_concepts/continuous_validation.md) and [../02_concepts/decision_validation.md](../02_concepts/decision_validation.md). Reports are omitted from `extension_report` unless `enabled: true`.

### PlanningScenario YAML (`mmm decide` `--scenario`)

Typed scenario for simulate / optimize. **Full walkthrough:** [../03_planning/planning_howto.md](../03_planning/planning_howto.md). Contract summary: [../03_planning/decision_runbook.md](../03_planning/decision_runbook.md) §2e.

```yaml
scenario_id: q1_promo_lift
scenario_version: "1"
description: Heavy promo in week 1 for geo G1
media:
  candidate_spend:
    search: 120000
    social: 80000
controls:
  control_overlay_plan:
    overrides:
      - geo: G1
        week: 1
        column: promo_flag
        value: 1.0
```

Legacy flat keys (`candidate_spend`, `control_overlay_plan`, …) remain accepted and are normalized to `PlanningScenario`.
