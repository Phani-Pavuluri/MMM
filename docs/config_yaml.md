# YAML configuration

Canonical keys mirror `MMMConfig` in `mmm/config/schema.py`. Important sections:

- `framework`: `ridge_bo` | `bayesian`
- `model_form`: `semi_log` (default) | `log_log`
- `pooling`: `none` | `full` | `partial`
- `data`: paths, column names, channel list
- `transforms`: `adstock` (`geometric`|`weibull`), `saturation` (`hill`|`log`|`logistic`), optional param dicts
- `cv`: `mode` (`auto`|`rolling`|`expanding`), `n_splits`, `min_train_weeks`, `horizon_weeks`, `gap_weeks`
- `ridge_bo` / `bayesian`: backend-specific knobs
- `objective`: composite weights for Ridge+BO
- `calibration`: optional experiments path + match levels
- `artifacts`: `local` store root or `mlflow` experiment name

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
- **`store_full_control_overlays_in_artifacts`**: when `true`, embed canonical overlay rows in `scenario_lineage` (default hashes only). See [planning_artifact_schema.md](planning_artifact_schema.md).

### PlanningScenario YAML (`mmm decide` `--scenario`)

Typed scenario for simulate / optimize (see `docs/decision_runbook.md` §2e):

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
