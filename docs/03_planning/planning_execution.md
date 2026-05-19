# Planning execution

How full-panel Δμ simulation and optimization run internally, and the current media vs non-media contract.

**Operator how-to:** [planning_howto.md](planning_howto.md). **Artifact fields:** [../04_governance/artifact_schema.md](../04_governance/artifact_schema.md).

## Simulation pipeline

```
training panel (sorted geo × week)
    ↓
counterfactual panel copy
    ↓
media spend overwrite (constant | per-geo | piecewise path)
    ↓
optional ControlOverlaySpec (sparse geo×week×column values)
    ↓
build_design_matrix (adstock + saturation + control_columns from panel)
    ↓
predict_ridge (fixed coef from ridge_fit_summary)
    ↓
aggregate mean μ → Δμ = μ(plan) − μ(baseline)
```

**Optimize path:** SLSQP varies media spend vector; each evaluation calls the same `simulate()` with optional fixed `OptimizeNonMediaContext` overlays.

## Media vs non-media contract

| Surface | Media | Non-media (controls) |
|---------|-------|----------------------|
| **`mmm decide simulate`** | User-specified spend (national / geo / path) | Default **observed** panel; optional sparse overlays via scenario YAML |
| **`mmm decide optimize-budget`** | **Optimized** (SLSQP on Δμ) | Default **observed**; optional **`--scenario`** fixes overlays on **every** optimizer evaluation |
| **Curve diagnostics** | Local spend proxies | **Not** full-panel non-media simulation |

## Planning assumptions (semantic contract)

Prod decision bundles require `planning_assumptions` with allowed enum literals and valid combinations with `scenario_lineage`. Unknown typos and invalid combinations fail closed in prod.

Key enums: `controls_assumption` (`observed` | `overlay` | `frozen_scenario`), `media_assumption` (`constant` | `geo_channel` | `piecewise_path` | `optimized`), `world_assumption` (`historical_panel` | `explicit_scenario`; `multi_world` reserved and **rejected in prod** today).

Full field tables and validation rules: [../04_governance/artifact_schema.md](../04_governance/artifact_schema.md).

## Config (`extensions.planning_policy`)

- `promo_columns`, `pricing_columns`, `macro_columns`, `seasonality_columns`
- `name_heuristic_warnings` (default true)
- `strict_prod_requires_explicit_control_scenario` (default false)
- `store_full_control_overlays_in_artifacts` (default false)

See [../01_getting_started/config_yaml.md](../01_getting_started/config_yaml.md).

## Not supported today

- Multi-world / weighted `scenarios` on `PlanningScenario`
- Macro/promo **generation** (only sparse overlays on existing panel columns)
- Optimizing non-media controls jointly with media
- Bayesian prod optimize path on the same contract as Ridge full-panel Δμ
