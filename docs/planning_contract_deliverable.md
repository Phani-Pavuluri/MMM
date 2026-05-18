# Planning contract deliverable (changelog)

Grouped summary of the explicit media vs non-media planning contract.

## Contract (operator view)

| Surface | Media | Non-media (controls) |
|---------|-------|----------------------|
| **`mmm decide simulate`** | User-specified spend (national / geo / path) | Default **observed** panel; optional sparse overlays via scenario YAML |
| **`mmm decide optimize-budget`** | **Optimized** (SLSQP on Δμ) | Default **observed**; optional **`--scenario`** fixes overlays on **every** optimizer evaluation |
| **Curve diagnostics** | Local spend proxies | **Not** full-panel non-media simulation |

## Code modules

| Module | Role |
|--------|------|
| `mmm/planning/assumptions.py` | `planning_assumptions` builder + disclosures |
| `mmm/planning/scenario.py` | `PlanningScenario`, lineage, overlay SHA-256 |
| `mmm/planning/policy.py` | Sensitive-column warnings / strict prod block |
| `mmm/planning/optimize_context.py` | Fixed non-media context for optimizer |
| `mmm/planning/cli_display.py` | CLI stderr summaries |
| `mmm/decision/service.py` | Wires simulate / optimize + bundles |
| `mmm/optimization/budget/simulation_optimizer.py` | Passes overlays into each `simulate()` |

## Artifact fields (new / required in prod)

- `planning_assumptions`
- `scenario_lineage` (with overlay hashes; optional full overlay rows)
- `control_scenario_policy`

## Config (`extensions.planning_policy`)

- `promo_columns`, `pricing_columns`, `macro_columns`, `seasonality_columns`
- `name_heuristic_warnings` (default true)
- `strict_prod_requires_explicit_control_scenario` (default false)
- `store_full_control_overlays_in_artifacts` (default false)

## Deferred (out of scope)

- Multi-world / weighted `scenarios` on `PlanningScenario`
- Macro/promo **generation** (only sparse overlays on existing panel columns)
- Bayesian prod optimize path

## Tests

`tests/test_planning_scenario_contract.py` — assumptions, overlay hashes, policy warning, optimizer overlay parity, bundle validation, CLI help.
