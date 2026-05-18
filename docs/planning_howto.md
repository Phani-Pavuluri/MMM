# Planning how-to (media vs non-media)

End-to-end guide for **production decision planning**: full-panel Δμ simulation and media-only budget optimization with an explicit, auditable contract for what changes and what stays fixed.

**Related docs**

| Doc | Use when |
|-----|----------|
| [decision_runbook.md](decision_runbook.md) §2e | Operator rules, prod gates, artifact tiers |
| [config_yaml.md](config_yaml.md) | `extensions.planning_policy` and scenario YAML fields |
| [planning_artifact_schema.md](planning_artifact_schema.md) | JSON / bundle field reference |
| [planning_execution.md](planning_execution.md) | Internal simulate pipeline (developers) |

---

## 1. Mental model

| What you change | `mmm decide simulate` | `mmm decide optimize-budget` |
|-----------------|----------------------|------------------------------|
| **Media spend** | You set candidate (and optional baseline) | **Optimized** by SLSQP on Δμ |
| **Controls** (promo, price, macro, …) | Default: **observed** historical panel per row; optional sparse **overlays** | Default: **observed**; optional **`--scenario`** fixes overlays on **every** optimizer step |
| **Truth surface** | Full-panel `simulate()` (not curves) | Same simulator inside the optimizer |

Every successful decide output includes:

- **`planning_assumptions`** — `controls_assumption`, `media_assumption`, `world_assumption`
- **`scenario_lineage`** — `scenario_id`, `scenario_hash`, overlay SHA-256 (and optional full overlay rows)
- **`control_scenario_policy`** — warnings when sensitive controls use observed values without a scenario

The CLI prints assumptions, disclosures, and **PLANNING POLICY WARNING** lines on stderr when applicable.

---

## 2. Prerequisites

1. **Train** a Ridge+BO model and write an extension report:

   ```bash
   mmm train --config configs/your_train.yaml
   # → artifacts/extension_report.json (path varies by config)
   ```

2. **Prod decision paths** require (see [decision_runbook.md](decision_runbook.md) §2a):

   - `run_environment: prod`
   - Extension report with `ridge_fit_summary.coef`, governance, `model_release.state: planning_allowed`
   - **`--out`** on every `mmm decide …` command
   - Panel path in config matching the trained panel

3. **Python 3.10+** for CLI and API.

---

## 3. Scenario file (`PlanningScenario`)

Use a YAML file with `--scenario` (or pass legacy flat keys as inline JSON for quick tests).

**Minimal media-only** (non-media = observed; may warn if `promo` is in `data.control_columns`):

```yaml
scenario_id: q2_media_lift
media:
  candidate_spend:
    tv: 1200000
    search: 800000
```

**Media + promo overlay** (sparse overrides on existing panel columns):

```yaml
scenario_id: q1_promo_geo_g1
scenario_version: "1"
description: Promo on in week 1 for geo G1; candidate media levels national

media:
  candidate_spend:
    tv: 100000
    search: 50000

controls:
  control_overlay_plan:
    overrides:
      - geo: G1
        week: 1
        column: promo_flag   # must exist on panel + in data.control_columns
        value: 1.0
```

Rules:

- Overlays match **`(geo, week)`** rows in the training panel; unknown geo/week/column **fails**.
- The library does **not** invent future macro/promo calendars — only overwrites values on columns you already modeled.
- Legacy flat keys (`candidate_spend`, `control_overlay_plan`, …) still work and are normalized to `PlanningScenario`.

Example file in-repo: [examples/planning_scenario_promo.yaml](../examples/planning_scenario_promo.yaml).

---

## 4. CLI: simulate a candidate plan

Answer: “What is Δμ if we run this media plan (and optional control overlays)?”

```bash
mmm decide simulate \
  --config configs/prod.yaml \
  --extension-report artifacts/extension_report.json \
  --scenario scenarios/q2_media_lift.yaml \
  --out decisions/sim_q2.json
```

**Inline JSON** (no file):

```bash
mmm decide simulate \
  --config configs/prod.yaml \
  --extension-report artifacts/extension_report.json \
  --scenario '{"candidate_spend": {"tv": 1.2e6, "search": 8e5}}' \
  --out decisions/sim_inline.json
```

**Read the result**

```bash
jq '.simulation.delta_mu, .planning_assumptions, .scenario_lineage.scenario_hash' decisions/sim_q2.json
```

Typical `planning_assumptions`:

- `controls_assumption`: `observed` (no overlay) or `overlay` (sparse overrides)
- `media_assumption`: `constant` (national `candidate_spend`)
- `world_assumption`: `explicit_scenario` when `scenario_id` is set

---

## 5. CLI: optimize media under a fixed non-media world

Answer: “What media allocation maximizes Δμ given budget bounds, holding non-media fixed?”

**Observed controls only** (default):

```bash
mmm decide optimize-budget \
  --config configs/prod.yaml \
  --extension-report artifacts/extension_report.json \
  --out decisions/opt_observed_controls.json
```

Stderr should note: **Non-media: no PlanningScenario overlay supplied; using observed historical panel controls.**

**Fixed promo/pricing via scenario** (overlays applied on every SLSQP evaluation):

```bash
mmm decide optimize-budget \
  --config configs/prod.yaml \
  --extension-report artifacts/extension_report.json \
  --scenario scenarios/q1_promo_geo_g1.yaml \
  --out decisions/opt_promo_fixed.json
```

Typical `planning_assumptions` when a plan overlay is set:

- `controls_assumption`: `frozen_scenario`
- `media_assumption`: `optimized`
- `world_assumption`: `explicit_scenario`

```bash
jq '.planning_assumptions, .scenario_lineage.plan_overlay_spec_sha256' decisions/opt_promo_fixed.json
```

Configure budget bounds in config (`budget.total_budget`, `channel_min` / `channel_max`, geo variants). See [budget_optimization.md](budget_optimization.md) for constraint concepts.

---

## 6. Policy warnings and strict prod

List sensitive columns in config:

```yaml
extensions:
  planning_policy:
    promo_columns: [promo_flag, discount_depth]
    pricing_columns: [price_index]
    name_heuristic_warnings: true
    strict_prod_requires_explicit_control_scenario: false
```

When `controls_assumption=observed` and a sensitive column is on the panel, the CLI emits:

```text
PLANNING POLICY WARNING: Sensitive control columns ['promo_flag'] use observed historical panel values; ...
```

To **block** instead of warn in prod:

```yaml
strict_prod_requires_explicit_control_scenario: true
```

Then supply a scenario with `control_overlay_plan` (or baseline + plan overlays).

---

## 7. Python API

Same policy gates as the CLI (`simulate_decision` / `optimize_budget_decision` via `mmm.decision.api`):

```python
from pathlib import Path

from mmm.decision.api import run_decision_optimization, run_decision_simulation

# Simulate
sim_payload = run_decision_simulation(
    config=Path("configs/prod.yaml"),
    scenario=Path("scenarios/q2_media_lift.yaml"),
    extension_report=Path("artifacts/extension_report.json"),
    out=Path("decisions/sim_q2.json"),
)
print(sim_payload["planning_assumptions"])
print(sim_payload.get("scenario_lineage", {}).get("scenario_hash"))

# Optimize (media only; optional fixed non-media scenario)
opt_payload = run_decision_optimization(
    config=Path("configs/prod.yaml"),
    extension_report=Path("artifacts/extension_report.json"),
    scenario=Path("scenarios/q1_promo_geo_g1.yaml"),  # omit for observed controls only
    out=Path("decisions/opt_promo_fixed.json"),
)
bundle = opt_payload.get("decision_bundle") or {}
print(bundle.get("planning_assumptions"))
print(bundle.get("scenario_lineage"))
print(opt_payload.get("control_scenario_policy"))
```

Build or validate scenarios in code:

```python
from mmm.planning.scenario import planning_scenario_from_yaml, planning_scenario_from_dict

ps = planning_scenario_from_yaml("scenarios/q1_promo_geo_g1.yaml")
print(ps.scenario_id, ps.scenario_hash())
lineage = ps.lineage_payload()  # hashes; pass store_full_overlays=True if config allows
```

---

## 8. What not to use for planning

| Tool | Role |
|------|------|
| **`mmm simulate-diagnostic-curves`** | Media-only curve diagnostics — **not** full-panel Δμ, **not** non-media scenarios |
| **`mmm simulate` / `mmm optimize-budget`** (top-level) | Deprecated shims → use **`mmm decide …`** |
| **Decomposition / curve ROI** | Approximate / local — not audited planning truth (runbook §2e) |

---

## 9. Checklist before sharing a decision

- [ ] `model_release.state` is `planning_allowed`
- [ ] `planning_assumptions` matches what you intended (observed vs overlay vs frozen)
- [ ] For promo/pricing stories: scenario file attached and `scenario_hash` recorded
- [ ] Read `control_scenario_policy` and `scenario_validation_warnings` (if any)
- [ ] `decision_bundle.artifact_tier` is `decision` and `decision_safe` is understood in context of gates
