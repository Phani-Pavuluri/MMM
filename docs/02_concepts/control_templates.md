# Control templates

The **control template** helper generates starter CSV scaffolds for analysts who are unsure which **non-media controls** to include in a geo-level MMM panel. It is an **onboarding and governance utility only**.

It does **not**:

- Insert columns into training automatically
- Modify `data.control_columns` in YAML
- Change feature engineering, optimization, or planning
- Download macro series from external APIs

## Why controls matter

Media channels explain part of weekly (or monthly) KPI movement, but **baseline and contextual drivers** also move outcomes: holidays, promotions, pricing, outages, macro shocks, weather, and competitive activity.

Controls help the model **attribute** variation that is not media, so media coefficients and budget simulations are less confounded.

## Controls vs causal drivers

| | Controls (in MMM panel) | Causal drivers (experiments) |
|--|-------------------------|----------------------------|
| Role | Adjust for observed confounders in observational data | Identify incremental lift under designed variation |
| Source | Historical time series aligned to geo-week rows | Geo tests, holdouts, incrementality studies |
| This helper | Illustrates **column names and scales** only | Not a substitute for experiment design |

Example values in templates are **synthetic illustrations**. They are **not** priors, defaults, or recommended coefficient signs.

## CLI

```bash
mmm generate-control-template --domain b2b --frequency weekly --rows 52 --out controls.csv
```

| Flag | Values |
|------|--------|
| `--domain` | `generic`, `b2b`, `ecommerce`, `retail`, `saas`, `travel` |
| `--frequency` | `weekly` (`week` column), `monthly` (`month` column) |
| `--rows` | Number of periods (default 52) |
| `--out` | CSV path (default `control_template.csv`) |

Writes:

- **CSV** — period column + domain-specific control columns with illustrative values
- **`{stem}_template.meta.json`** — governance metadata (sidecar, not embedded in CSV)

Metadata flags (always):

- `template_only: true`
- `illustrative_values_only: true`
- `modeling_default: false`
- `replace_with_real_historical_data: true`
- `diagnostic_only: true`
- `auto_model_integration_forbidden: true`

## Common controls by industry

| Domain | Example columns (after `week` / `month`) |
|--------|------------------------------------------|
| **generic** | `holiday_indicator`, `promo_intensity`, `pricing_index`, `product_launch`, `site_outage` |
| **b2b** | + `sp500_index`, `unemployment_rate` |
| **ecommerce** | `promo_intensity`, `discount_rate`, `site_traffic_index`, `cart_abandonment_rate`, `competitor_price_index` |
| **retail** | `discount_rate`, `inventory_level`, `weather_index`, `competitor_promo` |
| **saas** | `trial_volume_index`, `churn_risk_index`, `product_launch` |
| **travel** | `weather_index`, `fuel_price`, `consumer_sentiment` |

Merge real historical values, align to your panel’s `geo` × `week` grain, then list chosen columns under `data.control_columns` in YAML. See [../01_getting_started/config_yaml.md](../01_getting_started/config_yaml.md).

## Warning: post-treatment controls

Do **not** include controls that are **caused by** media or that only exist **after** the media exposure you want to measure (for example, downstream conversions used as controls for upper-funnel spend). That introduces **post-treatment bias** and weakens or flips media effects.

Prefer:

- Pre-determined promos and price lists committed before the flight
- Exogenous macro and weather series
- Known operational shocks (outages) with clear timing

When in doubt, treat a column as a **scenario overlay** on decision paths ([planning_howto.md](../03_planning/planning_howto.md)) rather than a passive control.

## Programmatic use

```python
from pathlib import Path
from mmm.helpers.control_templates import generate_control_template

generate_control_template(domain="retail", frequency="weekly", n_rows=104, out=Path("controls.csv"))
```

## Related docs

- [config_yaml.md](../01_getting_started/config_yaml.md) — `data.control_columns` and planning policy
- [decision_vs_research.md](decision_vs_research.md) — decision vs diagnostic surfaces
- [../04_governance/prod_safety_checklist.md](../04_governance/prod_safety_checklist.md) — production controls discipline
