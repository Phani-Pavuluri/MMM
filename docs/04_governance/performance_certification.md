# Performance certification

Documents **practical operating limits** on synthetic panels — measurement only, no algorithm changes.

## Config

```yaml
extensions:
  performance_certification:
    enabled: false
    include_medium_scenario: false
    include_large_scenario: false
    n_trials_per_scenario: 1
```

## Scenarios

| Name | Geos | Channels | Weeks |
|------|------|----------|-------|
| small | 20 | 15 | 52 |
| medium | 100 | 40 | 104 |
| large | 500 | 100 | 156 |

Medium/large are opt-in (can be slow).

## Artifact

`performance_certification_report` with `runtime_by_stage`, `memory_estimate_kb`, `bottlenecks`, `recommendations`.

## Limitations

- Synthetic DGP only; production panels and IO may differ.
- Does not optimize code paths — identifies bottlenecks for human prioritization.
