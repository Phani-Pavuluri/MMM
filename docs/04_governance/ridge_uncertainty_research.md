# Ridge uncertainty research findings

**Status:** research / diagnostic only. Production decision surfaces remain **point-estimate** with explicit disclosure.

## Methods investigated

| Method | Production | Notes |
|--------|------------|-------|
| Bootstrap intervals | Blocked | Row-resample refit on holdout; coverage reported in `ridge_uncertainty_research` artifact |
| Conformal intervals | Blocked | Split-conformal residual quantiles; not calibrated for monetary decisioning |
| Coverage diagnostics | Research | Empirical coverage on synthetic and holdout slices |

## Guardrails

- `decision_uncertainty.uncertainty_available` stays `false` for Ridge production.
- `ridge_uncertainty_research.production_intervals_allowed` is always `false`.
- Enable research extension: `extensions.ridge_uncertainty_research.enabled: true` (off by default).

## Failure modes

- Adstock/carryover breaks exchangeability for naive row bootstrap.
- Structural breaks violate conformal stationarity.
- Small panels produce unstable coverage estimates.

See implementation: `mmm/governance/ridge_uncertainty_research.py`.
