# Mini audit — Bayes-H3 research sandbox MVP fit

**Date:** 2026-06-01  
**Scope:** `mmm.research.bayes_h3_sandbox` MVP hierarchical fit (research only)  
**Verdict:** **Pass for sandbox MVP** — diagnostic fit runs under fences; **not** production promotion

## Evidence

| Check | Status |
|-------|--------|
| Single entrypoint `run_sandbox_fit` | ✅ |
| H2d-aligned partial pooling prototype (`model.py`) | ✅ |
| Research-only labels on all artifacts | ✅ |
| Diagnostic outputs only (posterior / pooling / convergence) | ✅ |
| No production DecisionSurface / optimizer / recommendations | ✅ |
| Negative guardrail tests | ✅ |
| Fast CI smoke (wrap + fixture) | ✅ |
| PyMC sampling test (`@pytest.mark.slow`) | ✅ when `mmm[bayesian]` installed |

## Blocked (unchanged)

- Production Bayesian decisioning
- `approved_for_prod` / `prod_decisioning_allowed`
- Bayes-H4 recovery worlds (next validation phase)

**Next:** Bayes-H4 recovery worlds — scientific behavior under known truth.
