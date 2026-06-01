# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-05-22

First production-ready release. This version **freezes the governed Ridge semi-log full-panel Δμ decision path** for budget and scenario decisions. It does **not** certify all research or diagnostic features for production use.

**Full release notes:** [docs/04_governance/v1_release_notes.md](docs/04_governance/v1_release_notes.md)

### Summary

- Production budget and scenario decisions must use **`mmm decide simulate`** and **`mmm decide optimize-budget`** on a trained Ridge semi-log model with geometric adstock and Hill saturation.
- **Full-panel Δμ** on the trained model is the decision estimand; response curves, decomposition, robust optimization research, uncertainty propagation, Bayesian outputs, and train-time stress diagnostics are **not** budget truth.
- Fail-closed governance, artifact tier validation, fingerprint matching, and optional strict production certification gates are in place for prod environments.

### Production-safe scope

- Ridge + Bayesian optimization (Ridge BO)
- `semi_log` model form only in prod decision paths
- Geometric adstock + Hill saturation (canonical prod transform stack)
- Full-panel Δμ simulation and optimization
- Production artifact validation and decision bundles
- Train→decide fingerprint matching
- Replay calibration governance and prod gates
- Optimizer, synthetic, reproducibility, and performance certification (diagnostic rollups)
- Promotion workflow and `decision_trace.json`

### Not production-supported

- Bayesian production decisioning (`mmm decide optimize-budget` blocked in prod)
- `log_log` production decisioning
- Weibull / logistic / log transform stacks as prod decision paths
- Robust optimization as a production allocator
- Causal lift guarantees from observational MMM alone
- Auto-retraining, auto-promotion, agentic orchestration
- State-space / time-varying coefficient production paths

### Safety (high level)

- Prod `log_log` and Bayesian optimize blocked
- Unsupported transforms blocked in prod
- Stale or incomplete decision artifacts rejected
- Panel fingerprint mismatch rejected unless waived
- Failed production readiness emits a **severe warning** on prod decide; `governance.require_production_certification: true` fails closed

### Migration

- **Python 3.11+** required; **Python 3.10 dropped**
- Production canonical path: `semi_log` + geometric adstock + Hill saturation
- `log_log` and extended transform kinds remain research-only unless explicitly validated
- Decide paths may fail on fingerprint or artifact contract mismatches (intentional)
- Optional strict gates: `require_production_certification`, `require_promoted_model_for_prod_decision`

### Validation

```bash
ruff check mmm tests
pytest tests/ -q -m "not slow"
```

CI runs on Python **3.11** and **3.12**; nightly workflow runs extended certification suites.
