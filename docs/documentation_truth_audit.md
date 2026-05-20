# Documentation truth audit

Audit date: 2026-05-18 (updated). Scope: canonical `docs/` journey tree vs package behavior (Ridge prod decisioning; diagnostics separated from decisions).

## Verified (docs match code)

| Topic | Evidence |
|--------|----------|
| Ridge prod uses full-panel Δμ for decide/optimize | `mmm/planning/decision_simulate.py`, `mmm/decision/service.py` |
| Curves are diagnostic-only | `curve_bundles_are_diagnostic_only` in `extension_runner`, `decision_bundle` unsupported_questions |
| Prod replay calibration gate | `mmm/governance/replay_evidence.py`, `mmm/calibration/replay_prod_gate.py` |
| `planning_assumptions` on decision bundles | `mmm/planning/assumption_contract.py`, prod CLI validation |
| Artifact tiers (research vs decision) | `artifact_tier_disclosure` in extension report; CLI `artifact_tier=decision` |
| Local artifact backend default | `ArtifactBackend.LOCAL` in `mmm/config/schema.py`; trainer uses `mmm/artifacts/factory.py` |
| Artifact lifecycle E2E | `mmm/artifacts/lifecycle.py`, `compatibility.verify_run_roundtrip`, `tests/test_artifact_lifecycle_e2e.py` |
| MLflow file-store (experimental) | `classify_backend()` — CI contract-tested; remote tracking not guaranteed |
| Accepted-run drift registry | `mmm/evaluation/run_registry.py`, `extensions.drift_historical.use_registry` |
| Ridge uncertainty research | `mmm/governance/ridge_uncertainty_research.py`, `docs/04_governance/ridge_uncertainty_research.md` |
| Performance audit artifact | `mmm/evaluation/performance_audit.py`, `performance_report` extension |
| Control policy packs | `mmm/governance/control_policy_packs.py` (guidance only) |
| CI release gates | `.github/workflows/ci.yml` — Python 3.10–3.12, docs + artifact + reproducibility jobs |
| Documentation inventory | `docs/DOCUMENTATION_INVENTORY.md` |
| Doc link validation | `scripts/validate_docs.py` + `tests/test_docs_validation.py` |
| Seed contract + `seed_resolution` artifact | `mmm/contracts/seed_resolution.py`, `load_config` / `resolve_config` |
| Fingerprint v2 fields | `mmm/data/fingerprint.py` `fingerprint_details` |
| Ridge uncertainty disclosure | `mmm/governance/decision_uncertainty.py` |
| Cross-branch doc quarantine | [`docs/_archive/cross_branch_not_shipped.md`](_archive/cross_branch_not_shipped.md) — no `02_concepts` scheduler/separability docs on this branch |

## Corrected (stale wording; prefer these sources)

| Was | Now |
|-----|-----|
| Flat `docs/planning_artifact_schema.md` only | Canonical: `docs/04_governance/artifact_schema.md`; flat file is redirect stub |
| “Bayesian prod budgeting” implied in older overviews | Bayesian blocked for prod decision surfaces; research/diagnostic only |
| Implicit `sampler_seed: 42` always | `None` inherits from `random_seed`; explicit YAML overrides |
| Panel fingerprint = channels only | v2 includes controls, transforms, seeds, `data_version_id`, optional planning assumptions |

## Fingerprint v2 migration (legacy-compatible)

New runs emit `data_fingerprint` / `panel_fingerprint` with:

| Field | Role |
|-------|------|
| `fingerprint_version` | `"fingerprint_v2"` |
| `sha256_combined` | Primary hash over panel + schema + config + seeds (+ optional planning assumptions) |
| `sha256_panel_keycols_sorted_csv` | Legacy panel-only hash (still present) |
| `sha256_schema_json` | Legacy schema hash (still present) |
| `fingerprint_details` | `included_fields`, `omitted_fields` (timestamps, `run_id`, generated IDs excluded) |

**Older bundles** (pre-v2) may only have `sha256_panel_keycols_sorted_csv` and `sha256_schema_json` without `sha256_combined`. Lineage and prod gates remain **legacy-compatible**: validators accept non-empty fingerprint dicts; equality checks should prefer `sha256_combined` when both sides have it, else fall back to `sha256_panel_keycols_sorted_csv`. Drift monitoring uses the same fallback.

Implementation: `mmm/data/fingerprint.py`. See [04_governance/artifact_schema.md](04_governance/artifact_schema.md).

## Documentation structure (canonical)

| Area | Path |
|------|------|
| Getting started | `docs/01_getting_started/` |
| Concepts | `docs/02_concepts/` |
| Planning | `docs/03_planning/` |
| Governance | `docs/04_governance/` |
| Inventory | `docs/DOCUMENTATION_INVENTORY.md` |

No duplicate flat copies except redirect stubs (`docs/planning_artifact_schema.md` → governance artifact schema).

## Cross-branch documentation policy

Features documented under `02_concepts/` may still be gated by config or branch merge state. See [`docs/_archive/cross_branch_not_shipped.md`](_archive/cross_branch_not_shipped.md) before treating roadmap-only notes as shipped product.

## Removed / archived (do not treat as current product)

| Doc | Status |
|-----|--------|
| `docs/_archive/roadmap_causal_calibration_governance.md` | Roadmap only — not shipped features |
| Duplicate flat docs (`docs/quickstart.md`, etc.) | Redirect to `01_getting_started/` per inventory |

## Intentionally unsupported (documented as out of scope)

| Capability | Supported | Experimental | Unsupported |
|------------|-----------|--------------|-------------|
| Prod Bayesian budget optimization | | | ✓ |
| MLflow remote artifact backend | | ✓ | ✓ (not contract-tested) |
| MLflow file-store backend | | ✓ (CI roundtrip) | |
| Calibrated Ridge monetary CIs | | | ✓ |
| Automatic retraining on drift | | | ✓ |
| External drift monitoring SaaS | | | ✓ |
| Auto control insertion from diagnostics | | | ✓ |
| Bootstrap/conformal Ridge monetary CIs | | | ✓ (deferred; disclosure in `decision_uncertainty`) |
| GPU / distributed training | | | ✓ |
| Curve-as-decision-truth | | | ✓ (diagnostic only) |
| `override_unsafe` without waiver (prod) | | | ✓ (blocked) |
| Geo-rank CV in prod | | | ✓ |

## Capability matrix (summary)

| Capability | Supported | Experimental | Unsupported |
|------------|-----------|--------------|-------------|
| Ridge BO + calendar CV prod | ✓ | | |
| Full-panel simulate / optimize-budget | ✓ | | |
| Replay calibration (decision path) | ✓ | | |
| Response curves / ROI bridge | ✓ (diagnostic) | | |
| Point-only Ridge decisions | ✓ | | |
| Bayesian PyMC fit | | ✓ | prod decision |
| MLflow logging | | ✓ | |
| Seed contract artifacts | ✓ | | |
| Fingerprint v2 | ✓ | | |
| Drift report + historical prior-run compare | ✓ | | |
| Control governance diagnostics | ✓ (guidance) | | |
| Curve vs Δμ alignment matrix | ✓ (diagnostic) | | |
| Stakeholder `model_card.md` sections | ✓ | | |
| Artifact store factory | ✓ local / experimental mlflow | | |

## Tests / link checks

- Unit tests: `test_docs_validation`, `test_artifact_factory`, `test_curve_decision_scenarios`, `test_drift_historical`, `test_control_governance`, plus seed/fingerprint/curve/uncertainty/drift/model_card suites.
- Link validation: `python scripts/validate_docs.py` (also `tests/test_docs_validation.py`).
- Doc examples: Python snippets in `python_api.md` are illustrative; run via package tests rather than doc execution.

## Remaining doc risks

- `docs/02_concepts/bayesian.md` must keep prod-blocking language prominent.
- Cross-branch merges should update `DOCUMENTATION_INVENTORY.md` when adding pages.
