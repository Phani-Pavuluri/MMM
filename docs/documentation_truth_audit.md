# Documentation truth audit

Audit date: 2026-05-18. Scope: `docs/` tree vs current package behavior on `fix/prod-governance-evidence-gates` lineage (Ridge prod decisioning, no experiment scheduler on this branch).

## Verified (docs match code)

| Topic | Evidence |
|--------|----------|
| Ridge prod uses full-panel Δμ for decide/optimize | `mmm/planning/decision_simulate.py`, `mmm/decision/service.py` |
| Curves are diagnostic-only | `curve_bundles_are_diagnostic_only` in `extension_runner`, `decision_bundle` unsupported_questions |
| Prod replay calibration gate | `mmm/governance/replay_evidence.py`, `mmm/calibration/replay_prod_gate.py` |
| `planning_assumptions` on decision bundles | `mmm/planning/assumption_contract.py`, prod CLI validation |
| Artifact tiers (research vs decision) | `artifact_tier_disclosure` in extension report; CLI `artifact_tier=decision` |
| Local artifact backend default | `ArtifactBackend.LOCAL` in `mmm/config/schema.py` |
| Seed contract + `seed_resolution` artifact | `mmm/contracts/seed_resolution.py`, `load_config` / `resolve_config` |
| Fingerprint v2 fields | `mmm/data/fingerprint.py` `fingerprint_details` |
| Ridge uncertainty disclosure | `mmm/governance/decision_uncertainty.py` |
| Cross-branch doc quarantine | [`docs/_archive/cross_branch_not_shipped.md`](_archive/cross_branch_not_shipped.md) — no `02_concepts` scheduler/separability docs on this branch |

## Corrected (stale wording; prefer these sources)

| Was | Now |
|-----|-----|
| Flat `docs/planning_artifact_schema.md` only | Canonical: `docs/04_governance/artifact_schema.md` (flat copies kept for links) |
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

Implementation: `mmm/data/fingerprint.py`. See also [planning_artifact_schema.md](planning_artifact_schema.md#data_fingerprint-panel_fingerprint).

## Cross-branch documentation policy

This branch ships **flat** `docs/` only. It does **not** include `docs/02_concepts/experiment_scheduler.md`, `feature_separability.md`, or `control_templates.md`. Those belong to `feat/feature-separability-governance` and are listed in [`docs/_archive/cross_branch_not_shipped.md`](_archive/cross_branch_not_shipped.md). **Do not** advertise experiment scheduling or separability CLI until that branch merges.

## Removed / archived (do not treat as current product)

| Doc | Status |
|-----|--------|
| `docs/_archive/roadmap_causal_calibration_governance.md` | Roadmap only — not shipped features |
| Duplicate flat docs (`docs/quickstart.md`, etc.) | May exist on other branches; this branch uses flat `docs/*.md` |
| Experiment scheduler / separability user docs | Quarantined — see `_archive/cross_branch_not_shipped.md` |

## Intentionally unsupported (documented as out of scope)

| Capability | Supported | Experimental | Unsupported |
|------------|-----------|--------------|-------------|
| Prod Bayesian budget optimization | | | ✓ |
| MLflow artifact backend in CI | | ✓ (config enum) | ✓ (not contract-tested) |
| Calibrated Ridge monetary CIs | | | ✓ |
| Automatic retraining on drift | | | ✓ |
| External drift monitoring SaaS | | | ✓ |
| Experiment scheduler (this branch) | | | ✓ (on `feat/feature-separability-governance` only) |
| Feature separability extension (main/governance branch) | | | ✓ (separability branch) |
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
| Drift report artifact | ✓ | | |
| Auto `model_card.md` | ✓ | | |
| Experiment scheduler | | | ✓ (other branch) |

## Tests / link checks

- New unit tests: `test_seed_resolution`, `test_fingerprint_completeness`, `test_curve_decision_alignment`, `test_decision_uncertainty`, `test_drift_monitor`, `test_model_card`.
- Link validation: not automated in CI; manual spot-check `docs/README.md` and `01_getting_started` paths.
- Doc examples: Python snippets in `python_api.md` are illustrative; run via package tests rather than doc execution.

## Remaining doc risks

- Duplicate flat vs nested docs can diverge across branches until consolidated (deferred).
- `docs/bayesian.md` must keep prod-blocking language prominent.
- Merging a docs-restructure PR without `cross_branch_not_shipped.md` checks may re-introduce unavailable-feature docs.
