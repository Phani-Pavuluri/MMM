# Investigation index

Grouped view of [open_investigations.md](open_investigations.md). Full records remain in the backlog file.

**Planning frame:** [platform_roadmap.md](../05_validation/platform_roadmap.md) (contract-driven Marketing Intelligence Platform)

**Last consolidated:** 2026-05-29 (Phase 5F Monte Carlo pilot; DR-03/DR-06 resolved)

---

## By platform track (primary)

### Track 1 — Platform Contract Layer

| ID | Title | Severity |
|----|-------|----------|
| [INV-014](open_investigations.md#inv-014--nested-train-decision_bundle-is-research-tier-not-prod-decide-output) | Train `decision_bundle` vs CLI decide | medium |
| [INV-024](open_investigations.md#inv-024--documentation-drift-replay-holdout-split-described-inconsistently) | Doc drift (replay holdout) | medium |
| [INV-054](open_investigations.md#inv-054--cross-branch-features-quarantined-in-archive-docs) | Cross-branch doc quarantine | low |

### Track 2 — Reliability & Validation Program

| ID | Title | Severity |
|----|-------|----------|
| [INV-056](open_investigations.md#inv-056--exact-recovery-failure-analysis-phase-5c) | Exact recovery analysis (5C) — **closed** | — |
| [INV-057](open_investigations.md#inv-057--decision-vs-attribution-threshold-separation) | Decision vs attribution thresholds (5D) — **closed** | — |
| [INV-055](open_investigations.md#inv-055--dedicated-val-012-drift_detection_runner-phase-5e) | VAL-012 drift runner (5E) — **closed** | — |
| [INV-058](open_investigations.md#inv-058--transformβ-compensation-monitoring) | Transform/β compensation monitoring | medium |
| [INV-059](open_investigations.md#inv-059--reliabilityscorecard-metric-class-refactor) | Scorecard metric-class refactor — **closed** | — |
| [INV-060](open_investigations.md#inv-060--monte-carlo-threshold-calibration) | MC thresholds — tier-0 done; approval open | medium |
| [INV-022](open_investigations.md#inv-022--monte-carlo-reliability-program-and-reliabilityscorecard-not-built) | Tier-1 MC batch runner | medium |
| [INV-017](open_investigations.md#inv-017--validation-registry-thresholds-remain-tbd_v1) | VAL thresholds `TBD_v1` | critical |
| [INV-008](open_investigations.md#inv-008--synthetic-certification-check_registry-not-yet-driven-by-groundtruthworld-bundles) | CHECK_REGISTRY not world-driven | high |
| [INV-016](open_investigations.md#inv-016--world-materializer-renders-constant-panel-not-generative-dgp) | Constant panel materializer | high |
| [INV-020](open_investigations.md#inv-020--minimal-dgp-archetype-library-not-delivered-roadmap-phase-2) | DGP archetype library | high |
| [INV-023](open_investigations.md#inv-023--certification-runner-not-wired-to-world-bundles) | Cert runners ↔ worlds | high |
| [INV-027](open_investigations.md#inv-027--pilot-threshold-calibration-100-worlds-not-executed) | Pilot calibration | high |
| [INV-018](open_investigations.md#inv-018--world-validator-level-4-certification-compatibility-not-implemented) | Validator L4 | medium |
| [INV-025](open_investigations.md#inv-025--negative-world-catalog-representation-unresolved-dr-03) | Negative worlds DR-03 | medium |
| [INV-029](open_investigations.md#inv-029--runtime-vs-ci-certification-split-undefined-dr-05) | Runtime vs CI cert DR-05 | medium |
| [INV-030](open_investigations.md#inv-030--reliabilityscorecard-release-role-undefined-dr-06) | Scorecard role DR-06 | medium |
| [INV-031](open_investigations.md#inv-031--threshold-ownership-process-unset-dr-04) | Threshold ownership DR-04 (draft resolution) | medium |
| [INV-045](open_investigations.md#inv-045--external-benchmarks-phase-6-must-not-set-prod-approval) | External benchmarks | medium |
| [INV-050](open_investigations.md#inv-050--production-catalog-index-file-for-worlds-not-created) | World catalog index | low |
| [INV-053](open_investigations.md#inv-053--l3-replay-and-optimizer-numeric-tolerances-still-tbd_v1) | L3 tolerances TBD_v1 | medium |

*Note: INV-021 ScenarioBuilder — **resolved** by Phase 3B ✅; retained in backlog for history only.*

### Track 3 — Core Production Decisioning

| ID | Title | Severity |
|----|-------|----------|
| [INV-003](open_investigations.md#inv-003--observational-mmm-and-replay-do-not-establish-causal-incrementality) | No causal incrementality claim | critical |
| [INV-001](open_investigations.md#inv-001--optimizer-certification-on-synthetic-surfaces-not-real-panel-geometry) | Optimizer cert synthetic only | high |
| [INV-002](open_investigations.md#inv-002--production-readiness-default-allows-decide-with-severe-warning-only) | Readiness warn-not-block | high |
| [INV-004](open_investigations.md#inv-004--fingerprint-traindecide-alignment-uses-legacy-fallback-path) | Fingerprint legacy fallback | high |
| [INV-005](open_investigations.md#inv-005--full-panel-replay-refit-in-prod-requires-explicit-waiver) | Replay refit waiver | high |
| [INV-007](open_investigations.md#inv-007--severe-replay-generalization-gap-warns-by-default-blocks-only-if-configured) | Replay gap warn-by-default | high |
| [INV-009](open_investigations.md#inv-009--identifiability-and-optimization-approval-can-be-waived-in-prod) | Identifiability waivers | high |
| [INV-011](open_investigations.md#inv-011--aggregate-only-experiment-evidence-on-geo-panels) | Aggregate-only evidence | high |
| [INV-028](open_investigations.md#inv-028--fingerprint-mismatch-and-unsafe-api-waivers-depend-on-signed-json-discipline) | Waiver discipline | high |
| [INV-042](open_investigations.md#inv-042--collinear-channels-separability-diagnostic-does-not-fix-attribution) | Collinear attribution | high |
| [INV-010](open_investigations.md#inv-010--panel-qa-prod-block-can-be-waived-via-prod_block_waiver) | Panel QA waiver | medium |
| [INV-013](open_investigations.md#inv-013--reproducibility-self-certification-is-not-independent-evidence) | Repro self-cert | medium |
| [INV-041](open_investigations.md#inv-041--coefficient-shift-handling-limited-to-drift-detection-and-freshness-warnings) | Drift advisory only | medium |
| [INV-043](open_investigations.md#inv-043--log_log-and-non-canonical-transforms-blocked-in-prod) | Canonical transforms | medium |
| [INV-051](open_investigations.md#inv-051--geo-rank-cv-not-supported-in-prod) | Geo-rank CV | medium |
| [INV-012](open_investigations.md#inv-012--allocated-shocks-are-computational-bridges-only) | Allocated shocks | medium |

### Track 4 — Research Sandbox

| ID | Title | Severity |
|----|-------|----------|
| [INV-032](open_investigations.md#inv-032--bayesian-production-budget-decisioning-blocked) | Bayesian prod blocked | informational |
| [INV-033](open_investigations.md#inv-033--bayesian-experiment-likelihood-is-research-only) | Bayesian exp likelihood | informational |
| [INV-034](open_investigations.md#inv-034--bayesian-hierarchy-partial-pooling-is-research-only) | Bayesian hierarchy | informational |
| [INV-037](open_investigations.md#inv-037--robust-optimization-extension-is-research-only) | Robust optimization | informational |
| [INV-035](open_investigations.md#inv-035--ridge-monetary-confidence-intervals-not-production-grade) | Ridge CIs | medium |
| [INV-036](open_investigations.md#inv-036--conformal-intervals-not-implemented-for-ridge) | Conformal N/I | low |
| [INV-040](open_investigations.md#inv-040--state-space-and-time-varying-coefficients-not-production-supported) | State-space | informational |
| [INV-052](open_investigations.md#inv-052--dynamic-priors-explicitly-out-of-synthetic-validation-scope) | Dynamic priors | informational |
| [INV-049](open_investigations.md#inv-049--stan-backend-stub-pymc-is-default-bayesian-path) | Stan stub | low |
| [INV-063](open_investigations.md#inv-063--bayesian-hierarchical-geo-mmm-design-bayes-h1) | Bayes-H1–H2b ADRs ✅ — worlds next | closed |
| [INV-066](open_investigations.md#inv-066--local-experiment-priors-for-hierarchical-bayesian-mmm-bayes-h1h4) | Scope propagation — Bayes-H2b ✅ | closed |
| [INV-064](open_investigations.md#inv-064--partial-pooling-validation-on-hierarchical-worlds-bayes-h2h4) | Partial pooling validation | medium |
| [INV-065](open_investigations.md#inv-065--posterior-calibration-and-coverage-bayes-h4) | Posterior calibration | medium |
| [INV-067](open_investigations.md#inv-067--bayesian-compute-scalability-for-geo-hierarchical-models-bayes-h3) | Bayes compute scale | medium |
| [INV-068](open_investigations.md#inv-068--bayesian-trustreport-compatibility-bayes-h1h4) | Bayes TrustReport | medium |
| [INV-069](open_investigations.md#inv-069--bayesian-decisionsurface-compatibility-bayes-h4h5) | Bayes DecisionSurface | high |
**Roadmap:** [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md) (Bayes-H1–H2b ✅; materialize `WORLD-BAYES-*`; no PyMC until catalog).

### Track 5 — Conversational / Orchestration

| ID | Title | Severity |
|----|-------|----------|
| [INV-039](open_investigations.md#inv-039--auto-retrain-auto-promotion-agentic-orchestration-out-of-v1-scope) | No auto-retrain / agents | informational |

---

## By investigation category (cross-cutting)

| Category | Representative IDs |
|----------|-------------------|
| **contract compatibility** | INV-014, INV-043, INV-048 |
| **semantic drift** | INV-024, INV-054 |
| **decision-surface fragmentation** | INV-015, INV-014 |
| **orchestration compatibility** | INV-039 |
| **replay semantics** | INV-005, INV-007, INV-011, INV-024 |
| **TrustReport consistency** | INV-002, INV-001 |
| **certification reliability gaps** | INV-008, INV-017, INV-023, INV-018, INV-058, INV-060, INV-064 |
| **Exact recovery / behavioral reliability** | INV-056 ✅, INV-057 ✅, [reliability_threshold_governance.md](../05_validation/reliability_threshold_governance.md) |
| **Bayesian hierarchical geo MMM** | INV-063–069, [bayesian_hierarchical_geo_mmm_roadmap.md](../05_validation/bayesian_hierarchical_geo_mmm_roadmap.md) |

---

## Legacy groupings (severity review)

### Immediate production risks (critical / high, Track 3)

Review **every release** — see Track 3 table above.

### Validation gaps (Track 2)

Review each **major reliability-program phase** — see Track 2 table.

### Intentional limitations

INV-003, INV-006, INV-012, INV-015, INV-032–034, INV-035, INV-038, INV-039, INV-041, INV-043, INV-044, INV-051, INV-055, INV-063

---

## Unresolved design decisions

| ID | Topic |
|----|-------|
| DR-01–DR-02 | ✅ Resolved (bundle, versioning) |
| DR-03–DR-06 | Open (negative worlds, thresholds, CI split, scorecard) |
| DR-07 | Platform contract ABI — [platform_roadmap.md](../05_validation/platform_roadmap.md) |

---

## Severity summary

| Severity | Count |
|----------|------:|
| critical | 2 |
| high | 14 |
| medium | 22 |
| low | 12 |
| informational | 5 |
| **Total** | **55** |

---

## Review cadence

| Bucket | Cadence |
|--------|---------|
| Track 3 critical/high (open) | Every release |
| Track 2 (open / blocked) | Every major reliability phase |
| Track 4–5, intentional limitations | Quarterly |

---

## Related docs

- [platform_roadmap.md](../05_validation/platform_roadmap.md)
- [synthetic_validation_roadmap.md](../05_validation/synthetic_validation_roadmap.md)
- [open_investigations.md](open_investigations.md)
