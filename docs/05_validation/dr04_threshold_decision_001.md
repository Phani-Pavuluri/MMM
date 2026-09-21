# DR-04 threshold decision 001

**Status:** implementation decision record; no gate implementation
**Owner:** MMM repository governance
**Source evidence:** `docs/05_validation/archives/MMM_TIER1_MONTE_CARLO_CHARACTERIZATION_001.json`
**Source tier:** `tier_1_calibration`, `world_count=108`
**Decision scope:** validation-registry threshold status and required evidence only
**Authority impact:** no production, model, optimizer, MIP, GeoX, or release authority change

## Decision

DR-04 treats **approval or honest retention** as a successful task outcome. A
row is not required to graduate merely because it has a measured Tier-1 result.
Rows whose scored evidence cannot carry a versioned threshold remain
`TBD_v1_runtime` and receive an explicit Tier-2 promotion bar.

The Tier-1 artifact reports nominal observations, not independent evidence
counts. Multiple worlds share templates and generator conditions; an effective
independent sample size is not recoverable from the committed summary for every
metric family. The decision therefore does not invent a discount factor. Where
effective coverage is unknown or thin, the row remains provisional.

Expected-failure worlds are not mixed into positive-recovery pass-rate
denominators. Tier-2 must score recovery and expected-failure behavior in
separate distributions.

## Per-row rulings

| Row(s) | Tier-1 evidence | Ruling | Reason and next bar |
|---|---:|---|---|
| VAL-001 | n=13, pass rate 0.0 | retain provisional, diagnostic-only | Attribution recovery is report-only; Tier-2 independent coefficient/geometry strata required |
| VAL-002 | n=13, pass rate 0.0 | retain provisional, diagnostic-only | Transform evidence is thin and failed the current characterization; Tier-2 lag/transform strata required |
| VAL-003 | covered with VAL-002/003 transform family, n=13 | retain provisional, diagnostic-only | Saturation evidence requires independent shape/support strata |
| VAL-004 | n=13, pass rate 1.0 | retain provisional | Decision-grade Δμ result is encouraging but cannot graduate on nominal n=13; Tier-2 independent Δμ strata required |
| VAL-005 | n=5, pass rate 0.6 | retain provisional | Optimizer evidence is thin and unstable; Tier-2 corner, constraint, dominance, and regret strata required |
| VAL-006 | n=5, pass rate 1.0 | retain provisional | Replay evidence needs experiment-quality and uncertainty strata; Tier-2 expected-failure cases required |
| VAL-007 | no independently scored Tier-1 distribution | retain provisional | Calibration robustness depends on VAL-006 and freshness/false-attach coverage |
| VAL-008 | logical gate semantics | approved (logical) | Existing logical approval remains; negative-world gate-match fixtures must be maintained |
| VAL-009 | structural/contract evidence, numeric fields unresolved | retain provisional for numeric tolerance fields | Tier-2 artifact and fingerprint tolerance evidence required |
| VAL-010 | no sufficient independent numeric characterization | retain provisional | Tier-2 independent reference-run and write-twice evidence required |
| VAL-011 | logical promotion workflow | approved (logical) | Existing logical approval remains; expired/mismatch negative fixtures remain required |
| VAL-012 | n=3, pass rate 1.0 | retain provisional | Trust severity cannot graduate from n=3; Tier-2 stable/drift and expected-warning/block strata required |
| VAL-013 | no sufficient independent numeric characterization | retain provisional | Tier-2 governance disagreement and expected-outcome fixtures required |
| VAL-014 | logical certification-level match | approved (logical) | Existing logical approval remains; expected certification-level negative worlds required |

## Effective-evidence limitation

The nominal structural count of 108 does not imply 108 independent worlds.
Coefficient/transform/lift scoring has nominal n=13; optimizer and replay n=5;
drift n=3; identifiability n=2. The committed artifact does not provide a
defensible effective-n estimate after template/generator correlation. DR-04
therefore records the limitation and retains affected numeric rows instead of
converting nominal counts into unsupported confidence.

## Tier-2 prerequisite and promotion bar

Before Tier-2 execution, a bounded runner-enhancement task must add:

- configurable panel size, geo count, privacy loss, missingness, noise,
  collinearity, drift, and seasonality strata;
- deterministic world/template/generator/stratum identifiers;
- independent-stratum coverage reporting;
- separate positive-recovery and expected-failure scorecards;
- assertions that expected failures produce the correct warning or block; and
- per-metric evidence summaries.

Tier-2 promotion bars must be written before execution and expressed in
independent-stratum coverage, not raw world count. Each retained row needs a
metric-specific minimum stratum bar, pass/error bound, scope coverage, and
re-verification trigger.

## Non-claims and deferred work

This record does not claim causal validity, universal MMM superiority, production
readiness, or threshold sufficiency for unlisted conditions. It does not change
release-gate code, fitters, transforms, optimizers, Bayesian paths, TrustReport,
MIP, GeoX, or any authority flag. Tier-2 runner enhancement and Tier-2 evidence
expansion are deferred successor tasks.

## Sign-off

DR-04 owner sign-off is represented by the exact reviewed task head through the
MMM lifecycle. This document records the ruling; it does not self-authorize the
result.
