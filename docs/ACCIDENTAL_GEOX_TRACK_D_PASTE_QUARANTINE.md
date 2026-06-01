# Accidental GeoX Track B / Track D paste — quarantine notice

**Status:** Quarantine / rollback record  
**Date:** 2026-05-29  
**Action:** Removed from the **MMM** repository (not committed to `main`)

---

## What happened

GeoX / unified-MIP roadmap instructions for **Track B** (contract identity, B5b/B5c/B5d/M2 dual-write) and **Track D** (statistical robustness, SCM/TBR/placebo audits, method inventory D0–D8) were pasted into the **standalone MMM project** chat. An agent created docs and cross-links here by mistake.

That work belongs in the **GeoX / Panel Experimentation / unified MIP** repository, not in `mmm`.

---

## Removed from MMM (deleted)

| Path | Role |
|------|------|
| `docs/TRACK_D_METHOD_INVENTORY_AND_ROBUSTNESS_MATRIX_001.md` | Track D0 method inventory |
| `docs/TRACK_D_LITERATURE_CROSSCHECK_001.md` | Track D0b literature cross-check |
| `docs/DEFERRED_WORK_REGISTRY.md` | Track B/D execution order (B5*, M2, D0–D8) |
| `docs/OPEN_INVESTIGATIONS.md` | GeoX Track B/D investigations index |

## Edited in MMM (GeoX sections stripped)

| Path | Change |
|------|--------|
| `docs/05_validation/platform_roadmap.md` | Removed Track B & Track D sections; restored numeric tracks 1–5 only |
| `docs/README.md` | Removed Track B/D / deferred-registry links |
| `docs/DOCUMENTATION_INVENTORY.md` | Removed Track D registry entries |
| `docs/06_investigations/open_investigations.md` | Removed **INV-070** (Track D umbrella) |
| `docs/06_investigations/investigation_index.md` | Removed INV-070 row |

---

## Not removed (manual review if needed)

These may reference **GeoX** as an experiment evidence *source name* or **CalibrationSignal** as an MMM platform contract term. They were **not** part of the Track D paste rollback unless you explicitly want them quarantined too:

- `docs/05_validation/bayes_h*_*.md` — Bayes-H1/H2/H2b ADRs
- `docs/BAYES_H2B_VALIDATION_WORLDS_001.md`, `docs/BAYES_H2B_VALIDATION_RUNNER_002.md`
- `validation/worlds/WORLD-BAYES-*/`
- `docs/05_validation/synthetic_validation_roadmap.md` and synthetic validation code under `mmm/validation/synthetic/`
- MMM-native `TrustReport` semantics in Phase 5E (`trust_report_semantics.md`) — **not** GeoX Track B TrustReport composer

---

## Restore in GeoX project

Re-create in the **GeoX / MIP** repo:

- Registry `TRACK-D-STATISTICAL-ROBUSTNESS`
- D0 / D0b docs and D1–D8 phase plan
- Track B near-term: B5b, B5c, B5d, M2
- `DEFERRED_WORK_REGISTRY.md` and platform investigations index

---

## MMM-native follow-ups (valid here)

If useful, express separately in MMM language only:

- MMM model robustness audit (Ridge, transforms, leakage)
- MMM response curve validation
- MMM calibration / replay evidence intake (`ExperimentEvidence`, not GeoX ExperimentSpec pipeline)
- MMM prior/posterior diagnostics (Bayes sandbox, blocked until H2d/H3)
- MMM optimizer/planner robustness
- Synthetic falsification worlds (Track 2 — already in `synthetic_validation_roadmap.md`)

**Do not** re-import GeoX estimator promotion (SCM, TBRRidge, AugSynth, placebo-as-CI) into this repo without an explicit MMM scope decision.

**Exception (research lane only):** [track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md](track_d/D5_POW_SCM_UNIT_JACKKNIFE_READOUT.md) and `mmm/research/track_d/` characterize SCM+JK readout behavior under synthetic panels. This does **not** restore GeoX production estimators or promote SCM+JK to lift detection.
