# Phase 5C — Exact Recovery Investigation Program

**Status:** Investigation complete (Phase 5C) — analysis artifacts published  
**Investigation:** [INV-056 — Exact recovery failure analysis](../06_investigations/open_investigations.md#inv-056--exact-recovery-failure-analysis-phase-5c)  
**Report:** [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md)  
**Priority:** Critical

---

## Purpose

Understand **why** WORLD-008 exact-recovery worlds (and L5B lattice variants) fail **coefficient and transform recovery** despite using the **same functional family** as production MMM (Ridge BO, `semi_log`, geometric adstock, Hill).

**Success criterion:** Explain observed recovery failures **before** attempting to fix them or expand the model family.

---

## Evidence motivating this phase (Phase 4B + 5B)

| Observation | Source |
|-------------|--------|
| **Structural reliability high** | Phase 5A/5B lattice: structural score ~0.89; contracts pass |
| **Behavioral reliability materially lower** | Phase 5B: behavioral score ~0.57 |
| **Exact-recovery coef/transform failures** | WORLD-008, L5B exact_recovery: REC-4B2-001–003 fail |
| **Platform contracts stable** | DecisionSurface, Estimand, CalibrationSignal, TrustReport unchanged |
| **Optimizer / replay / drift / identifiability** | Partially exercised; distinct failure modes |

The package can now **tell us where it is wrong**. The next stage is **scientific reliability**, not additional feature expansion.

---

## Questions to answer (INV-056)

| # | Question |
|---|----------|
| 1 | Are coefficients unrecoverable because of **Ridge shrinkage**? |
| 2 | Because of **hyperparameter search** (decay / Hill)? |
| 3 | Because of **identifiability** (channels, geos)? |
| 4 | Do **decay and Hill estimates compensate** for coefficient errors? |
| 5 | Does the **CV objective conflict** with truth recovery? |
| 6 | Are **TBD_v1 tolerances unrealistic** for the current estimator? |
| 7 | Does **truth-pinned transform fitting** recover coefficients? |
| 8 | Recovery behavior as **noise**, **geos**, **periods**, **channels** increase? |
| 9 | What is the **theoretical recovery ceiling** for the current architecture? |

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| **Recovery investigation report** | Root-cause narrative with evidence tables |
| **Recovery failure taxonomy** | Shrinkage vs search vs identifiability vs tolerance vs architecture ceiling |
| **Recovery sensitivity analysis** | Axes: noise, n_geos, n_periods, n_channels; optional truth-pinned ablations |
| **Threshold calibration recommendations** | Encoded in Phase 5D — [reliability_threshold_governance.md](reliability_threshold_governance.md) (DR-04 draft) |
| **Bayesian validation world recommendations** | Input to Track 4 Bayes-H2 — what worlds must exist before Bayes work |

Artifacts:

| Artifact | Path |
|----------|------|
| Main report | [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md) |
| Machine-readable bundle | [investigations/exact_recovery_findings.json](investigations/exact_recovery_findings.json) |
| Regenerate | `python -m mmm.validation.synthetic.exact_recovery_investigation` or `write_investigation_report(Path('.'))` |

Supporting JSON under `docs/05_validation/investigations/` (e.g. [exact_recovery_findings.json](investigations/exact_recovery_findings.json)).

---

## Explicit non-goals (Phase 5C)

- No new MMM methods
- No Bayesian development
- No state-space models
- No new transforms
- No new optimizers
- No orchestration work
- No production release gates based on this investigation alone

---

## Relationship to later phases

| Phase | Depends on 5C |
|-------|----------------|
| **5D** — Reliability threshold governance ✅ | Metric classes, DR-04 draft, scorecard v1.1 |
| **5E** — Drift detection & TrustReport downgrade | Drift partials understood in context of recovery limits |
| **5F** — Monte Carlo reliability program | Sampling design informed by failure taxonomy |
| **Track 4** — Bayesian hierarchical geo MMM | Bayes-H2 worlds designed after recovery ceiling is known |

---

## Platform roadmap gate

No **major production modeling expansion** may proceed until:

1. **Exact recovery investigation** (Phase 5C) is complete  
2. **Reliability threshold governance** (Phase 5D) is complete  
3. **Drift validation** (Phase 5E) is complete  

See [platform_roadmap.md](platform_roadmap.md).

---

## Related

- [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md)  
- [reliability_scorecard.md](reliability_scorecard.md)  
- [dgp_materialization.md](dgp_materialization.md)  
- [recovery_certification.py](../../mmm/validation/synthetic/recovery_certification.py)
