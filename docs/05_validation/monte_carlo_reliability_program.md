# Monte Carlo Reliability Program (Phase 5F)

**Version:** `monte_carlo_reliability_v1.0.0`  
**Status:** Active — pilot characterization complete; tier-1+ execution planned  
**Module:** `mmm/validation/synthetic/monte_carlo_reliability.py`  
**Pilot artifact:** [investigations/monte_carlo_pilot_characterization.json](investigations/monte_carlo_pilot_characterization.json)  
**Recommendations:** [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md)

**Governance context:** [reliability_threshold_governance.md](reliability_threshold_governance.md) · [trust_report_semantics.md](trust_report_semantics.md)

---

## 1. Objectives

Replace **provisional** `TBD_v1_runtime` thresholds with **statistically justified recommendations** derived from large-scale synthetic evidence — without building new MMM estimators or changing production decision logic.

| Objective | Success criterion |
|-----------|-----------------|
| Characterize reliability **distributions** per capability | Percentiles and pass rates by world stratum |
| Map **failure regions** to world axes | Envelopes: noise, correlation, geos, periods, channels, drift |
| Propose **threshold recommendations** with confidence | Per VAL row; not auto-approved |
| Calibrate **TrustReport** green/yellow/red | Empirical basis for `trust_grade` |
| Inform **release-gate influence** | Decision-grade vs diagnostic vs structural |

**Non-goals:** Bayesian MMM, state-space, Nevergrad, new transforms, production decisioning changes.

---

## 2. World families

| Family | Purpose | Current tier-0 count |
|--------|---------|----------------------|
| **anchor_recovery** | WORLD-008–012 reference worlds | 5 |
| **behavioral_lattice** | L5B rich DGP axis sweep | 10 |
| **structural_lattice** | L5A contract-only sweep | 12 |
| **identifiability_mini** | INV-056 controlled 1ch/2ch worlds | 3 |
| **volume_sweep** | geos × periods × noise ablations | 12 |
| **negative_gate** | Expected gate failures (DR-03) | Planned tier-1 |

---

## 3. Sampling strategy

### 3.1 Principle: structured coverage, not uniform random

Do **not** draw i.i.d. uniform samples over the full Cartesian product of axes. That wastes budget on redundant cells and under-samples failure boundaries.

Use **stratified coverage**:

1. **Anchor replication** — every tier includes WORLD-008–012 (regression anchors).
2. **World-type strata** — at least `n_min` worlds per `world_type` (exact_recovery, optimizer, replay, drift, identifiability).
3. **Boundary emphasis** — oversample axes where INV-056 showed cliffs: `correlation_level=severe`, `drift=on`, `n_channels≥3`.
4. **Latin-hypercube / fractional factorial** — for tier-1+, explore noise × geos × periods without full factorial.
5. **Negative worlds** — fixed catalog slice per release (not random).

### 3.2 Axes (coverage dimensions)

| Axis | Levels (tier-1 target) | Rationale |
|------|------------------------|-----------|
| `noise_level` | zero, low, medium, high | Observation noise vs identifiability |
| `signal_strength` | low, medium, high | KPI signal-to-noise in DGP |
| `n_geos` | 1, 2, 4, 8 | Panel breadth |
| `n_periods` | 10, 14, 18, 26, 52 | Time series length |
| `n_channels` | 1, 2, 3 | Multi-channel homogenization |
| `correlation_level` | low, moderate, severe | Collinearity |
| `drift` | off, on | VAL-012 / trust modifiers |
| `replay` | off, on | VAL-006 / calibration |
| `calibration_freshness` | fresh, stale, missing | VAL-007 trust path |
| `missingness` | none, sparse_geo, sparse_period | Robustness (tier-2+) |

### 3.3 Tier staging

| Tier | Target N | Use |
|------|----------|-----|
| **tier_0_pilot** | ~25 | Characterize distributions (current) |
| **tier_1_calibration** | 100 | First percentile estimates; threshold **recommendations** |
| **tier_2_release_review** | 1,000 | DR-04 approval evidence |
| **tier_3_full_monte_carlo** | 10,000 | Full failure maps; Bayesian world design input |

---

## 4. Metrics

Metrics align with [validation_registry.md](validation_registry.md) and metric classes from Phase 5D.

| Capability | VAL / REC | Metric class | Primary statistic |
|------------|-----------|--------------|-------------------|
| Coefficient recovery | VAL-001 | diagnostic_attribution | max \|β̂−β\|, pass rate |
| Transform recovery | VAL-002/003 | diagnostic_attribution | decay/Hill error |
| Δμ recovery | VAL-004 | decision_grade | relative Δμ error |
| Optimizer recovery | VAL-005 | decision_grade | L1 allocation, regret |
| Replay recovery | VAL-006 | decision_grade | lift error |
| Drift detection | VAL-012 | trust_modifier | severity, val_012_outcome |
| Identifiability | VAL-013 | trust_modifier | VIF, warnings |
| Structural / contract | CERT-4A | structural | pass rate |

**Scorecard rollups:** `decision_reliability_score`, `attribution_diagnostic_score`, `structural_reliability_score`, `trust_modifier_status`, `trust_report_interpretation`.

---

## 5. Outputs

| Output | Path | Description |
|--------|------|-------------|
| Pilot characterization | `investigations/monte_carlo_pilot_characterization.json` | Tier-0 distributions + boundaries |
| Threshold recommendations | [monte_carlo_threshold_recommendations.md](monte_carlo_threshold_recommendations.md) | Human-readable recommendations |
| ReliabilityScorecard batch | `validation/reports/monte_carlo_tier_*_scorecard.json` | Future tier runs |
| Failure maps | Embedded in characterization JSON | Axis → failure driver |
| Governance packet | Input to DR-04 approval | tier-2+ only |

Regenerate pilot JSON:

```python
from pathlib import Path
from mmm.validation.synthetic.monte_carlo_reliability import write_pilot_characterization
write_pilot_characterization(Path("."))
```

---

## 6. Threshold calibration methodology

### 6.1 Process (per metric)

1. **Collect** metric values across tier-N worlds (stratified).
2. **Stratify** by `world_type` and critical axes (correlation, channels).
3. **Estimate distribution** — pass rate, percentiles of error metrics.
4. **Compare** to current `TBD_v1_runtime` constants in code.
5. **Propose** threshold — e.g. P95 error under anchor stratum, or pass rate ≥ 0.95 for decision-grade on anchor-only.
6. **Assign confidence** — low (tier-0), medium (tier-1), high (tier-2).
7. **Submit** to DR-04 — **no auto-approval**.

### 6.2 Decision-grade vs diagnostic

| Class | Calibration goal |
|-------|------------------|
| decision_grade | Tight bounds; candidate for release gate after approval |
| diagnostic_attribution | Loose bounds; report-only unless attribution profile |
| trust_modifier | Severity bands (drift), not point estimates only |
| structural | Logical + contract; near-zero false-pass tolerance |

### 6.3 TrustReport calibration

Map empirical scorecard joint distribution to:

| trust_grade | Business label | Pilot heuristic (tier-0) |
|-------------|----------------|--------------------------|
| high | green | decision ≥ 0.85, structural ≥ 0.9, trust acceptable |
| moderate | yellow | decision usable, attribution unsafe, trust caution OK |
| low / insufficient | red | structural fail, trust degraded, or decision &lt; 0.75 |

See [trust_report_semantics.md](trust_report_semantics.md) and recommendations doc §7.

---

## 7. Reliability boundary analysis (method)

Answer: **at what conditions does recovery fail?**

Methods:

- **Envelope tables** — slice pass rate by axis level (from tier-1+ batch).
- **Failure maps** — link drivers (shared transform, collinearity) to capabilities.
- **Coverage tables** — N per stratum vs target.

Tier-0 findings (INV-056 + L5B):

- Coef recovery fails on multi-channel shared-transform worlds **independent of** noise (0 vs 0.02) and geos/periods in volume sweep.
- Δμ recovery passes on same cells.
- Pinned transforms recover coef on 1ch / 2ch-orth mini-worlds.

---

## 8. Design decisions resolved (Phase 5F)

### DR-03 — Negative world representation ✅

**Decision:**

1. **`metadata.negative_world: true`** on catalog entries that expect failures.
2. **`artifact_truth.expected_failures`** — enumerated list of `{gate_id, expected_outcome, optional_error_class}`.
3. **Shared archetypes** — negative worlds use same bundle layout and materializer as positive; differ only in truth and expected gates.
4. **Scorecard exclusion** — negative worlds contribute to **gate-contract** tests only; excluded from recovery pass-rate numerators unless explicitly tagged `negative:gate_test`.

**Rejected:** Separate materialization path; unparameterized PolicyError-only templates without registry rows.

### DR-06 — ReliabilityScorecard release role ✅

**Decision:**

| Mode | Role |
|------|------|
| **CI tier-0** (pilot, N&lt;100) | **Advisory** — publish JSON; no semver block |
| **CI tier-1** (N≥100) | **Regression gate** on `decision_reliability_score` + structural; diagnostic failures warn only |
| **Release review tier-2** (N≥1000) | **Required evidence** for DR-04 threshold approval; still not `approved_for_prod` alone |
| **Production** | Scorecard never replaces promotion workflow or TrustReport on real runs |

**Regression policy:** If unit tests pass but `decision_reliability_score` drops &gt; 0.05 vs baseline tier-N snapshot, block merge pending review.

**Minimum worlds:** tier-1 = 100, tier-2 = 1000 (aligned with roadmap §10).

---

## 9. Track 4 gate (post-5F)

Only after tier-1+ threshold recommendations are reviewed:

1. Bayesian Hierarchical Geo MMM roadmap refinement  
2. CalibrationSignal integration design  
3. Hierarchical experiment priors  
4. Partial pooling validation worlds (INV-064)  
5. Bayesian prototype  

**Not next:** Nevergrad, state-space, dynamic coefficients.

**Platform principle:** Next model plugs into DecisionSurface, Estimand, CalibrationSignal, TrustReport, ReleaseGate — does not redefine them.

---

## 10. Related investigations

| ID | Status |
|----|--------|
| INV-022 | Partial — pilot characterization; tier-1 runner open |
| INV-060 | Open — threshold approval after tier-1 |
| INV-056 | Closed — failure taxonomy feeds sampling |
| INV-031 | Partial — DR-04 sign-off pending tier-2 |

---

## 11. References

- [synthetic_validation_roadmap.md](synthetic_validation_roadmap.md) §10 scale tiers  
- [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md)  
- [behavioral_lattice_sweep.md](behavioral_lattice_sweep.md)
