# Drift detection (VAL-012)

**Module:** `mmm/validation/synthetic/drift_detection_runner.py`  
**Registry:** [validation_registry.md](validation_registry.md) VAL-012  
**Trust semantics:** [trust_report_semantics.md](trust_report_semantics.md)

---

## Purpose

Phase **5E** replaces partial VAL-012 behavior (pre-period train + ad hoc degradation checks) with a **dedicated drift detection runner** that compares model and panel behavior to declared `drift_truth` on synthetic worlds.

Production drift monitoring (`mmm/evaluation/drift_monitor.py`) remains separate; the runner validates that **synthetic worlds surface the right trust signals** when truth includes changepoints.

---

## Execution scope

| World family | Runner invoked |
|--------------|----------------|
| `WORLD-011-drift-recovery` | Yes — full VAL-012 |
| Behavioral lattice `world_type=drift`, `drift=on` | Yes — via recovery certification |
| Exact-recovery / optimizer / replay (no drift truth) | Pass — drift not in scope (`drift_expected=false`) |
| Structural lattice (Phase 5A) | VAL-012 deferred (structural-only cert) |

---

## Signals measured

1. **Changepoint alignment** — estimated first degradation period vs `drift_truth.changepoints`
2. **Fit degradation** — post/pre in-sample MAE ratio vs `expected_reliability.post_period_fit_degradation_min_ratio`
3. **KPI degradation** — relative level shift in target column across regimes
4. **Operational drift report** — `build_drift_report` with pre-period reference panel
5. **Calibration degradation** — replay staleness / missing units when replay active
6. **Readiness reaction** — governance + scorecard optimization approval vs truth

---

## Severity model (evidence-based)

| Level | Typical evidence | TrustReport effect |
|-------|------------------|-------------------|
| `none` | MAE ratio &lt; 1.05, no detected drifts | No downgrade |
| `minor` | Small shifts; decisions stable | Monitor only |
| `moderate` | MAE ratio ≥ 1.15 or drift_report `warning` | **Caution** — reduced decision confidence |
| `severe` | MAE ratio ≥ 1.35 or `critical` drift report | **Degraded** — optimization block recommended |

Bands use `TBD_v1_runtime` constants in `drift_detection_runner.py` until Phase **5F** Monte Carlo calibration.

---

## VAL-012 outcomes

| Outcome | Meaning | Certification `REC-4B5-DRIFT` status |
|---------|---------|--------------------------------------|
| `pass` | Detection and readiness match drift-world expectations | `pass` |
| `warning` | Partial alignment (e.g. changepoint offset) or weak severity on drift world | `pass` (scorecard may score `partial`) |
| `severe` | Missed detection or missing readiness downgrade | `fail` |

---

## Usage

```python
from pathlib import Path
from mmm.validation.synthetic.drift_detection_runner import run_drift_detection_for_bundle

result = run_drift_detection_for_bundle(Path("validation/worlds/WORLD-011-drift-recovery"))
print(result.val_012_outcome, result.drift_severity_level)
```

Recovery certification calls the runner automatically when `dgp:drift_recovery` is in scenario tags.

---

## Investigation status

**INV-055** (dedicated VAL-012 runner) — **closed** as of Phase 5E.
