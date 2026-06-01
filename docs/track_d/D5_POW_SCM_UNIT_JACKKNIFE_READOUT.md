# D5-POW — SCM+UnitJackknife power / null-monitor readout characterization

| Field | Value |
|-------|--------|
| **Investigation ID** | **D5-POW** |
| **Lane** | Track D research (MMM repo sandbox — not GeoX production) |
| **Status** | Complete (report-only) |
| **Results** | [archives/D5_POW_results.json](archives/D5_POW_results.json) |
| **Code** | `mmm.research.track_d.d5_pow` |

---

## 1. Purpose

After **D5-POW-001a** showed that TBRRidge+KFold MDE is an optimistic proxy for SCM+UnitJackKnife (SCM+JK) readout feasibility, **D5-POW** characterizes SCM+JK **after the UnitJackKnife target fix** on unit-level panels.

**Stop question:** Does SCM+JK support power/MDE interpretation, **only** null-monitor interpretation, or a **different readout-aligned** power metric?

---

## 2. Scope (research only)

| In scope | Out of scope |
|----------|----------------|
| Point-effect recovery vs injection grid | Production power analysis changes |
| Null-monitor behavior at zero injection | Estimator / inference code changes |
| Interval-excludes-zero as detection criterion | TrustReport / CalibrationSignal eligibility |
| False-positive rate at null | MMM / planning / optimizer feeds |
| Pooled vs per-replicate summaries | Promoting SCM+JK to lift detection |

---

## 3. Method

- **Estimator:** Research-only SCM weights (pre-period level match) + **unit jackknife** on control units (post-fix target = unit-level SCM lift).
- **Panel:** Synthetic unit-level series (`n_control=10`, pre/post windows).
- **Grid:** Injections `0, 0.02, 0.05, 0.10, 0.15, 0.20`; 12 replicate seeds per level.

---

## 4. Findings (see JSON for numbers)

| Question | Answer |
|----------|--------|
| **Point-effect recovery** | **Yes** — correlation(injection, mean point) ≈ 1.0 |
| **Null-monitor (point near zero)** | **Yes** — at null, \|point\| small; suitable for monitoring only |
| **Interval excludes zero as detection** | **No** — not a valid power/MDE criterion; low power at small positives, OR-pooling inflates apparent detection |
| **001a 100% null detection degeneracy** | **Not fully replicated** under corrected JK + this sandbox SCM; 001a likely pooled/wrong readout — see `comparison_to_d5_pow_001a` |
| **Power / MDE for SCM+JK** | **Not supported** — use readout-aligned metric if needed later |

---

## 5. Recommended disposition

```text
scm_jk_supports_power_mde_interpretation: false
scm_jk_supports_null_monitor_only: true
requires_different_readout_aligned_power_metric: true
```

**Production Bayes / GeoX promotion:** blocked (unchanged).

---

## 6. Related artifacts

| Artifact | Role |
|----------|------|
| [D5_POW_001a_reference.json](archives/D5_POW_001a_reference.json) | 001a summary for comparison |
| External `TRACK_D_D4_POWER_MDE_AUDIT_001.md` | GeoX program (not in MMM repo) |

---

## 7. Re-run

```bash
poetry run python -m mmm.research.track_d.d5_pow
```
