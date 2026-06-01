# TrustReport semantics (synthetic reliability)

**Module:** `mmm/validation/synthetic/trust_report_semantics.py`  
**Governance:** [reliability_threshold_governance.md](reliability_threshold_governance.md)  
**Scorecard:** [reliability_scorecard.md](reliability_scorecard.md)

---

## Purpose

**TrustReport** is the platform contract for expressing **business-facing reliability** — not raw coefficient error. Phase **5C** showed decision recovery (Δμ) can succeed when attribution recovery fails; Phase **5D** split metric classes; Phase **5E** connects **drift severity** and other trust modifiers to a unified interpretation object attached to the ReliabilityScorecard.

Production `TrustReport` fields in train/decide extensions are separate but should follow the same semantic split: **decision-grade** vs **diagnostic** vs **trust modifiers**.

---

## Primary vs diagnostic signals

| Signal class | Used for TrustReport grade? | Examples |
|--------------|----------------------------|----------|
| **Decision-grade** | **Yes** | `decision_reliability_score`, VAL-004/005/006 |
| **Structural** | **Yes** (gate) | `structural_reliability_score`, contracts |
| **Trust modifiers** | **Yes** (downgrade) | Drift, identifiability, replay, freshness |
| **Attribution diagnostic** | **No** (unless attribution profile) | `attribution_diagnostic_score`, VAL-001/002/003 |

**Rule:** Do not block decision usability solely because `attribution_diagnostic_score` is low.

---

## `trust_modifier_status` (scorecard)

Object on `synthetic_reliability_scorecard.json`:

| Field | Meaning |
|-------|---------|
| `status` | `acceptable` \| `caution` \| `degraded` \| `not_evaluated` |
| `min_score` | Minimum scored trust-modifier capability mean |
| `drift_severity_max` | Max `drift_severity_level` from VAL-012 runner across worlds |
| `warnings` | Machine-readable downgrade reasons |
| `failures` / `partials` | Check IDs that failed or ran partial |

**Degraded when:** trust capability failures, min score &lt; 0.5, or **severe** drift on any evaluated world.

**Caution when:** partial execution, moderate drift, or min score &lt; 1.0.

---

## Modifier inputs

| Modifier | Source | Severe trigger (synthetic) |
|----------|--------|----------------------------|
| **Drift** | VAL-012 / `REC-4B5-DRIFT` | `drift_severity_level: severe` |
| **Identifiability** | WORLD-012 / collinearity | Capability failures, VIF warnings |
| **Replay** | VAL-006 | Replay lift failure on replay worlds |
| **Freshness** | VAL-007 / calibration | Stale replay without loss (drift report) |
| **Optimizer stability** | VAL-005 | Optimizer recovery fail |

---

## `trust_report_interpretation` (scorecard)

Unified rollup produced by `build_trust_report_interpretation()`:

| Field | Meaning |
|-------|---------|
| `trust_grade` | `high` \| `moderate` \| `low` \| `insufficient` |
| `decision_usable` | May recommendations proceed on decision surface? |
| `attribution_safe` | Safe to claim channel attribution parity? |
| `optimization_blocked` | Should optimization gates block? |
| `release_gate_recommendation` | `block` \| `warn` \| `conditional_not_approved` |
| `primary_signals` | Decision + structural scores (not attribution-first) |
| `modifier_signals` | Per-modifier status |
| `interpretation_matrix` | Human-readable decision table |

---

## Interpretation matrix

| Condition | Decision usable? | Attribution safe? | Optimization blocked? | Release gate |
|-----------|------------------|-------------------|-------------------------|--------------|
| Structural fail | No | No | Yes | Block |
| Decision high, trust none/minor | Yes | Only if attribution diagnostic high | No | Conditional (thresholds not approved) |
| Decision high, coef diagnostic low | Yes | **No** | No | Conditional |
| Trust moderate (drift/ID) | Caution | No | Conditional | Warn |
| Trust degraded / severe drift | No / heavy caution | No | **Yes** | Block or warn |
| Decision grade low | No | No | Yes | Block |

---

## FAQ (acceptance criteria)

### When can a model be decision-usable despite poor coefficient recovery?

When `decision_reliability_score` is high, `structural_reliability_score` passes, `trust_modifier_status` is not `degraded`, and VAL-004/005/006 pass under approved thresholds. Low `attribution_diagnostic_score` only implies **attribution-unsafe** messaging.

### When should drift downgrade TrustReport?

When VAL-012 reports `drift_severity_level` of **moderate** (caution) or **severe** (degraded), or when post-period fit/KPI degradation indicates regime change without acceptable readiness reaction.

### When should optimization be blocked?

When `optimization_blocked` is true in `trust_report_interpretation` — typically **severe** drift, structural failure, or low decision-grade score. Moderate drift may block only when world truth expects readiness downgrade.

### How does TrustReport relate to release gates?

Release gates (`approved_for_prod`, promotion, VAL-008/011) are **structural + governance**. TrustReport adds **modifier downgrades**; synthetic scorecard never sets production approval while `TBD_v1_runtime` is provisional.

### Which failures are attribution-only vs decision-critical?

| Failure | Class |
|---------|--------|
| VAL-001 / 002 / 003 | Attribution-only (default) |
| VAL-004 / 005 / 006 | Decision-critical |
| VAL-012 / identifiability | Trust modifier (can block despite Δμ pass) |
| CERT-4A / VAL-008 / 009 | Structural (blocking) |

---

## Related

- [drift_detection.md](drift_detection.md)  
- [exact_recovery_investigation_report.md](exact_recovery_investigation_report.md)
