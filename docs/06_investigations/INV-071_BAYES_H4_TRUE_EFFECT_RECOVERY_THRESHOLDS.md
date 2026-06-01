# INV-071 — Bayes-H4 true-effect recovery threshold calibration

| Field | Value |
|-------|--------|
| **Investigation ID** | INV-071 |
| **Title** | Claim-specific true-effect recovery thresholds for Bayes-H4 research sandbox |
| **Status** | **Calibrated (report-only)** — hard gates and production promotion **deferred** |
| **Track** | Bayes-H4 research recovery — Research Sandbox only |
| **Policy artifact** | [BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json](../05_validation/archives/BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json) |
| **Implementation** | `mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy` |

---

## 1. Purpose

H4c produced a **reliability map**: the sandbox recovers synthetic truth under some conditions and warns or degrades under others. INV-071 turns that map into **claim-specific threshold policy** for true-effect recovery metrics.

This investigation does **not** answer: *Is Bayesian MMM ready for production?*

It answers: *Under which world roles should we pass, warn, restrict, or report-only when judging true-effect recovery on toy synthetic panels?*

---

## 2. Inputs / artifacts

| Artifact | Use |
|----------|-----|
| [BAYES_H4_THRESHOLD_PILOT_20260601.json](../05_validation/archives/BAYES_H4_THRESHOLD_PILOT_20260601.json) | H4a fast pilot per core world |
| [BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json](../05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json) | H4b extended multi-seed; primary vs legacy shrinkage |
| [BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json](../05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json) | H4c reliability map (primary calibration source) |
| [BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json](../05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json) | Sparse diagnostic context (not global gates) |

Prerequisites: [INV-H4-001](INV-H4-001_SPARSE_POOLING_BEHAVIOR.md) disposition **C+A** (pooling vs recovery separation).

---

## 3. World classification policy (claim-specific)

**Do not apply one global threshold across all worlds.**

| Policy role | Worlds | Gate posture |
|-------------|--------|--------------|
| **recovery_candidate** | `WORLD-BAYES-H4C-CLEAN-RECOVERY`, `WORLD-BAYES-H4C-SPARSE-RECOVERY`, `WORLD-BAYES-H4-SIMPLE-POOLING` | Calibrate report warn/restricted bands; `pass` / `warn` / `restricted` / `fail_for_claim` (report-only) |
| **stress_diagnostic** | `WORLD-BAYES-H4-SPARSE-GEO` | **report_only** — never global model failure |
| **weak_identification** | `WORLD-BAYES-H4C-CORRELATED-CHANNELS`, `WORLD-BAYES-H4C-WEAK-SIGNAL` | **warn** / **restricted** expected; not recovery failure |
| **transform_mismatch** | `WORLD-BAYES-H4C-ADSTOCKED-MEDIA`, `WORLD-BAYES-H4C-SATURATION` | **restricted** + mismatch warnings; not recovery failure |
| **conflict_diagnostic** | `WORLD-BAYES-H4-CONFLICTING-EVIDENCE` | Conflict warnings **required**; not a point-recovery gate |

---

## 4. Metric roles

### Pooling mechanics (not true-effect gates)

| Metric | Role |
|--------|------|
| `shrinkage_ratio_sparse` | vs posterior \(\hat\mu_c\) — pooling mechanics only (H4b-disposition C) |
| `shrinkage_ratio_sparse_vs_true_mu` | Legacy diagnostic vs \(\mu_c^\*\) — **not** a pooling or promotion gate |

### True-effect recovery (calibrated on recovery_candidate worlds only)

| Metric | Role |
|--------|------|
| `beta_gc_mae` | Point recovery of geo-channel effects |
| `mu_c_mae` | Point recovery of channel hyper-means |
| `beta_gc_coverage_90` | **Directional only** on toy panels — do not require exact 90% |
| `beta_interval_width_90_mean` | Uncertainty sanity — wide intervals expected under weak ID |

### Reliability diagnostics

| Signal | Role |
|--------|------|
| `h4c_classification` | Reliability map label from H4c pilot |
| `h4c_diagnostic_warnings` | Collinearity, transform mismatch, weak ID |
| `conflict_warnings` | Conflict world diagnostic contract |

---

## 5. Provisional report-only threshold bands

Derived from **recovery_candidate** observations across H4a/H4b/H4c artifacts (fast + extended MCMC mixes). See JSON for numeric values.

| Metric | Policy |
|--------|--------|
| `beta_gc_mae` | `report_warn_above` ≈ max(observed) × 1.15; `report_restricted_above` ≈ max × 1.35; **no hard_fail** |
| `mu_c_mae` | Same banding pattern on recovery_candidate runs only |
| `beta_gc_coverage_90` | Directional monitoring; observed often low on toy panels |
| Interval width | Sanity `report_restricted_above` when calibrated; weak-ID worlds exempt from global fail |

**H4c pilot snapshot (recovery candidates, fast MCMC):**

| World | `beta_gc_mae` | `mu_c_mae` |
|-------|---------------|------------|
| CLEAN-RECOVERY | ≈ 0.30 | ≈ 0.28 |
| SPARSE-RECOVERY | ≈ 0.27 | ≈ 0.26 |
| SIMPLE-POOLING (H4) | ≈ 0.29–0.47 | ≈ 0.28–0.39 |

Stress world `WORLD-BAYES-H4-SPARSE-GEO` shows **higher** MAE — excluded from calibration bands by design.

---

## 6. Report-only vs future gate

| Item | Status |
|------|--------|
| `hard_gate` | **false** |
| `production_promotion` | **false** |
| `approved_for_prod` | **false** |
| Merge / CI fail on threshold exceedance | **Not authorized** |
| Future hard gates | Require repeated stable evidence across seeds **per role**, not stress/mismatch worlds |

Gate outcomes (report-only vocabulary):

- **pass** — within warn band (recovery_candidate only)
- **warn** — exceeds warn band or expected diagnostic condition
- **restricted** — exceeds restricted band or transform/weak-ID role
- **report_only** — stress diagnostic worlds
- **fail_for_claim** — reserved; not enabled in v1 policy

---

## 7. Why production remains blocked

| Reason | Detail |
|--------|--------|
| Toy synthetic panels | Thresholds are not calibrated on client data |
| No global truth claim | Favorable worlds recover partially; mismatch worlds fail by design |
| Pooling ≠ recovery | Primary shrinkage can pass while \(\mu_c^\*\) recovery is weak |
| INV-071 v1 | Report-only policy; no DecisionSurface / optimizer / recommendations |
| Ridge path | Production MMM remains Ridge per roadmap |

---

## 8. Open questions

1. Re-calibrate bands after sparse-world / \(\tau\) tuning (disposition A).
2. Whether extended-MCMC-only bands differ materially from fast pilot (re-run aggregation).
3. When to enable `fail_for_claim` on recovery_candidate worlds only (needs multi-seed stability).
4. How INV-071 bands relate to future production TrustReport mapping (out of scope).

---

## 9. Related investigations

| ID | Status |
|----|--------|
| INV-H4-001 | Closed (C+A) |
| INV-H4-001b | Closed |
| INV-071 | This document — report-only policy calibrated |
