# INV-H4D — Bayes-H4 sparse/τ tuning and recovery-candidate stability

| Field | Value |
|-------|--------|
| **Investigation ID** | INV-H4D |
| **Title** | Sparse/τ prior tuning and multi-seed stability for recovery_candidate worlds |
| **Status** | **Complete (report-only pilot)** |
| **Track** | Bayes-H4 research sandbox — follows H4c reliability map and INV-071 thresholds |
| **Pilot artifact** | [BAYES_H4D_SPARSE_TAU_STABILITY_20260601.json](../05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_20260601.json) |
| **Runner** | `mmm.research.bayes_h3_sandbox.h4d_sparse_tau_stability` |

---

## 1. Purpose

H4c answered *where* the sandbox behaves well or poorly across world **designs**. INV-H4D asks whether **true-effect recovery is stable across seeds** on **recovery_candidate** worlds and whether **τ prior tuning** improves sparse recovery without damaging clean recovery.

This is **not** a repeat of H4c. H4c worlds are held fixed; H4d varies **sampler seeds** and **`tau_channel_prior_sigma`** on a focused world set.

**Production Bayes remains blocked.**

---

## 2. Inputs / prerequisites

| Prerequisite | Role |
|--------------|------|
| H4c extended pilot | Reliability map — world roles already assigned |
| INV-071 threshold policy | Report-only `evaluate_world_against_policy()` outcomes |
| H4b-disposition C+A | Pooling vs true-effect separation |
| INV-H4-001 variant sweep | Sparse design context (optional diagnostic variants) |

---

## 3. Pilot design

| Dimension | Setting |
|-----------|---------|
| **Worlds** | `WORLD-BAYES-H4C-CLEAN-RECOVERY`, `WORLD-BAYES-H4C-SPARSE-RECOVERY`, `WORLD-BAYES-H4-SIMPLE-POOLING`, `WORLD-BAYES-H4-SPARSE-GEO` (stress only) |
| **τ grid** | default (0.5), 0.30, 0.20, 0.15 |
| **Seeds** | NUTS seeds 4400, 4401, 4402; panel_seed 4400 |
| **Sampler** | Fast profile for pilot JSON (`draws=200`, `tune=200`, `chains=2`); extended optional via CLI |

---

## 4. Research questions answered

### Are recovery_candidate worlds stable across seeds?

See artifact `aggregated_by_world_tau.*.stability` for `beta_gc_mae` and `mu_c_mae` (CV and spread across seeds at each τ).

- **CLEAN-RECOVERY** at default τ: primary stability signal for whether model/spec work is needed before tightening INV-071 bands.
- **SPARSE-RECOVERY** at default τ: whether sparse **recovery** claim can be discussed separately from sparse **stress**.

### Is true-effect recovery weak only in sparse/stress worlds?

- **SPARSE-GEO** (stress): elevated MAE expected — **report_only**, not global failure.
- **CLEAN-RECOVERY** / **SIMPLE-POOLING**: if unstable or high MAE across seeds, weakness is **not** confined to sparse stress.

### Does τ prior tuning improve sparse recovery without hurting clean recovery?

Compare `aggregated_by_world_tau` means for `tau_0.15` vs `default` on SPARSE-RECOVERY vs CLEAN-RECOVERY.

### Should sparse recovery and sparse stress remain separate world roles?

**Yes.** Policy roles from INV-071:

| Role | Worlds |
|------|--------|
| recovery_candidate | CLEAN-RECOVERY, SPARSE-RECOVERY, SIMPLE-POOLING |
| stress_diagnostic | WORLD-BAYES-H4-SPARSE-GEO |

SPARSE-RECOVERY is a **recovery candidate** with sparse geo layout; SPARSE-GEO is the original **stress** reference. Do not conflate them in gates or claims.

---

## 5. Metrics reported

| Metric | Role |
|--------|------|
| `beta_gc_mae`, `mu_c_mae` | True-effect point recovery |
| `beta_gc_coverage_90` | Directional uncertainty (not exact 90% on toys) |
| `beta_interval_width_90_mean` | Uncertainty sanity |
| `shrinkage_ratio_sparse` | Pooling mechanics — **not** a true-effect gate |
| `shrinkage_ratio_sparse_vs_true_mu` | Legacy diagnostic vs μ* |
| `policy_evaluation` | INV-071 report-only outcome |
| `h4c_classification` / `world_role` | Claim context |
| `warning_summary` | Count and types |

---

## 6. Recommended disposition (report-only)

Artifact field: `recommended_disposition.disposition`

| Disposition | Meaning |
|-------------|---------|
| `model_spec_work_needed_before_thresholds` | CLEAN-RECOVERY unstable across seeds |
| `keep_sparse_claim_restricted` | CLEAN stable but SPARSE-RECOVERY unstable |
| `recommend_tau_prior_sandbox_research_update` | τ=0.15 helps sparse recovery without hurting clean |
| `continue_report_only_monitoring` | No strong τ signal; keep INV-071 report-only |

**Never** convert INV-071 thresholds to hard gates in this step.

---

## 7. Report-only vs candidate future gate

| Item | H4d status |
|------|------------|
| `hard_gate` | **false** |
| `production_promotion` | **false** |
| `approved_for_prod` | **false** |
| CI merge fail on MAE | **Not authorized** |
| Future hard gates | Require extended MCMC + multi-seed stability **per recovery_candidate role** |

---

## 8. Why production remains blocked

- Toy synthetic panels and fast MCMC pilot profile
- No global truth-recovery claim
- τ recommendations apply to **sandbox research config** only
- Ridge production path unchanged

---

## 9. Open questions

1. Re-run H4d with extended sampler — do stability classifications change?
2. Include sparse diagnostic variants (more weeks, outlier severity) in a follow-on sweep?
3. When to promote τ=0.15 (or other) into default sandbox overrides for sparse worlds only?

---

## 10. Related work

| ID | Status |
|----|--------|
| H4c | Complete — reliability map |
| INV-071 | Complete — report-only thresholds |
| INV-H4D | This investigation |
| Bayes-H3 production promotion | **Blocked** |
