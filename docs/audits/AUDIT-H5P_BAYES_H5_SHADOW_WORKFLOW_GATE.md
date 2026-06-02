# AUDIT-H5P — Bayes-H5 Shadow Workflow Gate (Research Only)

**Audit ID:** AUDIT-H5P  
**Date:** 2026-06-01  
**Scope:** Bayes-H5l through Bayes-H5o — hierarchy-faithful convergence, frozen policy replay, recommender governance, two public frozen-panel shadow replays  
**Prerequisites:** H5o complete @ `e1de6d6`  
**ADR:** [bayes_h5_model_spec_improvement_adr.md](../05_validation/bayes_h5_model_spec_improvement_adr.md)  
**Verdict:** **Pass (research checkpoint)** — governed shadow workflow is reproducible on two distinct in-repo panels; **production Bayes promotion remains blocked**

---

## 1. Purpose

This audit is a **research-only checkpoint** after H5o. It:

1. Summarizes evidence from H5l, H5m, H5n, and H5o.
2. Defines **minimum eligibility** and **stop conditions** for any future real-panel shadow run.
3. Defines what counts as **usable research shadow evidence**.
4. States what **still blocks production promotion**.

It does **not** authorize production Bayes, optimizer use, DecisionSurface emission, budget recommendations, production TrustReport wiring, or Ridge replacement.

---

## 2. Evidence summary (H5l → H5o)

### Workflow arc

```text
H5l  find hierarchy-faithful converged config (σ_floor, NC, learned τ)
  → H5m  replay from frozen policy JSON (--policy-path)
  → H5n  recommender: diagnostics → governed policy + forbidden claims
  → H5o  second panel: recommender → freeze → replay (generalization test)
```

### H5l — Hierarchy-faithful geometry (INV-H5L)

| Field | Value |
|-------|--------|
| **Status** | Complete (research) |
| **Panel** | `examples/sample_panel.csv` only |
| **Key outcome** | **H5L-B**: first hierarchy-faithful `converged_diagnostic_only` with `sigma_floor=0.05`, non-centered parameterization, learned τ |
| **Artifact** | [BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json](../05_validation/archives/BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json) |
| **Governance note** | Geometry runner used collinear **heuristic** (dropped social); superseded by explicit drop-tv in H5m |

**Ablation benchmarks** (pooled, fixed-τ) may converge but are **not promotable** as shadow evidence.

### H5m — Frozen shadow-policy replay (INV-H5M)

| Field | Value |
|-------|--------|
| **Status** | Complete (research) |
| **Policy** | [h5m_sample_panel_shadow_policy.json](../06_investigations/h5m_sample_panel_shadow_policy.json) (`bayes_h5m_sample_panel_shadow_policy_v1`) |
| **Channel policy** | Governed **drop tv**, keep search + social (`no_silent_dropping=true`) |
| **Geometry** | NC full hierarchy, learned τ, `sigma_floor=0.05`, prescaled log outcome |
| **Replay** | `converged_diagnostic_only`, rhat_max ≈ 1.01, divergences = 0 |
| **Artifact** | [BAYES_H5M_SHADOW_POLICY_REPLAY_…](../05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) |

Proves successful H5L-B config is **policy-replayable**, not only an experiment-runner artifact.

### H5n — Shadow-policy recommender (INV-H5N)

| Field | Value |
|-------|--------|
| **Status** | Complete (research) |
| **Module** | `mmm/research/bayes_h3_sandbox/h5_shadow_policy_recommender.py` |
| **Function** | Maps collinearity, sparsity, convergence history, optional business metadata → explicit shadow policy recommendation |
| **Sample-panel output** | Recommends H5m drop-tv + σ-floor; records alternatives (composite, keep-all blocked), forbidden claims |
| **Artifact** | [BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_…](../05_validation/archives/BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) |

Closes the governance loop: **diagnose → recommend → freeze → run → record forbidden claims**.

### H5o — Second real-panel shadow (INV-H5O)

| Field | Value |
|-------|--------|
| **Status** | Complete (research) — **one panel only** |
| **Panel** | `examples/benchmark_geo_panel_v1.csv` (`examples_mmm_benchmark_geo_panel_v1`) |
| **Diagnostics** | Low collinearity (max \|ρ\| ≈ 0.06) |
| **Recommendation** | **keep_all_channels** + σ-floor hierarchy (contrasts H5m drop-tv) |
| **Policy** | [h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json](../06_investigations/h5o_examples_mmm_benchmark_geo_panel_v1_shadow_policy.json) |
| **Replay** | `converged_diagnostic_only`, rhat_max = 1.0, divergences = 0 |
| **Artifacts** | [Recommendation](../05_validation/archives/BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) · [Shadow run](../05_validation/archives/BAYES_H5O_SHADOW_RUN_EXAMPLES_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json) |

Proves recommender + frozen-policy workflow **generalizes** beyond the sample panel pilot.

### Two-panel comparison (research evidence)

| Panel | panel_id | Collinearity | Channel policy | Convergence |
|-------|----------|--------------|----------------|-------------|
| Sample | `examples_mmm_sample_panel_v1` | High (social–tv ~0.99) | Drop tv (governed) | converged_diagnostic_only |
| Benchmark | `examples_mmm_benchmark_geo_panel_v1` | Low (~0.06) | Keep all | converged_diagnostic_only |

---

## 3. Minimum eligibility — future real-panel shadow run

A panel may enter the H5 shadow workflow **only if all** of the following hold:

| # | Requirement |
|---|-------------|
| E1 | **Authorization** — New milestone + manifest (e.g. `H5*_…_MANIFEST.md`); **one panel per milestone**; no batching |
| E2 | **Environment** — `run_environment=research`, `enable_h5_sandbox=true`, `model_spec_version=bayes_h5_sandbox_spec_v1` |
| E3 | **Lineage** — Immutable `panel_id`, `dataset_snapshot_id`, `panel_path` or governed source reference; `data_snapshot_hash` on artifact |
| E4 | **Schema** — Known `geo_column`, week/date column, outcome column, `media_columns`; passes `validate_panel` at report severity |
| E5 | **Privacy** — No client-confidential or production-secret data in committed artifacts unless explicitly governed |
| E6 | **Purpose** — Historical / frozen snapshot; not a live planning or budget-decision panel |
| E7 | **Recommender first** — Run `h5_shadow_policy_recommender` (or successor) and write recommendation JSON **before** any MCMC |
| E8 | **Runnable recommendation** — `recommended_shadow_policy.status` is `recommended` (not `do_not_run`, not `requires_external_calibration` unless waived in writing) |
| E9 | **Frozen policy** — `research_shadow_policy` JSON validated via `validate_shadow_policy`; explicit `channel_policy`, `h5_geometry_config`, `sampler_profile`, `forbidden_claims` |
| E10 | **Geometry** — Hierarchy-faithful config only; **no** pooled/fixed-τ ablation as promotable policy |
| E11 | **Channels** — No silent drop/merge; explicit `dropped_channels` / `kept_channels` when mode is `drop_collinear_channels` |
| E12 | **Transforms** — Declared `transform_config` per H5 registry; transform mismatch documented |
| E13 | **Execution** — Shadow run via `--policy-path` only (no ad hoc CLI overrides that contradict frozen policy) |
| E14 | **Production flags** — All false on recommendation, policy, and shadow artifact |

**Preferred (not required for eligibility):** Ridge diagnostic contrast on same snapshot; GeoX/CLS/calibration metadata when available.

**Exclusions:** H4/H5 recovery worlds, toy fixtures labeled `dry_run_shadow_artifact`, stress-only SPARSE panels without separate authorization.

---

## 4. Stop conditions — do not run or do not promote

### 4.1 Hard stops (no shadow MCMC)

| Stop | Condition | Action |
|------|-----------|--------|
| **S1 — do_not_run** | Recommender `recommended_shadow_policy.status == do_not_run` | **Stop.** Fix panel, gather convergence evidence, or change governed remedy. Do not force fit. |
| **S2 — external calibration required** | `status == requires_external_calibration` and no calibration evidence | **Stop** unless investigation explicitly waives with documented rationale. |
| **S3 — missing lineage** | Missing `panel_id`, `dataset_snapshot_id`, or panel schema | **Fail closed** — no artifact |
| **S4 — missing / invalid policy** | No frozen policy, failed `validate_shadow_policy`, implicit channel drop | **Fail closed** — no run |
| **S5 — missing forbidden claims** | Empty `forbidden_claims` on recommendation or policy | **Fail closed** |
| **S6 — ablation promotable** | Frozen policy uses pooled/fixed-τ ablation geometry | **Reject policy** |
| **S7 — production flags true** | Any `approved_for_prod`, `prod_decisioning_allowed`, `hard_gate`, etc. | **Abort** |
| **S8 — forbidden outputs** | Artifact contains DecisionSurface, optimizer fields, budget recommendations | **Abort** — protocol violation |

CLI: recommender exits with code **2** when recommendation is not runnable.

### 4.2 Soft stops (run allowed; evidence limited)

| Stop | Condition | Action |
|------|-----------|--------|
| **S9 — failed convergence** | `convergence_status` not `converged_diagnostic_only` OR `rhat_max > 1.05` OR `divergence_count > 0` | Record honestly; set `evidence_promotion_allowed=false`; do not claim evidence-ready |
| **S10 — weak convergence only** | Partial sampler cooperation without clean diagnostics | Diagnostic-only; no promotion |
| **S11 — transform unknown** | `h5:transform_unknown:real_panel` and undeclared mismatch | Continue as diagnostic; document gap |
| **S12 — keep-all under collinearity** | High \|ρ\| but keep-all attempted | Block channel-level business claims; forbidden claims required |

### 4.3 Program stops (shadow program halt)

Stop the **shadow program** (not a single run) if:

- H5 output is wired to production TrustReport, optimizer, or DecisionSurface.
- Any artifact sets production promotion flags true.
- Panels are batched without per-panel manifests and audit trail.
- Silent channel collapse appears in policies or preprocessing.

---

## 5. Usable research shadow evidence

A shadow run counts as **usable research shadow evidence** when **all** apply:

| # | Criterion |
|---|-----------|
| U1 | `artifact_type=real_panel_shadow_artifact` on a documented historical panel (not synthetic fixture dry-run) |
| U2 | Complete lineage: `run_id`, `panel_id`, `dataset_snapshot_id`, `data_snapshot_hash`, `policy_id`, `source_policy_path` |
| U3 | Matching **frozen policy** applied: `channel_policy_applied` matches declared drop/keep; `h5_geometry_config_applied` matches policy |
| U4 | `convergence_status=converged_diagnostic_only` with `rhat_max <= 1.05` and `divergence_count == 0` |
| U5 | `hierarchy_faithful=true` geometry (σ-floor NC hierarchy acceptable; ablation-only geometry **not** usable) |
| U6 | `evidence_promotion_allowed=true` on artifact means **research eligibility only** — not production promotion |
| U7 | `forbidden_claims` and `interpretation_changes` recorded on recommendation + policy + artifact envelope |
| U8 | `trust_report_candidate_diagnostics` present (H5d mapping); outputs marked diagnostic-only |
| U9 | Prior recommender artifact exists for the same `panel_id` (or successor audit-documents equivalent) |

**Not sufficient alone:** single-panel success, fast MCMC, ablation convergence, illustrative benchmark panels without client context, Ridge agreement, or recommender `recommended` status before a converged replay.

---

## 6. Panel expansion criteria (H5p+)

Future panels are authorized **only** under this discipline:

| Rule | Detail |
|------|--------|
| **One panel per milestone** | Each expansion gets INV manifest + recommendation + policy + shadow artifact |
| **Distinct purpose** | New panel must test a **new governed situation** (e.g. collinearity class, sparsity, geo count, calibration availability) — not duplicate H5m/H5o |
| **Recommender-led** | No shadow run without prior recommendation JSON |
| **Contrast documentation** | Manifest must state how panel differs from `examples_mmm_sample_panel_v1` and `examples_mmm_benchmark_geo_panel_v1` |
| **No batching** | Parallel multi-panel campaigns require separate program approval |
| **Checkpoint** | Re-read this audit (or successor) before authorizing H5q+ |

**Current in-repo panels (frozen):**

| panel_id | Role | Shadow evidence |
|----------|------|-----------------|
| `examples_mmm_sample_panel_v1` | Collinear pilot | H5m replay ✅ |
| `examples_mmm_benchmark_geo_panel_v1` | Low-collinearity benchmark | H5o replay ✅ |

**Next expansion should prefer:** client-style historical slice with Ridge contrast and/or calibration stubs when governance allows — not a third duplicate illustrative CSV without new diagnostic value.

---

## 7. Production promotion — still blocked

The following remain **blocked** regardless of H5l–H5o success:

| Blocker | Detail |
|---------|--------|
| **Production Bayes** | No `approved_for_prod`, no `prod_decisioning_allowed`, no production TrustReport integration |
| **Optimizer / DecisionSurface** | Posterior must not feed simulate/optimize or budget recommendations |
| **Ridge replacement** | Ridge remains production path |
| **Hard gates** | INV-071 / Promotion Gate not satisfied for Bayes-H5 |
| **σ floor as prod default** | Research stabilization only |
| **Channel claims** | Dropped/composite/collinear channels — forbidden claims on artifact |
| **Ablation geometry** | Pooled/fixed-τ not promotion evidence |
| **Recommender output** | Recommendations are shadow-policy suggestions, not business decisions |
| **Two public panels** | Insufficient for prod candidacy; no signed Promotion Gate ADR |

**Required before any production candidacy review (H5 production promotion milestone):**

1. Promotion Gate ADR signed.  
2. TrustReport contract ADR for H5d field mapping in production.  
3. Shadow catalog with extended MCMC on **governed** client panels (not batch illustrative).  
4. Shadow vs Ridge disagreement review.  
5. False-positive review on H5 warning taxonomy.  
6. Explicit decision on DecisionSurface draw vs mean policy.

---

## 8. Audit checklist (pre-authorize next panel)

| Check | Pass? |
|-------|-------|
| Manifest written with lineage, caveats, production boundary | |
| Recommender artifact written; status ≠ `do_not_run` | |
| Frozen policy validates; forbidden claims non-empty | |
| `--policy-path` replay only | |
| Convergence recorded honestly on artifact | |
| Production flags false; no DecisionSurface/optimizer/recommendations | |
| Investigation doc answers panel selection and comparators | |
| This audit (AUDIT-H5P) referenced in milestone | |

---

## 9. References

| Doc | Link |
|-----|------|
| H5 ADR | [bayes_h5_model_spec_improvement_adr.md](../05_validation/bayes_h5_model_spec_improvement_adr.md) |
| H5e protocol | [INV-H5E](../06_investigations/INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md) |
| H5n recommender | [INV-H5N](../06_investigations/INV-H5N_SHADOW_POLICY_RECOMMENDER.md) |
| H5o second panel | [INV-H5O](../06_investigations/INV-H5O_SECOND_REAL_PANEL_SHADOW_RUN.md) |
| Roadmap registry | [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md) |

---

## 10. Conclusion

**H5l–H5o establish a governed research shadow loop** with two converged frozen-panel replays under distinct channel policies. The workflow is fit for **controlled, one-at-a-time panel expansion** in research.

**Production Bayes remains blocked.** No optimizer, DecisionSurface, recommendations, production TrustReport, or Ridge replacement.
