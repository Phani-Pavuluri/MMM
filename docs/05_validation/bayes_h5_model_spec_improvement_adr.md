# Bayes-H5 — Sandbox Model-Spec Improvement

**ADR ID:** `bayes_h5_model_spec_improvement_v1`  
**Title:** Bayes-H5 — Next research sandbox hierarchical model specification (architecture only)  
**Status:** **Accepted** (architecture only — does **not** authorize implementation, PyMC changes, production promotion, or hard gates)  
**Date:** 2026-06-01  
**Accepted:** 2026-06-01  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisites:** [bayes_h2d_hierarchical_model_spec_adr.md](bayes_h2d_hierarchical_model_spec_adr.md) · [bayes_h3_research_sandbox_backend_adr.md](bayes_h3_research_sandbox_backend_adr.md) · [bayes_h4_recovery_worlds_adr.md](bayes_h4_recovery_worlds_adr.md) · Bayes-H4c reliability map ✅ · INV-071 report-only thresholds ✅ · [INV-H4D](../06_investigations/INV-H4D_SPARSE_TAU_AND_RECOVERY_STABILITY.md) (H4d extended MCMC confirmed) ✅  
**Governance:** [ROADMAP_ALIGNMENT_GATE.md](../ROADMAP_ALIGNMENT_GATE.md) · [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md)  
**Supersedes (partial):** H3 MVP generative detail in `mmm/research/bayes_h3_sandbox/model.py` — **only after** separate implementation approval and H5 validation worlds pass

---

## Acceptance

| Field | Value |
|-------|--------|
| **Decision** | **Accepted** — model-spec architecture for next sandbox version |
| **Acceptance date** | 2026-06-01 |
| **Rationale** | Aligns with H3/H4 evidence (H4c reliability map, H4d extended MCMC, INV-071 report-only policy); defines transform registry, prior/diagnostic policy, validation worlds, and promotion boundaries without authorizing code or production paths |
| **Does not authorize** | `model.py` changes, PyMC implementation, hard gates, DecisionSurface/optimizer/recommendations, `approved_for_prod`, Ridge replacement |
| **Next authorized step** | H5 validation world catalog + gated sandbox implementation + `BAYES_H5_RECOVERY_PILOT_*` artifacts (research-only) |

### Implementation pointer (H5a — research sandbox only)

| Field | Value |
|-------|--------|
| **Status** | **H5a complete (research)** — gated implementation + fast MCMC pilot; **not** production acceptance |
| **Code** | `mmm/research/bayes_h3_sandbox/h5_validation_worlds.py` · `h5_transforms.py` · `h5_pilot_runner.py`; H5 path in `model.py` / `entrypoint.py` / `fencing.py` (`model_spec_version=bayes_h5_sandbox_spec_v1`, `enable_h5_sandbox=True`) |
| **Pilot artifact** | [archives/BAYES_H5_SANDBOX_PILOT_20260601.json](archives/BAYES_H5_SANDBOX_PILOT_20260601.json) |
| **Still blocked** | Production Bayes; Ridge prod path; DecisionSurface; optimizer; recommendations; `approved_for_prod`; hard gates |

### H5a pilot execution status

| Field | Value |
|-------|--------|
| **Full pilot executed** | **Yes** — PyMC 5.28.5, `hasattr(pm, "Model")` true (2026-06-01 run) |
| **Command** | `poetry run python -c "from mmm.research.bayes_h3_sandbox.h5_pilot_runner import run_h5_pilot; run_h5_pilot(fast_mcmc=True)"` |
| **Sampler** | PyMC NUTS; `draws=200`, `tune=200`, `chains=2`, `target_accept=0.92`, `fast_mcmc_profile=true` |
| **Artifact** | [BAYES_H5_SANDBOX_PILOT_20260601.json](archives/BAYES_H5_SANDBOX_PILOT_20260601.json) — all 7 H5 worlds with per-world `beta_gc_mae` / `mu_c_mae` |
| **Aligned vs H4c mismatch baselines** | **Adstock:** H5-ADSTOCK-ALIGNED `beta_gc_mae≈0.264` vs H4c-ADSTOCKED-MEDIA `≈0.279` (modest gain). **Saturation:** H5-SATURATION-ALIGNED `≈0.113` vs H4c-SATURATION `≈0.171` (clear gain). Intentional mismatch worlds match or echo H4c poor recovery (`ADSTOCK-MISMATCH≈0.279`). |
| **Mismatch diagnostics** | **As expected** — `h5:transform_mismatch` on ADSTOCK-MISMATCH and SATURATION-MISMATCH; policy `transform_mismatch_warning`; `hard_gate=false`. |
| **Weak-ID diagnostics** | **As expected** — collinearity warning on CORRELATED-CHANNELS (`max_channel_corr≈0.95`); weak-ID policy outcome on CORRELATED / WEAK-SIGNAL. |
| **Implementation defects** | **None blocking spec** — minor follow-up: `transforms_aligned()` does not treat `linear`/`correlated`/`weak_signal` generative kinds as aligned with `identity` fit, causing benign `h5:unexpected_transform_mismatch` on non-transform probe worlds (diagnostic polish only). |
| **Production** | **Remains blocked** — pilot is research-only; no optimizer, DecisionSurface, or promotion flags. |

### H5b pilot execution status (diagnostic polish + repeated pilot)

| Field | Value |
|-------|--------|
| **Status** | **Complete (H5b)** — diagnostic alignment fix + 3-seed repeated fast MCMC |
| **Investigation** | [INV-H5B](../06_investigations/INV-H5B_REPEATED_PILOT_AND_DIAGNOSTICS.md) |
| **Repeated artifact** | [BAYES_H5B_REPEATED_PILOT_20260601.json](archives/BAYES_H5B_REPEATED_PILOT_20260601.json) |
| **Diagnostic fix** | `transforms_aligned()` recognizes linear/correlated/weak_signal + identity; no false `unexpected_transform_mismatch` |
| **Stability** | Saturation-aligned Δ≈−0.057 vs H4c stable across seeds; adstock-aligned Δ≈−0.015 stable |
| **Mismatch warnings** | 100% rate on intentional mismatch worlds |
| **Production** | **Still blocked** |

### H5c extended MCMC confirmation (INV-H5C)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — extended repeated pilot confirms H5b |
| **Investigation** | [INV-H5C](../06_investigations/INV-H5C_EXTENDED_MCMC_CONFIRMATION.md) |
| **Artifact** | [BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json](archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json) |
| **Sampler** | 600 tune / 600 draw / 4 chains / target_accept=0.95 |
| **Outcome** | All H5b conclusions hold; no material Δ vs H5b (\|Δ\| &lt; 0.05); H5 transform evidence **accepted for research** |
| **Production** | **Still blocked** |

### H5d TrustReport diagnostic mapping (INV-H5D — research only)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — candidate field catalog + warning taxonomy; **not** production TrustReport wiring |
| **Investigation** | [INV-H5D](../06_investigations/INV-H5D_TRUST_DIAGNOSTIC_MAPPING.md) |
| **Code** | `mmm/research/bayes_h3_sandbox/h5_trust_diagnostics.py` |
| **Artifact** | [BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json](archives/BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json) |
| **Production** | **Still blocked** — no optimizer, DecisionSurface, or prod TrustReport |

### H5e real-panel shadow-run protocol (INV-H5E — design only)

| Field | Value |
|-------|--------|
| **Status** | **Protocol defined** — no authorized shadow execution in H5e deliverable |
| **Investigation** | [INV-H5E](../06_investigations/INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL.md) |
| **Schema** | [BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json](archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json) |
| **Code** | `mmm/research/bayes_h3_sandbox/h5_shadow_protocol.py` |
| **Production** | **Still blocked** — Ridge unchanged; H5 not wired to prod pipelines |

### H5f real-panel shadow-run harness (INV-H5F — research execution)

| Field | Value |
|-------|--------|
| **Status** | **Harness complete** — fixture dry-run + CLI; real-panel runs require separate authorization |
| **Investigation** | [INV-H5F](../06_investigations/INV-H5F_REAL_PANEL_SHADOW_RUN_HARNESS.md) |
| **Code** | `mmm/research/bayes_h3_sandbox/h5_shadow_runner.py` |
| **Dry-run artifact** | [BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json](archives/BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json) |
| **Production** | **Still blocked** — no prod TrustReport, optimizer, DecisionSurface, or Ridge replacement |

### H5g first real-panel shadow run (INV-H5G — research evidence)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — first `real_panel_shadow_artifact` on in-repo `examples/sample_panel.csv` |
| **Investigation** | [INV-H5G](../06_investigations/INV-H5G_FIRST_REAL_PANEL_SHADOW_RUN.md) |
| **Manifest** | [H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md](../06_investigations/H5G_FIRST_REAL_PANEL_SHADOW_RUN_MANIFEST.md) |
| **Artifact** | [BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](archives/BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) |
| **Production** | **Still blocked** — illustrative panel only; not decision grade |

### H5h real-panel shadow hardening (INV-H5H)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — real-panel transform semantics + convergence classes + extended MCMC CLI |
| **Investigation** | [INV-H5H](../06_investigations/INV-H5H_REAL_PANEL_SHADOW_HARDENING.md) |
| **Artifact** | [BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json](archives/BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) |
| **Outcome** | Sample panel still `failed_convergence` — do not batch more panels until fixed |
| **Production** | **Still blocked** |

### H5i real-panel convergence diagnostics (INV-H5I)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — static + posterior diagnostics; controlled experiment matrix on sample panel only |
| **Investigation** | [INV-H5I](../06_investigations/INV-H5I_REAL_PANEL_CONVERGENCE_DIAGNOSTICS.md) |
| **Artifacts** | [H5I diagnostics JSON](archives/BAYES_H5I_CONVERGENCE_DIAGNOSTICS_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json) · [H5I experiment matrix](archives/BAYES_H5I_CONVERGENCE_EXPERIMENT_MATRIX_20260601.json) |
| **Outcome** | Collinearity + hierarchical geometry primary; scaling/single-channel probes help but do not pass evidence bar |
| **Production** | **Still blocked** |

### H5j collinearity geometry ablations (INV-H5J)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — explicit `channel_policy` + sample-panel geometry ablation matrix |
| **Investigation** | [INV-H5J](../06_investigations/INV-H5J_COLLINEARITY_GEOMETRY_ABLATIONS.md) |
| **Artifact** | [BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json](archives/BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json) |
| **Outcome** | Drop-collinear + prescale + extended → **weak_convergence** only; pilot not eligible for more panels |
| **Production** | **Still blocked** |

### H5k geometry stabilization (INV-H5K)

| Field | Value |
|-------|--------|
| **Status** | **Complete** — explicit `h5_geometry_config` (parameterization, likelihood scale, hierarchy) + sample-panel stabilization matrix |
| **Investigation** | [INV-H5K](../06_investigations/INV-H5K_GEOMETRY_STABILIZATION.md) |
| **Artifact** | [BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json](archives/BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json) |
| **Outcome** | Full hierarchy still weak (4 div); **pooled** and **fixed-τ** ablations → `converged_diagnostic_only` on pilot — research eligibility only |
| **Production** | **Still blocked** |

---

## Roadmap alignment gate (pre-authoring)

| Gate question | Answer |
|---------------|--------|
| **Tier** | 1 — Architecture / model spec |
| **MIP goal** | MMM calibration ecosystem; trust-aware measurement (research sandbox) |
| **Contracts touched** | TrustReport (diagnostic fields only); **not** DecisionSurface, optimizer, or prod CalibrationSignal eligibility |
| **Failure mode reduced** | False recovery claims from transform mismatch; weak-ID misread as “ready”; sparse stress conflated with recovery |
| **Proof artifact** | This ADR; future `WORLD-BAYES-H5-*` + H5 pilot JSON (not in scope of this deliverable) |
| **Gate level** | Architecture Gate |
| **Research allowed?** | Yes — design-only |
| **Production promotion** | **Blocked** |

---

## 1. Purpose

Bayes-H3 proved the sandbox runs safely. Bayes-H4 (H4a–H4d, INV-071, INV-H4D) mapped **where** the current MVP hierarchical spec recovers synthetic truth and where it fails.

**Bayes-H5** defines the **next sandbox model specification** (`bayes_h5_sandbox_spec_v1` conceptually) that:

1. Aligns the **media transform layer** with generative truth on transform-aligned worlds.
2. Surfaces **weak identification** and **transform mismatch** as first-class diagnostics (not silent failure).
3. Separates **sparse recovery** from **sparse stress** in priors, worlds, and TrustReport language.
4. Preserves **Bayes-H1/H2/H2b** contracts: no production DecisionSurface, no optimizer on posteriors, CalibrationSignal-only evidence ingress when used.

This ADR is **design-only**. Implementation requires a follow-on authorization and H5 validation program.

---

## 2. Current H3/H4 evidence summary

| Finding | Source | Implication for H5 |
|---------|--------|-------------------|
| **H3 sandbox safety** (fences, `run_sandbox_fit`, research-only labels) | Bayes-H3 MVP + guardrails | Preserve entrypoint and fencing; H5 is spec evolution inside same lane |
| **Pooling mechanics** pass (primary shrinkage vs posterior \(\hat\mu_c\)) | H4b-refresh, INV-H4-001 C+A | Keep partial-pooling structure; fix readouts, not abandon hierarchy |
| **True-effect recovery** open on stress; partial on favorable worlds | H4c map, INV-071 | Recovery metrics stay separate from pooling; claim-specific thresholds remain report-only |
| **CLEAN / SPARSE-RECOVERY / SIMPLE-POOLING** stable across seeds (extended MCMC) | H4d extended | Favorable worlds are valid regression targets for H5 spec |
| **SPARSE-GEO** elevated MAE; stress-only | H4c, H4d, INV-071 | Do not use stress world for τ tuning or promotion; keep `stress_diagnostic` role |
| **τ=0.15** no material sparse-recovery gain without clean harm | H4d | τ grid alone insufficient; transform + weak-ID policy required |
| **ADSTOCKED / SATURATION** worlds → transform_mismatch | H4c | MVP semi_log on raw standardized media is wrong generative match |
| **CORRELATED / WEAK-SIGNAL** → weak_identification | H4c | Expect warn/restricted; not global sandbox failure |
| **Interval-excludes-zero** not valid power proxy | D5-POW (Track D lane, separate) | H5 does not adopt detection-style gates from intervals |

**Production Bayes remains blocked.** Ridge remains the production estimator.

---

## 3. Non-goals

| Non-goal | Notes |
|----------|--------|
| Production Bayesian MMM | Ridge prod path unchanged |
| `approved_for_prod: true` or `prod_decisioning_allowed: true` | All H5 work research-labeled until Promotion Gate |
| DecisionSurface / optimizer / budget recommendations | [Bayes-H1](bayes_h1_decision_surface_preservation_adr.md) |
| Hard CI gates or merge-blocking recovery thresholds | INV-071 remains report-only until repeated H5 evidence |
| Replacing `WORLD-BAYES-*` H2b evidence-routing worlds | H2b orthogonal; H5 extends **generative** recovery catalog |
| NumPyro / second backend in H5 v1 | Per [bayes_h3_research_sandbox_backend_adr.md](bayes_h3_research_sandbox_backend_adr.md); backend ADR amendment later if needed |
| Claiming global true-effect recovery | H5 targets **classified worlds** only |

---

## 4. Model-spec changes proposed

| Area | H3 MVP (current) | H5 proposed (sandbox) |
|------|------------------|------------------------|
| Media enters likelihood | Standardized raw \(x_{g,t,c}\) | **Declared transform pipeline** per channel (registry-driven) |
| Transform truth | Implicit semi_log on \(y\) only | **Explicit** adstock/saturation options matching H4c generative kinds |
| Weak ID | Implicit in partial pooling | **Collinearity + SNR diagnostics**; optional ridge on \(z\) or tighter \(\tau\) policy |
| Sparse geo | Single stress world | **Role split**: `sparse_recovery` vs `sparse_stress`; geo-specific pooling diagnostics |
| Priors | Fixed \(\tau\) HalfNormal(0.5) override only | **Policy table**: by world role and channel sparsity |
| Recovery vs pooling metrics | Separated (H4b-disposition) | **Frozen** — same metric families, H5 adds transform/weak-ID fields |
| TrustReport | Diagnostic stub in sandbox | **Structured diagnostic blocks** (warn/restricted); no prod promotion fields |

**Versioning:** Sandbox artifacts label `sandbox_model_spec: bayes_h5_sandbox_spec_v1` when implemented; until then H3 MVP remains the running code path.

---

## 5. Media transform layer

### 5.1 Design principle

The likelihood must consume **the same functional transform** applied to media that generated the outcome on **transform-aligned** worlds. On **transform-mismatch negative** worlds, the spec deliberately fits the **wrong** transform to validate diagnostic behavior.

### 5.2 Adstock options

| ID | Definition (per channel \(c\), national parameters in v1) | Use |
|----|----------------------------------------------------------|-----|
| `media_raw_std` | Standardize raw spend (H3 MVP) | Baseline / negative control for mismatch worlds |
| `media_geometric_adstock` | Geometric decay with learnable or fixed \(\theta_c\) | Align with H4c `generative_kind: adstock` |
| `media_fixed_adstock` | Fixed \(\theta_c\) from world `generative_params` | Recovery worlds with known truth |

**H5 v1 rule:** Adstock parameters are **national per channel** unless a validation world proves geo-specific identifiability (deferred).

### 5.3 Saturation options

| ID | Definition | Use |
|----|------------|-----|
| `media_linear_std` | Linear standardized media (default branch) | CLEAN / weak-ID worlds |
| `media_hill_saturation` | Hill on standardized media with fixed or learned half/slope | Align with H4c `generative_kind: saturation` |

Outcome model remains **semi_log on \(y\)** unless a future ADR amends observation link; H5 focus is **media path alignment**, not changing \(y\) link in v1.

### 5.4 Transform identity / transform registry

Each sandbox fit carries:

```text
transform_registry_id: bayes_h5_media_transform_registry_v1
media_transforms_by_channel: { "tv": "media_geometric_adstock", "search": "media_raw_std", ... }
generative_kind_expected: adstock | saturation | linear | ...
transform_mismatch_mode: fit_aligned | fit_mismatch_intentional
```

| Mode | Behavior |
|------|----------|
| `fit_aligned` | Registry matches world `generative_kind` — expect improved \(\beta_{g,c}\) / \(\mu_c\) recovery vs H3 on same panel |
| `fit_mismatch_intentional` | Registry **differs** from generative kind — expect `transform_mismatch` diagnostic, not “recovery failure” gate |

**Identity rule:** `measurement_instrument_id` for sandbox runs remains research-only; no silent mapping from Ridge prod transforms.

---

## 6. Prior policy

### 6.1 Channel-level \(\mu_c\) priors

| Context | Prior | Rationale |
|---------|-------|-----------|
| Default | \(\mu_c \sim \mathcal{N}(0, 0.5)\) (match H3) | Baseline |
| Weak-signal worlds | Tighter \(\sigma_{\mu}\) or zero-mean with stronger shrinkage on \(z\) | Reduce false channel activation |
| Recovery candidates | Unchanged mean priors; evaluate via recovery metrics only | H4d showed clean stability |

### 6.2 \(\tau_c\) priors (heterogeneity)

| Context | Prior | Rationale |
|---------|-------|-----------|
| Default sandbox | \(\tau_c \sim \text{HalfNormal}(\sigma_\tau)\); default \(\sigma_\tau = 0.5\) | H3 MVP |
| Sparse **recovery** geos | Allow **per-world** \(\sigma_\tau \in \{0.5, 0.3, 0.2, 0.15\}\) in validation sweeps only | H4d: τ alone not sufficient; document, do not promote |
| Sparse **stress** geos | Fixed diagnostic \(\sigma_\tau\); **exclude** from τ calibration aggregates | INV-071 / H4d role separation |
| Correlated channels | **Linked** \(\tau\) or LKJ on scaled \(z\) (design option) | Reduce weak-ID blow-ups; implementation choice in H5b |

**H5 disposition:** No default promotion of \(\sigma_\tau = 0.15\) globally.

### 6.3 Weak-identification regularization

| Mechanism | Spec |
|-----------|------|
| Collinearity diagnostic | Max \|corr\| across channels in pre-period panel; warn if \(> 0.85\) |
| Optional ridge on non-centered \(z_{g,c}\) | Small \(\lambda\) jitter only on weak-ID worlds (research toggle) |
| Channel exclusion | TrustReport **restricted** flag; does not drop rows from likelihood automatically in v1 |

### 6.4 Sparse geo policy

| Role | World examples | Policy |
|------|----------------|--------|
| `recovery_candidate` | H4c SPARSE-RECOVERY, CLEAN-RECOVERY, H4 SIMPLE-POOLING | Eligible for recovery metric bands (report-only) |
| `stress_diagnostic` | H4 SPARSE-GEO | **report_only** — never pooled into recovery pass/fail |
| Pooling metric | `shrinkage_ratio_sparse` vs \(\hat\mu_c\) | **Not** a true-effect gate (unchanged) |
| Legacy diagnostic | `shrinkage_ratio_sparse_vs_true_mu` | Retained; not a promotion gate |

---

## 7. Diagnostics

### 7.1 Posterior recovery metrics (unchanged families)

| Metric | Role |
|--------|------|
| `beta_gc_mae`, `mu_c_mae` | Point recovery vs generative truth |
| `beta_gc_coverage_90` | Directional uncertainty on toys |
| `beta_interval_width_90_mean` | Uncertainty sanity |
| `shrinkage_ratio_sparse` | Pooling toward \(\hat\mu_c\) |
| `shrinkage_ratio_sparse_vs_true_mu` | Legacy recovery diagnostic |

Aggregated **per world role** per INV-071 — no global threshold.

### 7.2 Collinearity diagnostics

- `max_channel_corr_pre_period`
- `collinearity_warning` when above threshold on `weak_identification` worlds
- Posterior correlation summary (optional, research)

### 7.3 Transform mismatch diagnostics

- Compare `generative_kind` vs `media_transforms_by_channel`
- Emit `h5_transform_mismatch_warning` (structured list)
- On mismatch worlds: **expect** warning; classification `transform_mismatch` must not flip to global failure

### 7.4 Weak-signal diagnostics

- `media_scale` / effective SNR proxy from panel
- `weak_identification_warning` when SNR below world expectation
- Elevated `beta_gc_mae` on WEAK-SIGNAL world **expected** — document in pilot JSON

All diagnostics attach to `diagnostic_trust_report` / `h4_recovery` extension block (`h5_recovery` when implemented).

---

## 8. TrustReport mapping

### 8.1 Diagnostic only (always)

| Field class | Content |
|-------------|---------|
| Posterior summaries | \(\hat\mu_c\), \(\hat\tau_c\), \(\hat\beta_{g,c}\) — not DecisionSurface |
| Recovery metrics | MAE, coverage, shrinkage decomposition |
| H5 transform registry snapshot | IDs and parameters |
| Sandbox labels | `RESEARCH ONLY — NOT DECISION GRADE` |
| `approved_for_prod` | **false** |
| `prod_decisioning_allowed` | **false** |

### 8.2 Warn / restricted (report-only)

| Condition | TrustReport posture |
|-----------|---------------------|
| `transform_mismatch` world or warning | **restricted** — “do not interpret as recovery failure” |
| `weak_identification` / collinearity | **warn** or **restricted** |
| INV-071 exceeds report band on **recovery_candidate** only | **warn** / **restricted** per policy JSON |
| Sparse stress world | **report_only** — no warn escalation to global fail |

### 8.3 Remains blocked

| Item | Status |
|------|--------|
| Production DecisionSurface emission | **Blocked** |
| Optimizer / budget recommendation | **Blocked** |
| CalibrationSignal eligibility upgrade | **Blocked** |
| `decision_grade: true` | **Blocked** |
| Hard gates in CI | **Blocked** |
| Ridge replacement | **Blocked** |

---

## 9. Validation requirements before implementation

Implementation of `bayes_h5_sandbox_spec_v1` in code is **not authorized** until:

### 9.1 New H5 recovery worlds (`WORLD-BAYES-H5-*`)

| World ID (proposed) | Role | Intent |
|-------------------|------|--------|
| `WORLD-BAYES-H5-CLEAN-ALIGNED` | recovery_candidate | Linear generative; aligned transforms |
| `WORLD-BAYES-H5-ADSTOCK-ALIGNED` | recovery_candidate | Adstock generative + geometric adstock fit |
| `WORLD-BAYES-H5-SATURATION-ALIGNED` | recovery_candidate | Saturation generative + Hill media |
| `WORLD-BAYES-H5-SPARSE-RECOVERY` | recovery_candidate | Extends H4c sparse-recovery with aligned spec |
| `WORLD-BAYES-H5-SPARSE-STRESS` | stress_diagnostic | Supersedes misleading use of H4 SPARSE-GEO as recovery gate |
| `WORLD-BAYES-H5-WEAK-ID-CORR` | weak_identification | Correlated channels |
| `WORLD-BAYES-H5-WEAK-ID-SNR` | weak_identification | Low signal |
| `WORLD-BAYES-H5-ADSTOCK-MISMATCH` | transform_mismatch | Intentional wrong transform |
| `WORLD-BAYES-H5-SATURATION-MISMATCH` | transform_mismatch | Intentional wrong transform |

### 9.2 Transform-aligned vs mismatch negative worlds

- **Aligned** worlds: primary evidence that H5 spec improves MAE vs H3 on same seed/grid.
- **Mismatch** worlds: diagnostics must fire; **must not** fail global sandbox readiness.

### 9.3 Repeated multi-seed checks

- Minimum 3 NUTS seeds per world (fast profile for iteration; extended profile before any policy tighten).
- Stability on **recovery_candidate** roles only (H4d pattern).
- Compare H5 vs H3 MVP on same panels where possible.

### 9.4 Deliverables (post-implementation)

| Artifact | Purpose |
|----------|---------|
| `BAYES_H5_RECOVERY_PILOT_*.json` | H5 pilot metrics |
| Updated INV-071 policy or successor | Only after H5 repeated evidence |
| `tests/research/test_bayes_h5_*` | Fast contract + optional slow PyMC |

---

## 10. Promotion boundary

| Boundary | Rule |
|----------|------|
| **Production Bayesian MMM** | **Blocked** — unchanged |
| **Bayes-H3 production promotion row** | Remains **Blocked** in [ROADMAP_ALIGNMENT_REGISTRY.md](../ROADMAP_ALIGNMENT_REGISTRY.md) |
| **Optimizer / DecisionSurface** | Requires Bayes-H1 adapter proof + Promotion Gate + executive review — **out of H5 scope** |
| **Hard gates** | **Not authorized** by this ADR |
| **Ridge** | Default prod estimator until full promotion chain |

H5 success means: **credible sandbox spec + diagnostics + classified recovery map** — not production readiness.

---

## 11. Open questions

| ID | Question | Owner phase |
|----|----------|-------------|
| OQ-H5-01 | Learned vs fixed adstock \(\theta_c\) on aligned worlds | H5 implementation |
| OQ-H5-02 | National-only vs geo-specific media transforms | After identifiability worlds |
| OQ-H5-03 | Non-centered vs centered parameterization under strong \(\tau\) | Numerical stability pilot |
| OQ-H5-04 | Whether weak-ID ridge on \(z\) helps CORRELATED world without harming CLEAN | A/B H5 pilot |
| OQ-H5-05 | Merge H4c worlds into H5 catalog vs parallel catalogs | World catalog ADR |
| OQ-H5-06 | Re-calibrate INV-071 bands using H5 recovery_candidate runs only | Post H5 pilot |
| OQ-H5-07 | NumPyro backend timing — separate ADR vs H5 spec | Bayes-H5+ backend |
| OQ-H5-08 | Draw-based \(\Delta\mu\) in sandbox — remains forbidden for prod ([Bayes-H1](bayes_h1_decision_surface_preservation_adr.md)) | Out of scope |
| OQ-H5-09 | Fast vs extended MCMC profiles for H5 pilots (mirror H4d) | H5 validation |
| OQ-H5-10 | Promotion evidence chain: reproducible Δμ adapter + Promotion Gate before any prod tier | Bayes-H5 production candidacy ADR (separate) |

---

## Consequences

- **Accepted:** architectural direction for `bayes_h5_sandbox_spec_v1` (this ADR).  
- **Not authorized:** code changes to `model.py`, production paths, promotion flags, or hard CI gates.  
- **Next step:** implement `bayes_h5_sandbox_spec_v1` behind `run_sandbox_fit` feature flag + H5 world materialization + pilot JSON — **separate implementation authorization**.

**This ADR does not authorize production Bayesian decisioning.**
