# Monte Carlo threshold recommendations (Phase 5F)

**Version:** `monte_carlo_threshold_recommendations_v1.0.0`  
**Status:** **Recommendations only** — not approved for production gates  
**Tier:** tier_0_pilot (N≈25 effective evidence points)  
**Program:** [monte_carlo_reliability_program.md](monte_carlo_reliability_program.md)  
**Machine-readable:** [investigations/monte_carlo_pilot_characterization.json](investigations/monte_carlo_pilot_characterization.json)

---

## Executive summary

Tier-0 evidence confirms Phase **5C–5E** conclusions at scale:

- **Decision-grade** metrics (Δμ, optimizer, replay) achieve **high pass rates** on scored pilot cells.
- **Attribution diagnostic** metrics (coef, transform) **fail by design** on multi-channel shared-transform worlds — not because `TBD_v1_runtime` is mis-calibrated alone.
- **Structural** reliability is **high** (~0.89 behavioral structural score; ~0.89 lattice structural).
- **Trust modifiers** (drift) behave as designed; severity bands need tier-1 percentiles.

**Do not promote any threshold to `approved` until tier-1 (N≥100) and DR-04 sign-off.**

---

## 1. Evidence base

| Source | Worlds / points | Role |
|--------|-----------------|------|
| L5B behavioral lattice MVP | 10 cells | Stratified behavioral recovery |
| WORLD-008–012 anchors | 5 | Regression anchors |
| INV-056 investigation | 4 decomposition + 12 volume + 3 identifiability | Failure boundaries |
| L5A structural lattice | 12 | Structural-only (~0.89 score) |

Regenerate: `write_pilot_characterization(Path('.'))` from `monte_carlo_reliability.py`.

---

## 2. Observed distributions (tier-0)

| Capability | Metric class | Pass rate (pilot) | n scored | Interpretation |
|------------|--------------|-------------------|----------|----------------|
| coefficient_recovery | diagnostic | **0.0** | 3 | Expected on exact_recovery L5B cells |
| transform_consistency | diagnostic | **0.5** | 12 | BO transform misspecification |
| delta_mu_recovery | decision | **1.0** | 3 | Passes despite coef fail |
| optimizer_recovery | decision | **1.0** | 4 | Dedicated worlds |
| replay_recovery | decision | **1.0** | 8 | Replay-on cells |
| drift_behavior | trust | **0.5** | 2 | Partial/warning on drift cells |
| structural_integrity | structural | **0.73** | 30 | Some CERT-4A noise on optimizer/replay bundles |
| platform_contract_compatibility | structural | **0.91** | 34 | Occasional contract failures on non-baseline templates |
| anchor Δμ (BO) | decision | **1.0** | 4 | Same worlds as 0% anchor coef |

**Behavioral score:** ~0.57 · **Structural score:** ~0.89

---

## 3. Failure regions

### 3.1 Expected failures (not bugs)

| Driver | Affected metrics | Predictor axes |
|--------|------------------|----------------|
| Shared transform across channels | coef, transform | `n_channels≥3`, exact_recovery |
| BO hyperparameter search | transform | all BO-path worlds |
| Severe collinearity | coef (unstable) | `correlation_level=severe` |

### 3.2 Expected passes

| Driver | Affected metrics |
|--------|------------------|
| Decision-surface invariance | Δμ, optimizer, replay |
| Contract stability | structural, CERT-4A (majority) |
| Truth-pinned transforms | coef on 1ch / 2ch-orth mini-worlds |

### 3.3 Boundary hypotheses (tier-1 to verify)

| Condition | Expected coef recovery | Expected Δμ |
|-----------|------------------------|-------------|
| `noise_std` ≤ 0.02 | Fail on WORLD-008 template | Pass |
| `n_geos` 1→4, `n_periods` 10→18 | Fail (pinned ~0.20 err) | Pass |
| `correlation_level=severe` | Fail / unstable | Pass |
| `drift=on` | Coef skipped | Drift VAL-012 pass/warning |
| `truth_pinned` + 1 channel | Pass | Pass |

---

## 4. Reliability envelopes

Volume sweep (INV-056): increasing geos/periods **does not** restore coef recovery on WORLD-008 pinned template — max |β̂−β| stays ~0.20. Failure is **identifiability / feature design**, not data volume alone.

Identifiability mini-worlds: **id-1ch**, **id-2ch-orth**, **id-2ch-severe** pass **pinned** coef recovery; WORLD-008 fails pinned multi-channel.

---

## 5. Threshold recommendations (not approved)

### VAL-001 — Coefficient recovery

| Field | Value |
|-------|-------|
| Current | rtol=0.20, atol=0.08 (`TBD_v1_runtime`) |
| Observed | 0% pass exact_recovery BO; pinned max err ~0.14–0.88 |
| Suggested | **Keep as diagnostic**; loosen to P90 of pinned single-channel stratum at tier-1; **never** default release gate |
| Confidence | medium |
| Evidence | INV-056, L5B |

### VAL-002/003 — Transform recovery

| Field | Value |
|-------|-------|
| Current | rtol=0.05 on decay/Hill |
| Observed | ~50% pass rate on transform checks (BO path) |
| Suggested | Diagnostic only; report decay/Hill error percentiles; consider per-channel transform only in research worlds |
| Confidence | medium |

### VAL-004 — Δμ recovery

| Field | Value |
|-------|-------|
| Current | rtol=0.35, atol=0.15 |
| Observed | 100% pass on scored cells; relative error can be large while abs error passes |
| Suggested | **Tighten relative_error cap** at tier-1 (e.g. P95 ≤ 0.5) **in addition to** abs tolerance; retain as **decision gate candidate** |
| Confidence | medium |

### VAL-005 — Optimizer recovery

| Field | Value |
|-------|-------|
| Current | allocation L1 rtol=0.15, atol=4.0 |
| Observed | 100% on optimizer L5B cells |
| Suggested | Retain provisional; expand tier-1 corner-dominant surfaces |
| Confidence | low (small n) |

### VAL-006 — Replay recovery

| Field | Value |
|-------|-------|
| Current | lift rtol=0.25, atol=0.02 |
| Observed | 100% on replay-on cells |
| Suggested | Retain; add tier-1 stale-calibration negative stratum |
| Confidence | low |

### VAL-012 — Drift detection

| Field | Value |
|-------|-------|
| Current | post_pre_mae_ratio_min=1.15; severity bands in runner |
| Observed | Drift cells pass/warning; severity severe on WORLD-011 as expected |
| Suggested | Calibrate `minor`/`moderate`/`severe` from tier-1 ratio percentiles; keep trust_modifier **conditional** block |
| Confidence | low until tier-1 |

---

## 6. What is too strict vs too weak?

| Metric | Too strict? | Too weak? |
|--------|-------------|-----------|
| Coef (BO) on WORLD-008 | **Yes** if used as release gate | N/A for diagnostic |
| Coef (diagnostic reporting) | No | Possibly — pinned 1ch passes easily |
| Δμ abs tolerance | No | **Yes** on relative error — large relative fail masked by atol |
| Transform BO | Strict for attribution claims | Appropriate as diagnostic |
| Drift severity | Unknown until tier-1 | moderate band may be wide |

---

## 7. TrustReport calibration (green / yellow / red)

| Grade | Criteria (empirical pilot) | Default for Ridge BO? |
|-------|---------------------------|------------------------|
| **Green** | decision≥0.85, structural≥0.9, trust acceptable, attribution optional | **No** — pilot never achieves green on full behavioral score |
| **Yellow** | decision usable, attribution unsafe, trust caution OK | **Yes** — matches ~0.57 behavioral / ~1.0 decision on scored Δμ |
| **Red** | structural&lt;0.75, trust degraded, decision&lt;0.75 | Drift severe + readiness block |

**Recommendation:** Production TrustReport default posture = **yellow** for typical Ridge BO synthetic certification: decision-usable, attribution-unsafe, until tier-2 MC approves green bounds.

Aligns with [trust_report_semantics.md](trust_report_semantics.md).

---

## 8. Release-gate influence

| Metric class | Release-gate influence (post tier-2 approval) |
|--------------|-----------------------------------------------|
| structural | **Always** — block on CERT/contract fail |
| decision_grade | **Yes** — Δμ, optimizer, replay after approved thresholds |
| trust_modifier | **Conditional** — severe drift blocks optimization |
| diagnostic_attribution | **No** — warn only unless attribution profile |

---

## 9. Open questions (tier-1+)

1. What P95 coef error is achievable on **per-channel transform** research worlds?
2. What relative Δμ error cap preserves decision quality on tier-1 N=100?
3. Do optimizer/replay passes hold under `noise=medium/high`?
4. Negative world catalog size and gate coverage (DR-03)?
5. Minimum N for DR-04 approval per metric class?

---

## 10. Success criteria checklist

| Question | Tier-0 answer |
|----------|---------------|
| Realistic recovery levels? | Δμ/optimizer/replay high; coef BO low on multi-channel BO |
| Expected failures? | Shared transform, collinearity, BO search |
| Too strict thresholds? | Coef as **release** gate; Δμ relative leg |
| Too weak thresholds? | Δμ relative; drift moderate band |
| World characteristics predicting failure? | channels≥3 + shared transform; severe correlation |
| TrustReport calibration? | Default **yellow**; green requires tier-2 |
| Release-gate metrics? | structural + decision; not coef |

---

## 11. Next steps

1. **Tier-1 batch** (N=100) — structured coverage runner (future module).
2. **DR-04 committee** — review this document + tier-1 percentiles.
3. **Track 4** — only after tier-1 recommendations accepted: Bayes-H roadmap refinement per [monte_carlo_reliability_program.md §9](monte_carlo_reliability_program.md#9-track-4-gate-post-5f).
