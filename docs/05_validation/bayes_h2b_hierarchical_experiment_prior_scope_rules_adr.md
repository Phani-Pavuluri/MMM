# Bayes-H2b — Hierarchical Experiment-Prior Scope Rules

**ADR ID:** `bayes_h2b_hierarchical_experiment_prior_scope_rules_v1`  
**Title:** Bayes-H2b — Hierarchical Experiment-Prior Scope Rules  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisites:** [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md) · [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md) (both **Accepted**)  
**Related:** [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) · [trust_report_semantics.md](trust_report_semantics.md) · [groundtruth_contract.md](groundtruth_contract.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) Decision 12

---

## 1. Status

| Field | Value |
|-------|--------|
| **Status** | **Accepted** |
| **Date** | 2026-05-29 |
| **Scope** | Architecture + validation-world specification **only** |
| **Authorizes** | Hierarchy propagation rules, claim semantics, TrustReport hierarchy fields, seven `WORLD-BAYES-*` world specifications |
| **Does not authorize** | PyMC, samplers, prior/likelihood objects, model classes, posterior decisioning, experiment-specific APIs, Bayesian-only release paths, production coefficient-facing outputs |

This ADR continues from the current **Marketing Intelligence Platform (MIP) / MMM checkpoint**: v1.0.0 contract freeze, Reliability Program Phases 5C–5F, Bayes-H1 (DecisionSurface preservation), Bayes-H2 (CalibrationSignal mapping). Ridge + BO remains the **production** path; Bayesian hierarchy remains **research/experimental**.

---

## 2. Context

### 2.1 What Bayes-H2 decided

[Bayes-H2](bayes_h2_calibration_signal_mapping_adr.md) answered:

> Given a **CalibrationSignal**, should it become local likelihood, national likelihood, hyper-prior, replay penalty, or TrustReport-only?

### 2.2 What Bayes-H2b must decide

Bayes-H2b answers:

> Once a signal is mapped to a hierarchy level, **what claims may it support above, below, or across that hierarchy** — and what must **TrustReport** disclose?

This is the **scope-governance layer** between evidence ingress and any future Bayesian fitter (Bayes-H3).

### 2.3 Platform ABI (unchanged)

| Contract | Role in this ADR |
|----------|------------------|
| **CalibrationSignal** | Sole evidence ingress — no alternate experiment APIs |
| **DecisionSurface** | Sole production decision object — full-panel Δμ = μ(plan) − μ(baseline) via `simulate()` |
| **Estimand** | Gates replay masks and evidence inclusion |
| **TrustReport** | Uncertainty, evidence quality, conflicts, sensitivity, freshness, hierarchy diagnostics |
| **Release Gates** | `approved_for_prod`, promotion, `PolicyError` — identical for Ridge and future Bayes |

### 2.4 Strategic checkpoint

The platform’s next direction is **reliability proof** (controlled worlds, certification, replay, governance, artifact integrity), not unconstrained model expansion. Any hierarchical Bayesian design that violates the ABI above is **rejected** by this ADR.

---

## 3. Non-goals

This ADR explicitly **does not**:

| Non-goal | Notes |
|----------|--------|
| Implement code (Python, PyMC, configs) | Specification only |
| Introduce PyMC, samplers, or model classes | Bayes-H3+ |
| Create Bayesian-only decision paths or optimizers | [Bayes-H1](bayes_h1_decision_surface_preservation_adr.md) |
| Create experiment-specific Bayesian APIs | [Bayes-H2](bayes_h2_calibration_signal_mapping_adr.md) Decision 1 |
| Change DecisionSurface, Estimand, CalibrationSignal base schema, Release Gates | Extensions documented; schema ADR at implementation |
| Change Ridge production path or full-panel Δμ definition | |
| Put posterior draws, coef tables, or priors into optimizer or release gates | Diagnostic / TrustReport only |
| Materialize validation worlds | Track 2 engineering — **specified** here, not built |

---

## 4. Definitions

| Term | Definition |
|------|------------|
| **MIP** | Marketing Intelligence Platform — contract-centric orchestration around MMM and adjacent capabilities |
| **CalibrationSignal** | Platform record for one experiment/replay evidence unit (scope, estimand, lift, uncertainty, freshness, source) |
| **Signal registry** | Internal map `signal_id → {mechanism, influence_class, scope_node, weight, trace}` — Bayes-H2 |
| **Scope graph** | Contract-governed hierarchy of geographic/analytic nodes and declared edges — not assumed strict tree |
| **Native evidence level** | Scope where the experiment was designed (e.g. single DMA, multi-DMA, national) |
| **Propagation** | Allowed influence/claims across scope levels — this ADR |
| **Influence class** | How evidence acts: local likelihood-style, regional pooling, national calibration, hyper prior-style, TrustReport-only, excluded |
| **Borrowed strength** | Child estimate informed by parent hyperparameters or parent signals — not a local experiment |
| **Claim level** | What operators may assert (`directly_observed_experimental_evidence`, …) |
| **Decision-grade influence** | Evidence that may affect fitted generative state used for DecisionSurface point policy — subject to gates |
| **Full-panel Δμ** | μ(plan) − μ(baseline) on training panel geometry via existing simulate path |

---

## 5. Hierarchy Model

### 5.1 Geographic ladder (binding propagation chain)

```text
DMA (geo)  →  state  →  region  →  national
     ↑___________________________________|
              (downward: shrinkage / calibration pressure only)
```

**Channel** and **segment** cross-cut the geo ladder. **Time window** modifies any node.

### 5.2 Scope graph (not a naive tree)

Geography is a **contract-governed scope graph**:

- Nodes: `dma`, `state`, `region`, `national`, `channel`, `segment`, `time_window` (modifier)
- Edges: declared `parent_scope_ids` / `child_scope_ids` on each CalibrationSignal
- Overlapping membership (e.g. DMA in two reporting regions) is allowed; propagation uses **declared** edges only

### 5.3 Required scope metadata

| Field | Required when | Purpose |
|-------|---------------|---------|
| `scope_type` | Always | Node level |
| `scope_id` | Always | Stable id |
| `parent_scope_ids` | Always (empty at national) | Upward links |
| `child_scope_ids` | Aggregating scopes | Downward membership |
| `coverage_ratio` | Multi-geo, parent claims | Treated ÷ eligible geos at target level |
| `scope_weight` | Recommended | Population/spend fraction |
| `population_basis` / `spend_basis` / `kpi_basis` | Weighted scopes | Aggregation semantics |
| `time_window` | Post-period experiments | Overlap + replay |

**Reject:** Ambiguous `scope_id` → panel resolution fails → signal **excluded**, `missing_scope_metadata_signals` in TrustReport.

### 5.4 Hierarchy parameters (reference — diagnostic-facing)

| Parameter | Level | Decision-facing? |
|-----------|-------|------------------|
| \(\beta_{g,c}\) | DMA-local | **No** — internal; DecisionSurface uses point generative state |
| State / regional contrasts | Aggregated \(\beta\) | **No** |
| \(\mu_c\) | National channel hyper-mean | **No** |
| \(\tau_c\) | Pooling scale | **No** — TrustReport pooling diagnostics |

---

## 6. CalibrationSignal Scope Mapping

Maps **CalibrationSignal.scope** → **hierarchy target** (before propagation rules).

| `scope_type` / scope declaration | Hierarchy target | Default influence class |
|----------------------------------|------------------|-------------------------|
| DMA / `geo_ids` | \(\beta_{g,c}\) for \(g \in\) treated set | Local likelihood-style |
| `state` + member geos | State-level effect on member \(\beta_{g,c}\) | Local likelihood-style on members |
| `region` + member geos/states | Regional effect on members | Regional pooling evidence |
| `national` | \(\mu_c\) national channel effect | National calibration / hyper prior-style |
| `channel` (national) | Channel hyper \(\mu_c\) | Hyper prior-style |
| `segment` | None (default) | TrustReport-only |
| `time_window` | Modifier on target above | Narrows likelihood support |

**Segment → geo MMM:** Requires explicit `scope_bridge_id` in signal; without bridge → **excluded** from geo fit, TrustReport-only.

**Time-window:** Must intersect panel; else **excluded**.

---

## 7. Evidence Influence Rules

After Bayes-H2 mechanism assignment, each included signal receives an **influence class**:

| Influence class | Meaning | Typical sources |
|-----------------|---------|-----------------|
| **local_likelihood_style** | Direct incremental evidence on declared local parameters | GeoX DMA, replay unit at geo scope |
| **regional_pooling_evidence** | Evidence on member geos + shared regional structure | Regional GeoX, multi-DMA with regional native level |
| **national_calibration_evidence** | Evidence on \(\mu_c\) / global calibration | National CLS, aligned national A/B |
| **hyper_prior_style** | Soft constraint on hyper-mean (not point mass) | Pre-data national lift, weak CLS |
| **trust_report_only** | Diagnostic — no decision-grade posterior shift | Observational, misaligned segment, failed gates |
| **excluded** | No fit influence; auditable reason | Estimand mismatch, expired, ambiguous scope |

**Hard rule:** Influence class never bypasses CalibrationSignal or creates a parallel experiment API.

---

## 8. Propagation Rules

### 8.1 DMA → state

| Direction | Rule |
|-----------|------|
| **Up (DMA → state)** | DMA signal may produce **state-level summary** in TrustReport only unless `coverage_ratio` of treated DMAs within state ≥ **0.60** and representativeness ≥ **0.70** — then **regional_pooling_evidence** at state may be considered (not point mass on state “truth”) |
| **Not allowed** | Single DMA → state causal claim as `directly_observed_experimental_evidence` at state level |

### 8.2 State → region

| Direction | Rule |
|-----------|------|
| **Up** | State-level aggregated evidence → region summary or gated pooling; requires coverage ≥ **0.60** at region |
| **Not allowed** | One state → regional point-mass overwrite |

### 8.3 Region → national

| Direction | Rule |
|-----------|------|
| **Up** | Region → national **parent_summary_of_child_evidence** by default; **national_calibration_evidence** only if coverage ≥ **0.80** and all upward gates pass |
| **Not allowed** | Single region → national \(\mu_c\) point mass |

### 8.4 National → region / state / DMA (downward)

| Direction | Rule |
|-----------|------|
| **Down** | National CLS/A/B (aligned) → **national_calibration_evidence** on \(\mu_c\); children receive **shrinkage center** only |
| **Child caveat** | `borrowed_strength_from_parent: true`, `claim_level: model_estimate_with_borrowed_strength` |
| **Not allowed** | National CLS → DMA-level **causal lift** claim; national A/B → state ROI as experiment result |

### 8.5 Multi-geo experiments

| Native level | Allowed targets | Parent claim |
|--------------|-----------------|--------------|
| Multi-DMA GeoX | Each treated \(\beta_{g,c}\) | Region/state/national only if coverage + representativeness gates pass |
| State GeoX | Member DMAs with caveats on non-treated | Region/national gated |
| Regional GeoX | Member geos/states | National gated (0.80 coverage) |

### 8.6 Sparse-geo evidence

| Condition | Rule |
|-----------|------|
| No local signal | May borrow via \(\mu_c\), \(\tau_c\) → `model_estimate_with_borrowed_strength` |
| Local signal present | Local wins same-scope over parent (precedence §9) |
| Sparse + conflict | Fail-closed or local-wins per §9 — never silent blend |

### 8.7 National CLS evidence

- **Native:** national  
- **Influence:** `national_calibration_evidence` / `hyper_prior_style` on \(\mu_c\)  
- **Downward:** pooling only + child caveats  
- **Never:** DMA causal attribution from CLS alone  

### 8.8 Local GeoX evidence

- **Native:** dma  
- **Influence:** `local_likelihood_style` on treated \(\beta_{g,c}\) only  
- **Upward:** diagnostics / parent summary unless gates pass  
- **Never:** national \(\mu_c\) point mass from single DMA  

### 8.9 Stale evidence

See §11 — stale signals **downweighted** or **excluded**; stale parent **does not** refresh children.

### 8.10 Missing-SE evidence

See §10 — cannot be **decision-grade** without SE (or policy-approved CI conversion).

### 8.11 Estimand-misaligned evidence

See §12 — **excluded** from influence; TrustReport-only listing.

### 8.12 Lateral propagation (summary)

**Default:** `lateral_borrowing: none`. **Allowed:** `pooled_only` via shared \(\mu_c\), \(\tau_c\). **Forbidden:** Region A experiment calibrates Region B; DMA A likelihood on DMA B.

---

## 9. Conflict Rules

**No silent averaging** — ever.

### 9.1 Conflict groups

Signals conflict when they share **overlapping** channel, geo scope, and time window and lifts differ beyond `conflict_tolerance` (default **15%** on comparable scale).

### 9.2 Pairwise behaviors

| Conflict type | Resolution |
|---------------|------------|
| **GeoX local vs GeoX local** (same scope) | `scope_conflicts` + downweight all in group or **fail-closed**; highest design quality + fresh + SE wins **only** if policy `conflict_resolution: precedence` — never unlabeled average |
| **GeoX local vs CLS national** | Different scopes — both may apply at native levels; if implied national lift inconsistent → **fail-closed** or local-wins at DMA, national diagnostic at \(\mu_c\) |
| **Fresh vs stale** | Fresh precedence; stale downweighted/excluded |
| **High-precision vs low-precision** | Higher \(\omega\) (lower SE) ranks higher within same tier |
| **Aligned vs misaligned estimand** | Misaligned **excluded** |
| **Overlapping geo scopes** | Declare precedence: **narrower native scope wins** for overlap geos |
| **Overlapping time windows** | Split by window or exclude overlapping group |
| **Conflicting lift scales** | **Exclude** group — `lift_scale_mismatch` |

### 9.3 Deterministic precedence (influence tier)

1. Same-scope causal (fresh, aligned estimand, SE present)  
2. Parent causal (downward gates passed)  
3. Sibling pooled (`pooled_only`)  
4. Observational (TrustReport-only)

Within tier: fresh > stale; aligned estimand > misaligned; known SE > missing SE; design quality A > B > C.

### 9.4 Fail-closed triggers

- Same-scope high-quality causal conflict beyond tolerance  
- Lift scale incompatible  
- Estimand incompatible  
- Required SE missing for decision-grade use  
- Scope mapping ambiguous  
- `coverage_ratio` below threshold for parent claim  

---

## 10. Missing-Uncertainty Rules

| CalibrationSignal uncertainty | Treatment |
|------------------------------|-----------|
| **`standard_error` / `lift_se`** | Map to precision \(\omega = 1/\sigma^2\); may be decision-grade if other gates pass |
| **Confidence interval, no SE** | Convert to SE via declared `ci_level` + distributional assumption; log conversion in `signal_weight_summary`; if conversion not declared → **diagnostic only** |
| **p-value, no SE** | **TrustReport-only** — cannot infer precision for decision-grade |
| **Point estimate only** | Source policy: GeoX post-data → **excluded** decision-grade unless `allow_missing_uncertainty: true` → weak diagnostic tier |
| **Missing entirely** | **Excluded** decision-grade; `missing_se_signals` in TrustReport |

**Hard rule:** Missing-SE evidence **cannot silently become decision-grade**.

Caps: \(\omega\) capped at governance max to prevent single-experiment domination.

---

## 11. Staleness Rules

Aligned with [calibration_freshness.md](../04_governance/calibration_freshness.md) and Bayes-H2 Decision 6.

| State | Posterior influence | TrustReport |
|-------|---------------------|-------------|
| **Fresh** | Full weight per §10 | `included_signals` |
| **Stale** (past `calibration_max_age_days`, in grace) | Multiply precision by **0.25** (default) | `stale_signals` + warning |
| **Expired** / revoked | **Excluded** | `stale_signals` + `excluded_signals` |

- Stale **parent** does not confer freshness on **children**  
- Freshness affects **trust modifiers** — may block optimization via readiness even if fit runs  

---

## 12. Estimand Alignment Rules

Inclusion requires (Bayes-H2 Decision 4 + hierarchy scope):

1. `estimand` in model allowlist  
2. `lift_scale` compatible with modeling KPI / replay  
3. Treated channels in panel  
4. Geo scope ∩ training geos non-empty  
5. Segment signals have `scope_bridge_id` or are excluded  

**Misaligned estimand:**

- **Influence:** `excluded`  
- **TrustReport:** `estimand_excluded_signals` with `exclusion_reason: estimand_mismatch`  
- **May appear** in TrustReport for audit — **zero** hierarchy propagation  

---

## 13. TrustReport Requirements

TrustReport remains the **only** layer for evidence quality, uncertainty disclosure, conflicts, and hierarchy diagnostics — **not** the optimizer (Bayes-H1).

### 13.1 Bayes-H2 fields (retained)

`included_signals`, `excluded_signals`, `stale_signals`, `conflicting_signals`, `signal_weight_summary`, `sensitivity_required`, `posterior_evidence_alignment`

### 13.2 Bayes-H2b hierarchy block (`hierarchy_evidence`)

| Field | Purpose |
|-------|---------|
| `hierarchy_scope_map` | Resolved scope graph |
| `included_signals` / `excluded_signals` / `stale_signals` / `missing_se_signals` / `conflicting_signals` / `estimand_excluded_signals` | Per hierarchy-aware listing |
| `propagated_evidence` | Edges with direction (up/down) and gate outcomes |
| `pooling_diagnostics` | \(\tau_c\), shrinkage flags, sparse-geo borrow paths |
| `hierarchy_diagnostics` | Coverage, representativeness per edge |
| `sensitivity_diagnostics` | Conflict groups, alternate resolution traces |
| `freshness_diagnostics` | Age, tier, parent/child staleness |
| `local_vs_national_alignment` | Consistency score; warnings when local vs parent implied lifts diverge |
| `direct_signal_ids`, `parent_signal_ids`, `child_signal_ids`, `sibling_signal_ids` | Traceability |
| `borrowed_strength_sources` | Per geo/channel parent linkage |
| `claim_level` | Per affected scope/channel |
| `unsupported_claims` | Disallowed narratives |
| `lateral_borrowing` | `none` \| `pooled_only` \| `unsupported` |
| `propagation_path` | Ordered edges with pass/fail |

---

## 14. Required Validation Worlds

All worlds: **CalibrationSignal** fixtures traceable to `experiment_truth`; hierarchy metadata populated; certification asserts TrustReport fields.

### 14.1 WORLD-BAYES-GEOX-LOCAL

| Item | Specification |
|------|----------------|
| **Purpose** | Prove local GeoX stays local; upward propagation gated |
| **Synthetic setup** | Single-DMA (or few DMA) GeoX on one channel; known \(\beta_{g,c}\) truth; optional parent geography for gate tests |
| **Expected behavior** | `local_likelihood_style` on treated DMA only; national \(\mu_c\) unchanged by point mass; parent gets `parent_summary_of_child_evidence` unless coverage gates pass |
| **Failure modes** | National lift claim from one DMA; missing `claim_level` on parent summary |
| **TrustReport obligations** | `direct_signal_ids`, `propagation_path` (dma→state blocked or summary-only), `lateral_borrowing: none` or `pooled_only` |
| **Release-gate implications** | Research artifact `prod_decisioning_allowed: false`; if hierarchy warnings severe → `decision_safe: false` per trust modifiers |

### 14.2 WORLD-BAYES-CLS-NATIONAL

| Item | Specification |
|------|----------------|
| **Purpose** | National CLS influences \(\mu_c\) only; children borrow with caveat |
| **Synthetic setup** | National CLS on channel; multi-DMA panel with known \(\mu_c\), \(\tau_c\) |
| **Expected behavior** | `national_calibration_evidence` on \(\mu_c\); all DMAs `borrowed_strength_from_parent: true`; no DMA causal CLS claim |
| **Failure modes** | DMA-level lift labeled `directly_observed_experimental_evidence` from CLS |
| **TrustReport obligations** | `child_signal_ids`, `claim_level: model_estimate_with_borrowed_strength`, `local_vs_national_alignment` |
| **Release-gate implications** | Mislabeled claims → trust modifier blocks optimization narrative |

### 14.3 WORLD-BAYES-CONFLICT

| Item | Specification |
|------|----------------|
| **Purpose** | Local GeoX vs national CLS (and/or second GeoX) — no silent average |
| **Synthetic setup** | GeoX +8% DMA-level; CLS +2% national same channel; optional overlapping windows |
| **Expected behavior** | `scope_conflicts` populated; `sensitivity_required: true`; precedence or fail-closed per §9 |
| **Failure modes** | Posterior target equals average of lifts; conflict fields empty |
| **TrustReport obligations** | `conflicting_signals`, `conflicting_hierarchy_signals`, conflict metric |
| **Release-gate implications** | `conflict_fail_closed: true` → may block promotion readiness |

### 14.4 WORLD-BAYES-STALE

| Item | Specification |
|------|----------------|
| **Purpose** | Stale downweight/exclude; visible in TrustReport |
| **Synthetic setup** | Fresh local GeoX + stale national CLS (or stale parent) |
| **Expected behavior** | Stale excluded or 0.25 weight; fresh wins; parent stale does not refresh child |
| **Failure modes** | Stale signal at full precision; no `stale_signals` entry |
| **TrustReport obligations** | `stale_signals`, `freshness_diagnostics` |
| **Release-gate implications** | Stale-heavy fits → calibration freshness trust modifier (INV-041 family) |

### 14.5 WORLD-BAYES-MISSING-SE

| Item | Specification |
|------|----------------|
| **Purpose** | Missing SE cannot be decision-grade |
| **Synthetic setup** | GeoX with point lift only; optional CI-only second signal |
| **Expected behavior** | Excluded or diagnostic tier; CI conversion logged if used |
| **Failure modes** | High-precision weight from point estimate alone |
| **TrustReport obligations** | `missing_se_signals`, weight tier `excluded` or `diagnostic` |
| **Release-gate implications** | No decision-grade evidence → stricter `decision_safe` |

### 14.6 WORLD-BAYES-SPARSE-GEO

| Item | Specification |
|------|----------------|
| **Purpose** | Sparse DMAs borrow; claim level correct |
| **Synthetic setup** | Many sparse DMAs, few with local GeoX; known \(\tau_c\) truth |
| **Expected behavior** | Sparse without local: `model_estimate_with_borrowed_strength`; with local GeoX: local wins at treated DMA |
| **Failure modes** | Sparse DMA labeled direct experiment without local signal |
| **TrustReport obligations** | `pooling_diagnostics`, `borrowed_strength_sources` |
| **Release-gate implications** | Attribution-unsafe if claim levels wrong (5D metric classes) |

### 14.7 WORLD-BAYES-ESTIMAND-EXCLUDE

| Item | Specification |
|------|----------------|
| **Purpose** | Incompatible estimand excluded from fit |
| **Synthetic setup** | A/B product estimand on geo MMM panel without scope bridge |
| **Expected behavior** | Zero hierarchy propagation; `estimand_excluded_signals` only |
| **Failure modes** | Posterior shift from excluded signal |
| **TrustReport obligations** | `estimand_excluded_signals`, `excluded_hierarchy_signals` |
| **Release-gate implications** | Exclusion must not block valid signals silently — all decisions visible |

---

## 15. Release Gate Implications

| Principle | Implication |
|-----------|-------------|
| **Same gates for Ridge and Bayes** | No `bayesian_approved` flag; use `approved_for_prod` + TrustReport + promotion |
| **DecisionSurface only** | Release checks VAL-004/005 on Δμ — not posterior or coef recovery as prod blockers (5D) |
| **Trust modifiers** | Hierarchy conflicts, stale-heavy evidence, missing SE, mislabeled claims → `decision_safe` / `optimization_blocked` via [trust_report_semantics.md](trust_report_semantics.md) |
| **Evidence visibility** | Any exclusion/downweight must appear in TrustReport — **no silent drops** |
| **Research default** | Bayesian artifacts: `prod_decisioning_allowed: false` until Bayes-H5 review |
| **Fail-closed conflicts** | May set `release_gate_recommendation: block` on scorecard interpretation — not a new gate type |

---

## 16. Anti-patterns

| Anti-pattern | Rejection rationale |
|--------------|---------------------|
| Local GeoX → national \(\mu_c\) point mass | §8.3, §8.8 |
| National CLS → DMA causal claim | §8.4, §8.7 |
| Single-region → global channel truth | §8.3 upward gates |
| Sibling region/DMA direct calibration | §8.12 |
| Silent local/parent or cross-signal averaging | §9 |
| Borrowed strength reported as direct experiment | §12 claim levels |
| Stale parent silently precision-weighting children | §11 |
| Missing-SE as high-precision calibration | §10 |
| Ambiguous geo mapping accepted | §5.3 |
| Experiment-specific Bayesian sampler API | Bayes-H2 §1 |
| Window-slice replay resetting adstock | Bayes-H2 §9 |
| Posterior draws / coefs in optimizer | Bayes-H1 |
| Bayesian-only release path | Bayes-H1 §5 |
| Bypass CalibrationSignal | Bayes-H2 |
| Bypass DecisionSurface / simulate | Bayes-H1 |

---

## 17. Acceptance Criteria

The ADR is **Accepted** because it satisfies:

| ID | Criterion | Met |
|----|-----------|-----|
| AC-1 | Preserves **DecisionSurface supremacy** | ✅ §2.3, §15 |
| AC-2 | Preserves **CalibrationSignal** as sole ingress | ✅ §2.3, §7 |
| AC-3 | Preserves **full-panel Δμ** decisioning | ✅ §2.3, §4 |
| AC-4 | Keeps **posterior uncertainty** out of optimizer | ✅ §3, §5.4, §15 |
| AC-5 | Keeps **coefficients/priors** diagnostic/non-decision-facing | ✅ §5.4 |
| AC-6 | Prevents **silent evidence averaging** | ✅ §9 |
| AC-7 | Defines **hierarchy propagation** for DMA→state→region→national and downward | ✅ §8.1–8.4 |
| AC-8 | Defines **multi-geo, sparse, CLS, GeoX, stale, missing-SE, estimand** cases | ✅ §8.5–8.11, §10–12 |
| AC-9 | Defines **exclusion** for incompatible estimands | ✅ §12 |
| AC-10 | Requires **TrustReport visibility** for all evidence decisions | ✅ §13 |
| AC-11 | Specifies **seven validation worlds** with purpose, setup, behavior, failures, TrustReport, gates | ✅ §14 |
| AC-12 | **No code / PyMC / samplers / model classes** | ✅ §3 |
| AC-13 | Rejects ABI-violating hierarchy designs | ✅ §2.4, §16 |

**World materialization** is Track 2 exit criteria before Bayes-H3 — not part of ADR acceptance, but specified here.

---

## 18. Open Questions

| ID | Question | Owner / phase |
|----|----------|----------------|
| OQ-1 | Exact `scope_bridge` schema for segment→geo MMM | Schema ADR + Bayes-H3 |
| OQ-2 | Numeric `representativeness_score` computation from panel weights | Bayes-H2c / world truth |
| OQ-3 | Whether `conflict_resolution: precedence` is allowed in prod or research-only | Bayes-H5 governance |
| OQ-4 | CI→SE conversion defaults per source (GeoX vs CLS) | Calibration ETL |
| OQ-5 | CERT check IDs for hierarchy TrustReport fields (REC-Bayes-h-*) | Bayes-H4 |
| OQ-6 | Nested geo trees (DMA→state→region) with overlapping region membership | Bayes-H2c worlds |

Unresolved OQ items **do not** authorize sampler code or ABI changes.

---

## 19. Final Decision

**We accept** `bayes_h2b_hierarchical_experiment_prior_scope_rules_v1` as binding architecture for Track 4.

1. **Hierarchy propagation** is governed by the scope graph, influence classes, explicit DMA→state→region→national (and downward) rules, lateral pooling-only constraint, conflict precedence, and claim semantics.  
2. **CalibrationSignal** remains the only evidence ingress; **DecisionSurface** remains the only production decision surface with **full-panel Δμ** via simulation.  
3. **TrustReport** must expose all evidence inclusion/exclusion, propagation, conflicts, missing SE, staleness, and hierarchy diagnostics.  
4. **Seven `WORLD-BAYES-*` worlds** are specified for materialization before Bayes-H3.  
5. Any design violating §16 anti-patterns or platform ABI is **rejected**.

**Next phase:** Materialize bundles + `hierarchy_evidence_validator` stub per [BAYES_H2B_VALIDATION_RUNNER_002.md](../BAYES_H2B_VALIDATION_RUNNER_002.md). **Bayes-H2d** and **Bayes-H3** blocked until `VAL-BAYES-H2B-SMOKE` passes.

---

## References

- [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md)  
- [bayes_h2_calibration_signal_mapping_adr.md](bayes_h2_calibration_signal_mapping_adr.md)  
- [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md)  
- [platform_roadmap.md](platform_roadmap.md)  
- [trust_report_semantics.md](trust_report_semantics.md)  
- [reliability_threshold_governance.md](reliability_threshold_governance.md)
