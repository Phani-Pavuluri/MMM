# Bayes-H2 — CalibrationSignal Mapping ADR

**ADR ID:** `bayes_h2_calibration_signal_mapping_v1`  
**Status:** **Accepted** (architecture only — does not authorize PyMC, priors in code, likelihood code, samplers, or model classes)  
**Date:** 2026-05-29  
**Track:** [platform_roadmap.md § Track 4 — Research Sandbox](platform_roadmap.md#track-4--research-sandbox)  
**Prerequisite:** [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md) (**Accepted**)  
**Related:** [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) · [trust_report_semantics.md](trust_report_semantics.md) · [groundtruth_contract.md](groundtruth_contract.md) § experiment_truth · [calibration_freshness.md](../04_governance/calibration_freshness.md) · [synthetic_architecture_decisions.md](synthetic_architecture_decisions.md) Decision 11

**Explicit scope:** This ADR binds how **experiment evidence** enters Bayesian hierarchical Geo MMM. It does **not** authorize implementation of PyMC models, priors, likelihoods, samplers, posterior artifacts, or production decisioning.

---

## Context

Bayes-H1 froze **DecisionSurface supremacy** and posterior separation. The remaining Track 4 design risk is **experiment ingestion**: GeoX, CLS, A/B, holdout, and replay evidence must enter the hierarchy without:

- a parallel experiment API  
- silent coef overwrite  
- conflict averaging without disclosure  
- bypass of replay / estimand / freshness semantics  

Ridge calibration already flows through **CalibrationUnit** → replay → loss. Bayesian fitting must use the **same CalibrationSignal contract** as the sole evidence ingress, with an internal **signal registry** mapping to prior / likelihood / penalty / exclusion.

---

## Glossary (binding)

| Term | Definition |
|------|------------|
| **CalibrationSignal** | Platform contract record for one piece of experiment or replay evidence (scope, estimand, lift, uncertainty, freshness, source). Materialized from approved **CalibrationUnit** / experiment registry payloads — not a second JSON schema. |
| **Signal registry** | Estimator-internal index: `signal_id → {mechanism, weight, scope, trace}` built only from CalibrationSignal inputs. |
| **Mechanism** | How evidence enters fitting: `prior_term`, `likelihood_term`, `pseudo_observation`, `calibration_penalty`, `excluded`, `trust_report_only`. |
| **Evidence scope** | Declared geographic / channel / segment / time coverage of a signal. |
| **Hyper-parameters** | \(\mu_c, \tau_c\) and national transform parameters — population-level hierarchy. |
| **Local parameters** | \(\beta_{g,c}\) for geo \(g\) and channel \(c\). |

---

## Decision 1 — CalibrationSignal as sole evidence ingress

### Decision

All experiment evidence types must enter Bayesian hierarchical Geo MMM **only** through **CalibrationSignal** (including materialized replay units, GeoX exports, CLS exports, A/B results, holdout panels, and future sources):

| Allowed ingress | Prohibited ingress |
|-----------------|-------------------|
| `CalibrationSignal` → signal registry → Bayesian mechanisms | `set_posterior_mean_beta(...)` from experiment lift |
| Approved experiment registry → CalibrationSignal ETL | GeoX-specific Bayesian API |
| Replay-implied lift via existing replay path | Direct sampler hooks from orchestration |
| Certification `experiment_truth` → synthetic CalibrationSignal fixtures | Ad hoc `experiment_coef` objects on decide |

**No direct experiment-specific Bayesian APIs.** Orchestration and copilots emit **CalibrationSignal-compatible** specs; the estimator adapter translates them.

### Rationale

- Replay certification (VAL-006), freshness (VAL-007), and promotion fingerprints assume a single evidence ABI.  
- Bayes-H1 requires DecisionSurface / TrustReport separation; a second ingress path would fragment audit trails.

### Consequences

- Bayes-H3 implementation must implement `build_signal_registry(signals: list[CalibrationSignal])` — no alternate constructor.  
- Negative tests: evidence without CalibrationSignal must not move posterior mass.  
- INV-063 / INV-066 close on mapping; scope propagation detail deferred to **Bayes-H2b**.

---

## Decision 2 — Evidence scope mapping

### Decision

Every CalibrationSignal must declare **evidence scope** (one primary scope; optional secondary scopes for multi-geo tests). Scope determines **eligible hierarchy targets** and **default mechanism** (overridable only by Decision 3 rules).

### Scope → hierarchy mapping table (binding)

| Evidence scope | Primary hierarchy target | Default mechanism | May also affect | TrustReport-only cases |
|----------------|-------------------------|-----------------|-----------------|------------------------|
| **geo-level** (`geo_ids` explicit) | \(\beta_{g,c}\) for \(g \in\) treated geos | `likelihood_term` or `local prior` | — | If channel not in model |
| **region-level** (region id + member geos) | Region contrast or subset \(\beta_{g,c}\) | `likelihood_term` on member geos | Hyper-mean only via pooling (Decision 8) | Region not in panel |
| **national** (all geos / national aggregation) | \(\mu_c\) hyper-mean | `hyper-prior` or `national calibration likelihood` | \(\tau_c\) weakly | Wrong KPI / estimand |
| **channel-level** (channel scope, any geo) | \(\mu_c\) for channel \(c\) | `hyper-prior` on \(\mu_c\) | Local \(\beta_{g,c}\) via partial pooling | Channel absent from model |
| **segment-level** (non-geo segment) | **Excluded** from posterior by default | `trust_report_only` | — | Unless segment maps to declared geo/channel panel |
| **time-window** (modifier on any scope) | Windows on replay / likelihood | Modifies **likelihood** or **pseudo-observation** support | Freshness (Decision 6) | Window outside panel |

**Segment-level** product or audience tests do not influence geo MMM unless an explicit, approved **scope bridge** is declared in the signal ([bayes_h2b ADR](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md) Decision 5). Default: **TrustReport-only**.

### Rationale

- Mis-scoped evidence is the primary failure mode for GeoX + national MMM coexistence.  
- Hierarchy parameters have different meanings; scope must be explicit before mechanism selection.

### Consequences

- CalibrationSignal schema must carry: `scope_kind`, `geo_ids`, `region_id`, `channel_ids`, `time_window`, `segment_id` (nullable).  
- Bayes-H2b will refine DMA → state → region → national propagation; this ADR sets default targets.

---

## Decision 3 — Prior vs likelihood decision rule

### Decision

The signal registry assigns exactly one **primary mechanism** per included signal using this **ordered rule set**:

| Step | Condition | Mechanism |
|------|-----------|-----------|
| R1 | Estimand incompatible with model estimand registry (Decision 4) | `excluded` → TrustReport |
| R2 | `observational_only: true` without `causal_prior_allowed: true` | `trust_report_only` |
| R3 | Expired freshness (Decision 6) | `excluded` (reported stale) |
| R4 | Missing uncertainty and source requires SE (Decision 5) | `excluded` or weak default per source policy |
| R5 | Replay unit with counterfactual paths + approved estimand | `likelihood_term` or `calibration_penalty` (same class as Ridge replay) |
| R6 | GeoX local lift, geo scope, compatible estimand | `likelihood_term` on local \(\beta_{g,c}\) **or** `local prior` if pre-data / design-only |
| R7 | CLS national lift, national scope | `hyper-prior` on \(\mu_c\) **or** `national calibration likelihood` |
| R8 | A/B product-level lift, scope/estimand misaligned | `trust_report_only` |
| R9 | Conflicting duplicate scope (Decision 7) | Downweight + conflict metric; no silent merge |
| R10 | Holdout panel with full replay geometry | `likelihood_term` via replay-implied lift |

**Default source policies (binding until Bayes-H2b refines):**

| Source | Typical scope | Default mechanism |
|--------|---------------|-------------------|
| **GeoX** | geo-level | `likelihood_term` (post-data); `local prior` only if signal declares `evidence_phase: pre_data` |
| **CLS** | national or region | `hyper-prior` or `national calibration likelihood` |
| **A/B** | national or channel | `hyper-prior` on \(\mu_c\) **if** estimand + scope align; else `trust_report_only` |
| **Replay** | panel-aligned unit | `likelihood_term` / `calibration_penalty` — **must** use replay semantics (Decision 9) |
| **Holdout** | time-window + geo/channel | `likelihood_term` on implied incremental lift |

**Pseudo-observation:** Allowed only when signal declares `pseudo_observation_allowed: true` **and** estimand maps to a scalar incremental target on the modeling scale with declared `lift_scale`. Treat as likelihood equivalent for audit purposes.

**Calibration penalty:** Ridge-comparable soft constraint on replay-implied lift — permitted for research comparison (Bayes-H4); prod default is `likelihood_term` when replay paths exist.

### Rationale

- Prior vs likelihood is an implementation detail; the platform cares about **scope, estimand, and auditability**.  
- Ordered rules prevent ad hoc per-PR mechanism choice.

### Consequences

- Signal registry emits `mechanism`, `rule_id` (R1–R10), and `trace` for TrustReport.  
- Bayes-H3 code must log mechanism assignment per signal at fit time.

---

## Decision 4 — Lift scale and estimand compatibility

### Decision

Every CalibrationSignal **must declare** before inclusion:

| Field | Required | Purpose |
|-------|----------|---------|
| `estimand` | Yes | Registry ID (e.g. geo-time ATT, national lift) |
| `lift_scale` | Yes | Units of lift (e.g. `mean_kpi_level_delta`) |
| `channel` / `channel_ids` | Yes for channel-targeted evidence | Maps to \(\mu_c\) / \(\beta_{g,c}\) |
| `geo scope` | Yes for geo-targeted evidence | Maps to treated geos |
| `time_window` | Yes when post-period defined | Replay and likelihood support |
| `uncertainty` | See Decision 5 | SE, CI, or explicit missing policy |

**Inclusion rule:** A signal influences the posterior **only if**:

1. `estimand` is in the **model estimand allowlist** for this fit, and  
2. `lift_scale` maps to the modeling KPI scale via declared transform (same rules as Ridge replay), and  
3. Treated channels exist in the training panel, and  
4. Geo scope intersects training geos (non-empty).

Otherwise: **`excluded`** with reason code; may appear in TrustReport as `excluded_signals` with `exclusion_reason: estimand_mismatch | lift_scale_mismatch | scope_empty | channel_absent`.

### Rationale

- A/B product lift must not move geo MMM without explicit bridge.  
- Prevents “helpful” experiment lift on wrong estimand from shifting \(\beta_{g,c}\).

### Consequences

- Certification worlds must include **incompatible estimand exclusion** (validation table below).  
- No posterior influence without passing inclusion rule — **hard fail closed** for prod fits when `strict_estimand_gate: true` (default research toward true).

---

## Decision 5 — Uncertainty mapping

### Decision

Uncertainty on experimental lift maps to **evidence strength** in the signal registry:

| Input | Mapping |
|-------|---------|
| `lift_se` or `standard_error` | Precision \(\omega = 1 / \sigma^2\) (or robust Student-\(t\) with \(\sigma\) from SE) for likelihood / pseudo-observation |
| Confidence interval | Convert to SE via declared `ci_level` and distributional assumption (documented in signal) |
| **Lower uncertainty** | **Stronger** influence (higher \(\omega\)) |
| **Missing uncertainty** | Source policy: **weak default** (wide prior / low \(\omega\)) **or** `excluded` if source requires SE (GeoX post-data default: **exclude** unless `allow_missing_uncertainty: true`) |
| **Observational-only** | `trust_report_only` unless `causal_prior_allowed: true` on signal |
| Upper bound / winsor | Cap \(\omega\) at governance max to prevent single experiment domination |

**TrustReport:** Emit `signal_weight_summary` with per-signal effective \(\omega\) or relative weight tier (`high` / `medium` / `low` / `excluded`).

### Rationale

- Missing SE has caused silent over-weighting in generic meta-analyses; platform requires explicit policy.  
- Observational evidence must not pose as causal likelihood without flag.

### Consequences

- `missing-uncertainty` validation world required before Bayes-H3.  
- Freshness downweight (Decision 6) multiplies effective precision, not replaces uncertainty.

---

## Decision 6 — Freshness and decay

### Decision

Evidence **freshness** follows platform calibration governance ([calibration_freshness.md](../04_governance/calibration_freshness.md)) and applies to Bayesian evidence weights:

| Freshness state | Posterior influence | TrustReport |
|---------------|---------------------|-------------|
| **Fresh** (within `calibration_max_age_days` or signal `fresh_until`) | Full weight per Decision 5 | Listed in `included_signals` |
| **Stale** (past max age, within grace) | **Downweighted** (default: multiply precision by 0.25; configurable tier) | `stale_signals` + warning |
| **Expired** (past hard expiry or revoked approval) | **Excluded** from posterior | `excluded_signals` + `stale_signals`; freshness modifier |
| **Missing freshness metadata** | Treat as **stale** with warning | `stale_signals` |

Freshness **must** affect TrustReport trust modifiers (Phase 5E alignment): stale evidence can block optimization via readiness even if posterior still runs.

### Rationale

- Ridge path already uses freshness for readiness; Bayesian must not ignore stale GeoX silently.  
- Expired evidence remains visible for audit — not deleted.

### Consequences

- `stale evidence` validation world required.  
- Replay units must carry `calibration_readiness` / age fields compatible with existing extension reports.

---

## Decision 7 — Conflicting evidence

### Decision

When multiple **included** signals target overlapping scope (same channel + overlapping geos + overlapping time window) with **materially different lift** (relative difference &gt; `conflict_threshold`, default 15% on comparable scale):

The Bayesian adapter **must not** silently average or precision-merge without disclosure.

**Required behavior:**

| Output | Requirement |
|--------|-------------|
| **Conflict metric** | e.g. `max_lift_spread / pooled_se` per scope group |
| **TrustReport** | `conflicting_signals` list + warning; possible `sensitivity_required: true` |
| **Posterior policy** | Default: **downweight all conflicting signals** in the group by factor 0.5 **or** exclude group if `conflict_fail_closed: true` |
| **Sensitivity** | When `sensitivity_required`, fit must record alternate mechanism assignment (e.g. prefer replay over GeoX) in trace — full sensitivity grid is Bayes-H4 |

**Example (binding interpretation):**

- GeoX +8%, CLS +2%, A/B +5% on same channel-national scope → **conflict declared**; none treated as sole truth; TrustReport warns; no single posterior target of 5% “average.”

### Rationale

- Marketing teams experience conflicting experiments frequently; hiding conflict destroys trust.  
- Bayes-H1 forbids using posterior mean contribution as decision — conflict handling is pre-decision hygiene.

### Consequences

- `conflicting evidence` validation world required.  
- INV-066 scope rules in Bayes-H2b may tighten overlap detection for hierarchy levels.

---

## Decision 8 — Local vs global influence

### Decision

**Borrowing rules** for how evidence moves parameters:

| Evidence scope | Primary influence | Hyper-mean \(\mu_c\) | Sparse geos |
|----------------|-------------------|----------------------|-------------|
| Local GeoX | \(\beta_{g,c}\) for treated \(g\) | Only via **partial pooling** from local likelihood — no direct point mass on \(\mu_c\) | Borrow strength from \(\tau_c\) and data |
| National CLS / A/B (aligned) | \(\mu_c\) | Direct prior / likelihood on \(\mu_c\) | All \(\beta_{g,c}\) shrink toward updated \(\mu_c\) |
| Region-level | Member geo \(\beta_{g,c}\) | Optional weak pull on \(\mu_c\) if `region_updates_hyper: true` in signal | Declared in Bayes-H2b |
| Multi-geo test | Listed geos’ \(\beta_{g,c}\) | No national \(\mu_c\) move unless `scope_kind: national` | Per-geo weights from signal |

**Rules:**

1. Local experiment evidence **primarily** influences **local** \(\beta_{g,c}\).  
2. Broad / national evidence influences **hyper-mean** \(\mu_c\) (and indirectly local via pooling).  
3. Sparse geos with no local signal use hierarchical prior centered on \(\mu_c\).  
4. Local evidence **cannot** set national \(\mu_c\) to experiment point estimate without national scope declaration — violates Decision 2.

### Rationale

- Prevents GeoX in one DMA from rewriting national channel priors without explicit scope.  
- Aligns with partial pooling semantics in refinement doc.

### Consequences

- `sparse geo with local experiment` validation world required.  
- Bayes-H2b ADR will formalize DMA → state → region → national propagation.

---

## Decision 9 — Replay compatibility

### Decision

Bayesian calibration mapping **must** preserve existing **replay semantics**:

| Replay invariant | Bayesian requirement |
|------------------|---------------------|
| Full-panel transform | Replay loss / likelihood uses same transform stack as Ridge — no window-slice adstock reset |
| Estimand mask | `full_panel_transform_estimand_mask` and unit `estimand` drive replay comparison |
| Observed vs counterfactual paths | Required when `requires_counterfactual_path: true` |
| Traceability | Each replay-derived term links `signal_id`, `experiment_id`, `payload_sha256` |
| Implied lift | `implied_delta` on declared `lift_scale` — not raw coef difference |

Replay signals use mechanisms R5 / R10 (Decision 3). **Bypassing `simulate()` / replay construction** to inject lift is prohibited (extends Bayes-H1 anti-patterns).

### Rationale

- VAL-006 and certification bundles assume replay geometry.  
- Bayesian “shortcut likelihood” on lift alone without paths breaks comparability with Ridge.

### Consequences

- `replay_calibration_bayes` world family in refinement doc remains valid target.  
- Bayes-H4 must run VAL-006-class checks on Bayesian fits with replay signals.

---

## Decision 10 — TrustReport outputs

### Decision

Bayesian evidence integration **must** populate TrustReport fields (existing object; extensions allowed with schema ADR):

| Field | Content |
|-------|---------|
| `included_signals` | Signals that entered posterior with mechanism + weight tier |
| `excluded_signals` | Signals excluded with `exclusion_reason` |
| `stale_signals` | Stale or expired (may overlap excluded) |
| `conflicting_signals` | Groups in conflict per Decision 7 |
| `signal_weight_summary` | Per-signal effective weight / precision tier |
| `sensitivity_required` | Boolean — conflict or high leverage |
| `posterior_evidence_alignment` | Summary: share of posterior mass on channels/geos with direct experimental support vs pooling-only |

These fields are **diagnostic / trust modifiers** — not optimization inputs (Bayes-H1).

### Rationale

- Operators must see evidence quality without opening sampler logs.  
- Aligns with Phase 5E trust modifier semantics.

### Consequences

- Bayes-H3 artifact schema adds blocks under TrustReport — not DecisionSurface.  
- Reliability scorecard may add REC-Bayes-signal-* checks in Bayes-H4.

---

## Anti-patterns (prohibited)

| Anti-pattern | Why prohibited |
|--------------|----------------|
| **Directly overwriting coefficients** with experiment lift | Breaks hierarchy + DecisionSurface adapter; not auditable |
| **Accepting evidence without estimand check** | Wrong estimand → wrong \(\beta_{g,c}\) / \(\mu_c\) |
| **Using stale signals silently** | Violates Decision 6; freshness trust modifier |
| **A/B influencing geo MMM** when scope/estimand incompatible | Violates Decision 4 |
| **Bypassing CalibrationSignal** | Contract fragmentation |
| **Conflict averaging without warning** | Violates Decision 7 |
| **Observational evidence as causal prior** without `causal_prior_allowed` | Violates Decision 5 |
| **National \(\mu_c\) point mass from local GeoX** | Violates Decision 8 |
| **Window-slice replay** with adstock reset | Violates Decision 9 |
| **Experiment-specific sampler API** | Violates Decision 1 |

---

## Validation requirements (worlds before Bayes-H3)

The following **synthetic validation worlds** must exist (materialized in Bayes-H2b or parallel Track 2 work) **before** sampler implementation is authorized:

| World ID (proposed) | Purpose | Primary checks |
|---------------------|---------|----------------|
| `WORLD-BAYES-GEOX-LOCAL` | Local GeoX prior / likelihood | Local \(\beta_{g,c}\) shift; TrustReport included |
| `WORLD-BAYES-CLS-NATIONAL` | National CLS hyper-prior | \(\mu_c\) recovery direction; no illegal local overwrite |
| `WORLD-BAYES-CONFLICT` | GeoX +8%, CLS +2%, A/B +5% | Conflict metric; no silent average; `conflicting_signals` |
| `WORLD-BAYES-STALE` | Fresh vs stale vs expired | Downweight / exclude per Decision 6 |
| `WORLD-BAYES-MISSING-SE` | Missing uncertainty policies | Exclude or weak per Decision 5 |
| `WORLD-BAYES-SPARSE-GEO` | Sparse geo + local experiment | Pooling borrows; local influence localized |
| `WORLD-BAYES-ESTIMAND-EXCLUDE` | Incompatible estimand A/B | `excluded_signals`; zero posterior shift |

Each world must emit **CalibrationSignal** fixtures traceable to `experiment_truth` per [groundtruth_contract.md](groundtruth_contract.md).

### Implementation checklist (Bayes-H3 gate)

| # | Check |
|---|-------|
| C1 | All evidence via CalibrationSignal only |
| C2 | Scope mapping table enforced (Decision 2) |
| C3 | Mechanism assignment logged (Decision 3) |
| C4 | Estimand / lift_scale gate (Decision 4) |
| C5 | Uncertainty and freshness policies (Decisions 5–6) |
| C6 | Conflict handling (Decision 7) |
| C7 | Local vs global borrowing (Decision 8) |
| C8 | Replay invariants (Decision 9) |
| C9 | TrustReport evidence fields (Decision 10) |
| C10 | DecisionSurface unchanged (Bayes-H1 V1–V7) |

---

## Acceptance criteria (contributor FAQ)

| Question | Answer |
|----------|--------|
| **How does GeoX enter Bayesian MMM?** | As **CalibrationSignal** (geo scope) → default **`likelihood_term`** on local \(\beta_{g,c}\) (or local prior if pre-data). |
| **How does CLS enter?** | As **CalibrationSignal** (national/region scope) → **`hyper-prior`** or **national calibration likelihood** on \(\mu_c\). |
| **How does A/B enter?** | As **CalibrationSignal** — influences posterior **only if** estimand + scope + channel align; else **TrustReport-only**. |
| **When is evidence excluded?** | Estimand/lift/scope mismatch; expired freshness; missing SE when required; observational without flag; revoked approval. |
| **How is uncertainty used?** | SE → precision; lower SE → stronger influence; caps prevent domination. |
| **How is freshness used?** | Fresh = full weight; stale = downweight; expired = exclude but report. |
| **How are conflicts handled?** | Metric + TrustReport warning; downweight or fail closed — **no silent average**. |
| **How does evidence affect local vs national parameters?** | Local scope → \(\beta_{g,c}\); national → \(\mu_c\); sparse geos borrow via pooling only. |
| **How does TrustReport expose evidence quality?** | `included_*`, `excluded_*`, `stale_*`, `conflicting_*`, `signal_weight_summary`, `sensitivity_required`, `posterior_evidence_alignment`. |

---

## Supersession and amendments

| Change type | Process |
|-------------|---------|
| Scope propagation (DMA tree) | **Bayes-H2b ADR** or patch with world evidence |
| New evidence source type | Extend CalibrationSignal schema + mapping row — platform ADR |
| Allow conflict precision-merge without warning | **Rejected** unless new ADR with executive approval |
| New TrustReport fields | Schema ADR + Bayes-H3 doc |

| ADR ID | Supersedes |
|--------|------------|
| `bayes_h2_calibration_signal_mapping_v1` | Draft §3.2–3.4 in [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md) (informal mapping) |

---

## Recommended next phase

**Bayes-H2b** is complete: [bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md](bayes_h2b_hierarchical_experiment_prior_scope_rules_adr.md).

**Next:** Materialize seven `WORLD-BAYES-*` worlds + **Bayes-H2c** hierarchical model design spec. **Bayes-H3** (PyMC) remains blocked until worlds exist in catalog.

---

## References

- [bayes_h1_decision_surface_preservation_adr.md](bayes_h1_decision_surface_preservation_adr.md)  
- [bayesian_hierarchical_geo_mmm_refinement.md](bayesian_hierarchical_geo_mmm_refinement.md)  
- [bayesian_hierarchical_geo_mmm_roadmap.md](bayesian_hierarchical_geo_mmm_roadmap.md)  
- [trust_report_semantics.md](trust_report_semantics.md)  
- [platform_roadmap.md](platform_roadmap.md) — Track 4
