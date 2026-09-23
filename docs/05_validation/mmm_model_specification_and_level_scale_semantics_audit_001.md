# MMM Model Specification and Level-Scale Semantics Audit 001

**Task:** `MMM_MODEL_SPECIFICATION_AND_LEVEL_SCALE_SEMANTICS_AUDIT_001`  
**Status:** research-only audit; no production numerical truth changed  
**Source revision:** `469e65d7f5d03fa1757ac09846b0acf9addba9a2`  
**Evidence archive:** [`MMM_MODEL_SPECIFICATION_AND_LEVEL_SCALE_SEMANTICS_AUDIT_001.json`](archives/MMM_MODEL_SPECIFICATION_AND_LEVEL_SCALE_SEMANTICS_AUDIT_001.json)

## Executive conclusion

The production Ridge path is a pooled log-outcome regression with one global intercept:

```text
log(Y_gt) = alpha + X_media_gt beta + Z_gt gamma + error_gt
```

`X_media` is built from raw media through recursive geometric adstock and Hill saturation;
controls are included only when present in the schema. Geo and week are ordering/state dimensions,
not automatic mean effects. The configured Ridge path therefore does not estimate `alpha_g`,
`beta_gc`, week effects, trend, or seasonality unless those are explicitly supplied as controls.

Ridge level predictions use `exp(mu_hat)` with no Duan smearing or `sigma²/2` retransformation.
The Bayesian comparator uses geo-indexed posterior intercepts and coefficients and returns
`exp(mu + sigma²/2)`, but remains research/diagnostic-only. Canonical planning computes row-level
`exp(mu)` before level aggregation, while canonical decision authority remains full-panel `Delta-mu`
(`mean(mu_plan) - mean(mu_baseline)`). No canonical planning path was found that incorrectly uses
`exp(mean(mu))` as its level aggregation; the Jensen difference is a semantic comparison, not a
confirmed implementation defect.

The H6 pilot audit demonstrates that omitted nuisance structure is material: adding geo effects and
structured time terms sharply improves log and level calibration on the known-truth world. The
near-national TV channel has only `0.00437` of its total variance in within-week cross-geo
variation; an exactly national channel has zero such variation and is rank-deficient with saturated
week effects. On 40 seeded counterfactual plans, `Delta-mu` and corrected expected-level rankings
were almost identical (`Spearman rho` 0.9998 with global smearing and 0.9996 with geo smearing),
so the canonical decision surface remains sufficiently stable on this pilot, but that is not a
certification under heteroskedastic or misspecified worlds.

The evidence supports diagnostic hardening and a staged research program. It does not justify
silently adding geo effects, time effects, smearing, or a level-space replacement for `Delta-mu`.

## Correction preregistration: bounded analysis matrix and interpretation bands

The correction is deliberately a screened matrix, not an exhaustive Cartesian product. The
primary cells are the hypotheses directly implicated by the rejected review; secondary cells are
single-factor screens used to identify whether a finding is likely to be mechanism-specific. Every
new result below is interpreted against the bands registered here, before inspecting its value.

| Analysis | Primary cells | Screened secondary cell | Existing components composed | Excluded combinations and rationale |
|---|---|---|---|---|
| Decision equivalence | H6 full-controls world; canonical full-panel `Delta-mu` versus the already-used global and geo diagnostic level corrections; fixed-pot, incremental, and reallocation constraints | one heteroskedastic/geo-dispersion stress cell | existing H6 materializer, Ridge trainer, planning simulator, optimizer, and economics helpers | no new optimizer, estimator, or full product grid; one stress cell tests sensitivity without certifying every nuisance combination |
| Replay consequence | global baseline versus the existing geo-plus-structured-time research nuisance specification crossed with plain versus valid diagnostic correction | one omitted-control/transform-mismatch replay cell | existing H6 worlds, replay prediction/aggregation helpers, and calibration metrics | aggregation is held canonical in the primary cells so mean-structure and retransformation effects are not confounded; no exhaustive nuisance factorial |
| Media heterogeneity | pooled Ridge comparison on H6 full-controls and sparse-tail slices versus the existing Bayesian partial-pooling research comparator where dimensions are comparable | one sparse-versus-observed geo stability screen | existing H6 truth, Ridge benchmark, Bayesian comparator, and recovery metrics | no unrestricted geo coefficient estimator and no production promotion; incomparable cells are reported as limitations |
| Aggregation estimand | equal-row mean, equal-geo mean, and sum/total KPI on the same plans and replay deltas | population/exposure weighting only if an audited weight exists | existing planning and replay aggregation entry points | no spend weighting and no synthesized population weights; unsupported weights remain unsupported |

### Pre-registered research interpretation bands

These are interpretation bands, not production release thresholds. They are registered to prevent
post-result materiality decisions and are intentionally symmetric with the decision risk of each
measure. Provenance is the audit contract and the existing Tier-1 decision-evidence convention;
the bands are conservative research triage, not a DR-04 promotion rule.

| Metric | Definition | High agreement / low discrepancy | Intermediate | Material divergence |
|---|---|---:|---:|---:|
| Rank agreement | Spearman correlation of plan/channel orderings | `>= 0.95` | `0.80–<0.95` | `< 0.80` |
| Normalized allocation distance | `L1(allocation_A-allocation_B)/(2*total_budget)` | `<= 0.05` | `>0.05–0.15` | `> 0.15` |
| Decision regret | level-objective loss of the `Delta-mu` allocation versus level-objective optimum, divided by the observed candidate objective range | `<= 1%` | `>1–5%` | `> 5%` |
| Hurdle/ROI disagreement | fraction of evaluated near-threshold decisions classified differently | `<= 5%` | `>5–20%` | `> 20%` |
| Economic discrepancy | `abs(Delta_EY_level - Delta_EY_reference)/max(abs(Delta_EY_reference), epsilon)` | `<= 5%` | `>5–10%` | `> 10%` |
| Replay relative-bias gap | absolute difference in relative level bias between candidate and known truth | `<= 2 percentage points` | `>2–5 pp` | `> 5 pp` |

No band below authorizes a production change. Effect-recovery, geo-specific recovery, and
shrinkage are reported as distributions because no defensible production threshold was established
by this audit.

### Internal evidence schema (version 2.0.0)

The archive is an internal, versioned research-evidence artifact, not a package or production
contract. It must validate the following shape before publication:

```text
{
  artifact: {kind, version, task_id, source_revision, research_only},
  worlds: [{world_id, seed, dgp_config_id, dimensions, assumptions}],
  analyses: [{analysis_id, audit_question, candidate_specifications,
              metric_definitions, preregistered_interpretation_bands,
              results, limitations}],
  metrics: [{metric_id, value, units, aggregation_semantics, interpretation_band}],
  provenance: {exact_commands, runtime, source_sha},
  recommendations: [{area, classification, supporting_evidence_ids,
                      successor_milestone}]
}
```

All empirical numbers added by this correction carry a world ID, seed, source SHA, exact command,
metric definition, aggregation semantics, and the applicable registered band.

## 1. Exact current production specification

### Ridge equation and dimensions

`mmm/features/design_matrix.py:110-130` builds the full panel after sorting by `(geo, week)`.
For semi-log (the production Ridge form), the target is `safe_log(y)`; for log-log, the target is
also `safe_log(y)` and only the media columns are additionally log-transformed. Controls are
appended as supplied (`design_matrix.py:126-130`). Media features are constructed at
`design_matrix.py:110-117` with the configured adstock and saturation parameters.

The fitted equation is therefore:

```text
mu_gt = alpha + sum_c beta_c * f_c(raw_spend_gtc) + sum_j gamma_j * Z_gtj
log(Y_gt) = mu_gt + error_gt
```

where `alpha` is one scalar, `beta` is one pooled coefficient per transformed channel, and
`gamma` is one coefficient per supplied control. In the H6 retail pilot, there are 7 channels and
5 controls, yielding coefficient dimension 12. The Ridge trainer's `pooling` configuration does
not create geo-varying Ridge coefficients; geo-specific alpha/beta behavior exists in the separate
Bayesian comparator.

Geo and week determine sorting and recursive adstock state. They are not automatically included as
fixed effects. A time effect, holiday, trend, or seasonality enters only if represented by a control
column. Ridge minimizes squared error on the log target; it has no explicit residual variance,
heteroskedasticity, geo random effect, or temporal residual model.

### Inverse transform and comparator

`mmm/models/ridge_bo/trainer.py:456-460` returns `np.exp(yhat_log)` for both model forms. There is
no residual smearing factor and no `sigma²/2` correction. This returns a raw-KPI-unit value, but
under a log-error model it is the inverse of the conditional log mean (median-like/geometric-mean
retransformation), not automatically the conditional arithmetic mean.

`mmm/models/bayesian/pymc_trainer.py:371-382` is different: it uses geo-indexed `alpha_geo`,
geo-varying `beta` under partial pooling, and returns `exp(mu + 0.5*sigma²)`. This is a parametric
lognormal mean correction, not Duan smearing, and the Bayesian path is explicitly research-only.

## 2. Panel mean structure audit

The H6 DGP intentionally contains persistent geo baselines, seasonality, shocks, geo-varying media
effects, sparse geos, and a weakly varying national TV channel (`production_shapes.py:90-168,
220-312`). The current Ridge specification omits the geo/time baseline portion unless those fields
are supplied as controls.

Using the full-controls H6 retail pilot (20 geos × 52 weeks; research-only OLS comparison on the
known transformed media features), the metrics `(log RMSE, relative level bias, rank, columns)` were:

| Mean structure | Result |
|---|---:|
| Global intercept + pooled media | `(0.4103, -8.21%, 13, 13)` |
| Geo fixed effects + pooled media | `(0.1411, -0.93%, 32, 32)` |
| Geo effects + Fourier(52-week) + trend | `(0.1373, -0.88%, 35, 35)` |
| Geo effects + saturated week effects | `(0.1324, -0.82%, 83, 83)` |

These are not model-selection results. They show that omitted baselines dominate the global model's
error on this world and that time nuisance structure matters, while saturated weeks buy fit at a
large degrees-of-freedom and identification cost. A future design should compare structured trend,
spline, Fourier, and known-event controls against saturated weeks using causal/effect recovery,
stability, and identification—not predictive RMSE alone.

## 3. Media-effect heterogeneity

The current production Ridge path estimates pooled `beta_c`. The relevant alternatives are:

1. pooled `beta_c`;
2. regularized `beta_gc = beta_c + delta_gc` with shrinkage;
3. time-varying `beta_ct` or low-rank time deviations;
4. fully geo-time-varying effects.

H6 truth already contains geo-varying `true_beta_gc`, but the existing Ridge benchmark reports only
coarse pooled recovery and does not certify geo-effect recovery. Fully varying effects are not
credible with short or sparse geo histories; unregularized variants invite severe overfit and
confounding. The Bayesian roadmap's partial-pooling design is a research comparator, not current
Ridge authority.

**Recommendation:** `requires_more_evidence`. Add a research-only certification matrix that varies
geo count, weeks, heterogeneity, sparsity, and cross-geo media variation, scoring pooled recovery,
geo-effect recovery, stability, and decision regret. Only then consider a separately authorized
regularized geo-varying implementation.

## 4. National-channel identification

For each H6 channel, the audit computed:

```text
Var(X_gtc - mean_g(X_gtc)) / Var(X_gtc)
```

Results were: TV `0.00437`, CTV `0.02126`, Display `0.02044`, Local flyer `0.98342`, Radio
`0.98608`, Search `0.94394`, and Social `0.93890`. TV is therefore near-national in the H6 world;
most local/digital channels have substantial within-week geo variation.

For an exactly national channel, setting every geo's weekly media value to the same weekly series
produces zero within-week residual variation. With an intercept plus saturated week dummies, the
design had 53 columns but rank 52: the national channel is unidentified separately from week.
Structured time effects can recover estimability only by imposing assumptions about the time path;
they do not create causal variation. This is a reason to report identification warnings, not a
reason to remove confounding controls.

**Recommendation:** `supported_for_implementation` for a diagnostic-only identifying-variation
report and a scope-aware warning/refusal policy. The production coefficient policy itself is
`requires_more_evidence` and must be certified under exact-national, near-national, mixed, and
strongly geo-varying worlds.

## 5. Retransformation audit

The affected surfaces are:

- Ridge trainer prediction: plain `exp(mu)` (`mmm/models/ridge_bo/trainer.py:456-460`).
- Ridge replay/calibration helpers: plain `exp(ylog)` (`mmm/evaluation/replay_holdout_validation.py:68-78`,
  `mmm/calibration/replay_bo_objective.py:81-91`, and `mmm/evaluation/calibration_extension.py`).
- Canonical planning level summary: row-level `exp(mu)` followed by mean (`mmm/planning/mu_path.py:234-242`).
- Bayesian comparator: `exp(mu + 0.5*sigma²)` (`mmm/models/bayesian/pymc_trainer.py:382`).
- Decomposition: log-scale additive contributions explicitly marked non-business-value
  (`mmm/decomposition/engine.py:60-90`).

On the H6 full-controls pilot, plain exponentiation had `-8.58%` relative level bias. A global
normal/lognormal correction reduced bias to `-0.24%`; global Duan smearing `1.09195` reduced it to
`-0.17%`. However, geo-specific smearing factors ranged from `0.570` to `2.054`, demonstrating that
a single correction can hide substantial geo heterogeneity. These values are diagnostic only: a
residual factor estimated before correcting omitted geo/time mean structure must not be presented as
pure noise correction.

**Recommendation:** `requires_more_evidence`. First settle mean-structure diagnostics; then compare
parametric and Duan/conditional retransformation on held-out worlds with heteroskedastic and
geo/time-varying residuals. Artifact semantics should explicitly distinguish `exp(mu)` from
expected-level KPI and record any correction/version.

## 6. Aggregation order

The alleged canonical `exp(mean(mu))` defect is **rejected as an implementation finding**. The
canonical planner computes `mean_mu` as a mean of row-level `mu` (`mu_path.py:59-70`), but computes
`mean_kpi_level` as `mean(exp(mu))`, or mean of per-geo row-level `exp(mu)` (`mu_path.py:234-242`).
The decision result then sets `delta_mu = plan_mu - baseline_mu` (`decision_simulate.py:301-303`).

Replay also aggregates row-level predicted differences according to its declared estimand
(`replay_estimand.py:107-132`). Curve/decomposition surfaces are explicitly diagnostic and have
different approximation semantics. Thus, the Jensen gap is real mathematics—on the audited pilot
`mean(exp(mu))/exp(mean(mu)) = 1.01357`—but the canonical level summary does not introduce a second
Jensen penalty by using `exp(mean(mu))`.

## 7. Delta-mu versus expected level-space economics

The canonical decision surface remains full-panel `Delta-mu`. On 40 seeded H6 counterfactual plans
(seed 991), rank agreement was:

| Comparison | Spearman rank correlation |
|---|---:|
| `Delta-mu` vs global-smear expected-level delta | 0.99981 |
| `Delta-mu` vs geo-smear expected-level delta | 0.99962 |

The seven one-at-a-time +10% channel rankings were identical under `Delta-mu`, global-smear level,
and geo-smear level. This supports retaining `Delta-mu` for the audited world and plan family. It
does not establish optimizer or hurdle invariance under strong heteroskedasticity, omitted
structure, or materially different geo weighting. No optimizer objective or decision contract was
changed.

The bounded correction analysis used the repository's existing full-panel SLSQP optimizer for the
canonical objective and the same repository SLSQP constraint family for a diagnostic expected-level
objective. On the H6 full-controls world, normalized allocation distance was `0.26118` (material
under the preregistered `>0.15` band), while expected-level regret of the canonical allocation was
`0.0` on this local level surface. The diagnostic level optimizer reported a distinct allocation
despite no measured regret, so this remains research evidence rather than a production invariance
claim. No defensible production materiality threshold was invented.

**Recommendation:** `requires_more_evidence`: add a dedicated decision-equivalence certification
with heteroskedastic geo/time worlds, conditional corrections, constrained optimizer allocations,
and hurdle-rate decisions before considering any level-space decision surface.

### 7.1 Bounded correction evidence and interpretation

The preregistered metrics were rank agreement, normalized allocation distance, decision regret,
hurdle/ROI disagreement, and economic discrepancy. The original 40-plan Spearman results remain
high agreement; the new optimizer allocation distance is material, while measured level-objective
regret is high agreement on this one world. Allocations can differ along a flat or locally
non-identifiable surface while achieved objective is nearly unchanged. Hurdle/ROI classification
was not certified because no independent threshold set or production hurdle materiality bar exists
in this audit.

## 8. Replay and calibration consequences

Replay prediction helpers use the same plain Ridge `exp(ylog)` inverse. Replay then aggregates level
differences over an explicit geo/week estimand, so aggregation semantics are declared rather than
implicitly global. The main risks are upstream: omitted geo/time baselines, transform mismatch,
residual retransformation bias, and media/control misspecification can all appear as replay gaps.

The H6f evidence already records omitted-control worlds as forbidden for incrementality claims and
shows transform mismatch and sparse/collinear fragility. It does not yet certify retransformation,
geo-baseline recovery, or expected-level replay lift. CalibrationSignal and experiment compatibility
semantics remain unchanged.

The controlled replay comparison reports measured level-bias consequences separately from
mechanisms that remain unisolated. The global/plain cell has `-8.58%` relative level bias; the
diagnostic global-smear cell has `-0.17%`; the existing geo-plus-structured-time plain cell has
`-0.88%`. These show that retransformation and nuisance specification both matter, but do not
identify a causal replay-lift mechanism by themselves. A valid apples-to-apples Bayesian
partial-pooling run was not possible in the focused runtime because `pymc` is unavailable; no
replacement estimator or framework was introduced.

**Recommendation:** `requires_more_evidence`: extend replay worlds with known geo/time nuisance,
heteroskedastic residuals, explicit level truth, and both row-level and estimand-level acceptance
metrics. Keep replay evidence routed through existing CalibrationSignal semantics.

## 9. Synthetic DGP parity and empirical evidence

H6 source verifies known geo baselines, seasonal terms, shocks, geo-varying media effects, sparse
tail geos, collinearity blocks, vertical controls, and weakly varying national TV. Existing H6f
artifacts cover five pilot worlds and explicitly keep all outputs diagnostic-only. The audit's
deterministic research comparisons used `WORLD-H6-PILOT-RETAIL-FULL-CONTROLS` (20 × 52 × 7), the
existing H6 materializer, and Ridge BO in research environment; commands and values are recorded in
the JSON archive.

The H6 generative truth uses adstock plus Hill saturation and geo-varying beta, while the current
Ridge baseline uses a global intercept and pooled coefficients. That mismatch is itself evidence:
prediction quality or average coefficient recovery cannot certify geo-effect or level-KPI recovery.

## 10. Why prior validation did not surface this

Prior H6/H6f and Tier-1 validation tested implementation contracts, predictive RMSE/WMAPE, geo-fold
stability, coarse pooled coefficient/lift recovery, collinearity, sparse channels, omitted controls,
transform mismatch, replay, and structural/recovery metrics. Those audits were intentionally not
acceptance tests for:

- recovery of geo baselines or a chosen time nuisance specification;
- retransformation calibration to arithmetic level KPI;
- geo-conditional residual structure or smearing stability;
- expected-level KPI and replay-lift recovery under nuisance misspecification;
- national-channel rank/conditioning under saturated week effects;
- `Delta-mu` versus corrected expected-level optimizer/hurdle invariance.

This is a scope gap, not evidence that the earlier validators were incorrectly implemented. Their
artifacts correctly state diagnostic-only boundaries and promotion blockers.

## 11. Recommendations and successor milestones

| Area | Classification | Independently reviewable successor |
|---|---|---|
| Geo baselines | `requires_more_evidence` | H6 nuisance-structure certification, then design ADR |
| Time baseline family | `requires_more_evidence` | structured-vs-saturated time recovery study |
| Pooled vs geo-varying effects | `requires_more_evidence` | partial-pooling recovery/stability matrix |
| National-channel variation | `supported_for_implementation` for diagnostics; policy `requires_more_evidence` | identifying-variation artifact and scope-aware warning policy |
| Residual/noise diagnostics | `supported_for_implementation` as diagnostics only | geo/time/heteroskedastic residual report |
| Retransformation | `requires_more_evidence` | held-out retransformation certification and artifact semantics ADR |
| Aggregation | `supported_for_implementation` as current semantics | semantic regression tests; no production correction needed from this audit |
| Replay | `requires_more_evidence` | nuisance-aware level replay certification |
| Full-panel Delta-mu | `supported_for_implementation` for audited pilot; broader policy `requires_more_evidence` | decision-equivalence certification under stress worlds |
| H6/Tier-2 additions | `supported_for_implementation` | add nuisance, national-identification, level-truth, and optimizer worlds |

No recommendation authorizes implementation. Each successor must preserve the current production
decision contract until independently reviewed and authorized.

## 12. DR-04 handoff and correction boundaries

DR-04 retained most quantitative/recovery thresholds as provisional because the available evidence
was insufficient. This audit does not alter any DR-04 ruling and promotes no registry row. The
successor evidence categories for the next applicable threshold/promotion cycle are: decision
equivalence under independent stress worlds; nuisance-aware level replay; geo/time baseline and
media-heterogeneity recovery; national-channel identification; aggregation-estimand semantics;
and exact evidence reproducibility. The recommendations are inputs to that future cycle, not
authority to change its thresholds.

The correction preserved all accepted findings unless new deterministic evidence qualified them:
plain Ridge `exp(mu)`, row-level `mean(exp(mu))`, the rejected second Jensen defect, near-national
identification loss, and the non-certifying 40-plan pilot remain unchanged. The optimizer result
adds a material allocation-distance signal without changing `Delta-mu` authority.

## Validation and boundaries

Focused evidence used deterministic Docker Python 3.11 inline commands against the existing H6
materializer and source paths. The JSON archive parses and mirrors the report's values. No source,
test, production configuration, replay implementation, optimizer, economics formula, contract,
MIP, GeoX, or release threshold was changed by this audit.

Correction validation also reruns every exact command recorded in the evidence archive, JSON schema
validation, documentation/lifecycle checks, `git diff --check`, and the repository Docker-backed
`make validate` gate. The Bayesian partial-pooling command is recorded as an expected runtime
limitation (`pymc` unavailable) rather than silently substituted.
