# MMM Geo/Time Nuisance-Structure Certification 001

**Task:** `MMM_GEO_TIME_NUISANCE_STRUCTURE_CERTIFICATION_001`  
**Source revision:** `6be3fa2879aca65c3c7268a8133fa35b90f999ec`  
**Status:** research-only certification; no production numerical behavior changed  
**Evidence archive:** [`MMM_GEO_TIME_NUISANCE_STRUCTURE_CERTIFICATION_001.json`](archives/MMM_GEO_TIME_NUISANCE_STRUCTURE_CERTIFICATION_001.json)

## Executive conclusion

The bounded H6 evidence supports making a geo baseline an explicit research/production-design
candidate before retransformation work, but it does not yet authorize changing the production Ridge
mean structure. The global-intercept reference leaves nearly all persistent geo pattern in residuals
(geo residual-pattern ratios approximately `0.95–1.00`) and has high residual serial correlation
(`lag-1` approximately `0.84–0.89`). Adding geo effects removes the measured residual geo pattern
and sharply reduces serial correlation, while recovering channel signs more reliably on the
full-controls worlds.

A single bounded Fourier/trend basis is the best screened structured-time candidate: it reduces
time-pattern contamination relative to geo-only and recovers the known smooth seasonal baseline
better than the saturated-week comparator. Saturated week effects minimize in-sample RMSE and
remove time residual pattern mechanically, but have materially worse time-baseline recovery,
higher degrees of freedom, and are unsafe for national-channel identification. The exact-national
TV stress design with an intercept and saturated week effects is rank deficient (`64` columns,
rank `63`); estimability under structured time comes only from stronger time-path assumptions, not
new causal variation.

The candidate nuisance columns are compatible with full-panel intervention semantics: baseline and
candidate use the same nuisance design and coefficient vector while only media columns change.
On the full-controls H6 world, the known-truth all-channel +10% Delta-mu was `0.00661`; candidate
values were global `0.01734`, geo `0.02577`, Fourier/trend `0.02460`, and saturated-week `0.00152`.
This is a recovery warning, not a reason to change canonical full-panel Delta-mu authority.

## 1. Preregistered bounded matrix and bands

The matrix was registered in the Git-authored task before empirical execution. It was intentionally
not an exhaustive Cartesian product:

| Candidate | Role |
|---|---|
| Global intercept + pooled media + supplied controls | Current production reference |
| Geo effects + pooled media + supplied controls | Persistent geo-baseline candidate |
| Geo effects + Fourier/trend basis + pooled media + controls | Primary structured-time screen |
| Geo effects + saturated week effects + pooled media | Research identification stress comparator |
| Regularized geo baseline | Not run; no existing apples-to-apples implementation was available |

Retransformation, optimizer configuration, media pooling, and Delta-mu authority were held fixed.
The five existing H6 worlds were used: retail full-controls, retail omitted-controls, retail
media-correlated-controls, CPG full-controls, and auto omitted-controls. They all contain 20 geos,
52 weeks, TV as the near-national channel, and sparse radio/local-flyer tails; the DGP source also
contains geo baselines, seasonality, shocks, and geo-varying media truth.

Research interpretation bands were fixed before runs: geo/time recovery error `<=10%` of truth SD
high, `>10–25%` intermediate, `>25%` material; pooled-beta relative error `<=10%`, `>10–25%`,
`>25%`; Delta-mu absolute error `<=0.02`, `>0.02–0.05`, `>0.05`; coefficient instability
`<=10%`, `>10–25%`, `>25%`; residual geo/time pattern reduction `>=80%`, `50–<80%`, `<50%`.
National-channel severity remains descriptive (`none`, `caution`, `non-identifiable`) rather than a
release threshold.

## 2. Evidence summary

### 2.1 Nuisance and residual structure

The following summaries are from the exact deterministic five-world run in the evidence archive.
Values are representative ranges across worlds; the archive retains per-world/per-candidate rows.

| Candidate | Geo recovery error | Fourier/time recovery error | Residual geo ratio | Residual time ratio | Lag-1 residual correlation | RMSE behavior |
|---|---:|---:|---:|---:|---:|---|
| Global | `1.000` | `1.000` | `0.954–1.003` | `0.126–0.514` | `0.838–0.892` | Worst, but RMSE is secondary |
| Geo | `0.402–0.672` | `1.000` | approximately `0` | `0.119–0.507` | `-0.029–0.077` | Large improvement |
| Geo + Fourier/trend | `0.397–0.675` | `0.212–0.571` | approximately `0` | `0.048–0.349` | `-0.076–0.024` | Further improvement |
| Geo + saturated week | `0.448–0.665` | `1.126–2.801` | approximately `0` | approximately `0` | `-0.053–-0.005` | Lowest in-sample RMSE, but overfit/identification cost |

Geo effects remove geo-pattern residual variance because they absorb the persistent baseline by
construction. Fourier/trend reduces time contamination and recovers the smooth seasonal truth;
saturated weeks remove residual time pattern by absorbing every week, but do not recover the smooth
truth and introduce many degrees of freedom.

### 2.2 Media recovery and identification

Pooled channel sign recovery was `0.714` for the global reference on several worlds, improved to
`0.857–1.000` for geo/Fourier candidates on full-controls worlds, and remained weak in omitted-
control worlds. This confirms that nuisance correction cannot rescue omitted controls or confounded
media. It is not a causal certification.

Within-week variation ratios consistently identify TV as near-national (`0.0044–0.0056`), while
search/social/radio/local-flyer are strongly geo-varying (roughly `0.93–0.99`); CTV/display are
mixed-to-low (`0.02–0.03`). These are estimability diagnostics, not causal-identification claims.

For an exact-national TV stress, the design containing an intercept, transformed media, supplied
controls, and saturated week effects had `64` columns and rank `63`: TV is not separately
identified from saturated week effects. Near-national TV remains ill-conditioned rather than
exactly singular. Structured time preserves a coefficient only by assuming a restricted time path;
it does not manufacture cross-geo causal variation.

### 2.3 Full-panel intervention compatibility

The bounded full-controls check held nuisance columns and fitted coefficients fixed while changing
only all media columns by `+10%`. The known-truth Delta-mu was `0.006607`; fitted candidate values
were: global `0.017342`, geo `0.025774`, Fourier/trend `0.024603`, saturated week `0.001517`.
This verifies the construction semantics (`mu_base` and `mu_candidate` share nuisance state), while
showing that nuisance choice materially affects recovery. Canonical full-panel Delta-mu remains
unchanged and no optimizer/economics path was modified.

## 3. Recommendations

| Area | Classification | Evidence-backed recommendation | Successor |
|---|---|---|---|
| Geo-baseline specification | `requires_more_evidence` | Keep global Ridge as current reference; carry geo-baseline treatment into a separately authorized design/certification milestone. | independent geo-baseline design with out-of-sample and sparse-geo bars |
| Time-baseline family | `requires_more_evidence` | Prefer structured time candidates for further study; Fourier/trend is the leading screened candidate, not a production decision. | structured-time recovery and holdout certification |
| Saturated-week policy | `rejected` as default production nuisance for national media; `research_only` comparator | Do not select by RMSE; prohibit silent use where national-channel rank/conditioning is lost. | explicit identification diagnostic/policy task |
| National-channel policy dependency | `supported_for_implementation` for diagnostics only | Emit variation/rank/conditioning evidence and scope-aware caution/non-identifiable labels. | diagnostic contract and fixture task |
| Residual readiness for retransformation | `requires_more_evidence` | Geo + structured-time residuals are more suitable than global residuals, but omitted-control and heteroskedastic worlds remain contaminated. | held-out retransformation certification |
| Full-panel Delta-mu compatibility | `supported_for_implementation` as unchanged contract; broader recovery `requires_more_evidence` | Preserve nuisance-fixed intervention semantics and canonical Delta-mu; do not infer production equivalence from this screen. | Delta-mu recovery across independent nuisance worlds |

Regularized geo-baseline candidate: `research_only`; no existing apples-to-apples implementation was
available within the owned paths, so no new estimator was introduced. Bayesian authority remains
unchanged. No conclusion authorizes production implementation or alters DR-04.

## 4. Limitations and prior-validation connection

These are in-sample research comparisons using existing H6 worlds and a bounded research Ridge
design. The sparse/short-history geos are represented by existing H6 tails, but no broad sample-size
grid was added. The Fourier/trend basis is a stronger modeling assumption than a saturated week
effect; its apparent recovery advantage is not causal proof. Omitted-control worlds correctly remain
non-certifying. The exact-national rank demonstration uses a deliberately stressed design and is a
policy warning, not evidence to remove confounding controls.

Earlier validation measured implementation contracts, predictive fit, and coarse pooled recovery;
it did not certify nuisance recovery, residual contamination, or national-channel identification.
This certification fills that evidence gap without changing production behavior.

## 5. Exact reproduction and schema

The machine-readable archive uses the merged audit's internal research-evidence pattern (`schema_id`
`mmm_research_evidence_v1`, artifact version `1.0.0`). The primary command was:

```bash
docker run --rm -e PYTHONPATH=/repo \
  -v /tmp/geo_audit_run.py:/tmp/geo_audit_run.py \
  -v /private/tmp/mmm-geo-time-nuisance-structure-certification-001:/repo \
  -w /repo mmm-fixture-ready:local python /tmp/geo_audit_run.py
```

The inline research program composes `materialize_h6_panel`, `h6_ridge_config`, `h6_panel_schema`,
`RidgeBOMMMTrainer`, `build_design_matrix`, existing transforms, and sklearn Ridge for the bounded
research candidates. The full-panel compatibility command and its output are recorded in the JSON
provenance. The temporary orchestration file is not a repository artifact and no new harness path
was added to the owned tree.

## 6. Boundary confirmation

No production Ridge fitting/prediction, Bayesian authority, transforms, retransformation, replay,
CalibrationSignal, TrustReport, optimizer, economics, full-panel Delta-mu semantics, public/package
contract, MIP, GeoX, DR-04 threshold, or production default changed.

**Unresolved execution-blocking design questions:** none.
