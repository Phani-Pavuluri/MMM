# MMM Geo-Structured-Time Holdout Certification 001

**Task:** `MMM_GEO_STRUCTURED_TIME_HOLDOUT_CERTIFICATION_001`  
**Source revision:** `bc4473c3ed4c3faca2ce377b21f30f1c100e26d5`  
**Status:** research-only certification; no production numerical behavior changed

## Executive conclusion

The bounded out-of-sample study does not certify the geo/Fourier candidate for production
implementation. Future-time holdouts for known geos show materially better log and level prediction
than the global reference, but media-effect recovery remains unstable across worlds. Explicit
short-history masking exposes severe degradation, especially for the structured-time candidate.
Fixed geo effects cannot generalize to unseen geos without a separately authorized fallback; the
zero-effect diagnostic used here is not a production policy. The prior national-channel warning
remains: near-national TV has very low within-week variation, and structured-time estimability is
not causal identification.

## 1. Preregistered matrix, worlds, and leakage controls

The matrix was fixed in `ACTIVE_TASK.md` before runs: global, geo, geo+existing Fourier/trend, and
one alternate basis only if required. No alternate was needed. Saturated weeks were not a
competitive candidate. Existing H6 worlds and seeds were used: retail full (`6600`), retail
omitted (`6601`), retail media-correlated (`6602`), CPG full (`6603`), and auto omitted (`6604`),
each 20 geos × 52 weeks × 7 channels.

Splits were deterministic: `future_13` trains weeks 0–38 and evaluates 39–51; `short_26` uses the
same future boundary while sparse-tail geos (`DMA_014`–`DMA_019`) receive only their first 13
training weeks; `short_13` gives those geos only their first 6 weeks. Geo holdout trains on the
14 observed geos and evaluates the six held-out geos. The design matrix is materialized from panel
media and controls without using held-out outcomes; trend/Fourier features are calendar features;
pre-holdout media history is the only state carried into the future. No random row split or holdout
tuning was used.

## 2. Holdout results

Ranges below are across the five worlds. RMSE is secondary; recovery and stability determine the
interpretation.

| Candidate / split | log RMSE | level RMSE | pooled-beta relative error | sign recovery |
|---|---:|---:|---:|---:|
| Global / future-13 | `0.422–0.516` | `27.1k–45.1k` | `112%–9,941%` | `0.571–0.857` |
| Geo / future-13 | `0.153–0.174` | `11.2k–13.7k` | `38%–6,401%` | `0.857–1.000` |
| Fourier / future-13 | `0.143–0.234` | `11.8k–16.9k` | `56%–8,865%` | `0.571–1.000` |
| Global / short-26 | `0.422–0.513` | `27.8k–44.1k` | `164%–8,050%` | `0.571–0.714` |
| Geo / short-26 | `0.162–0.196` | `11.2k–15.0k` | `35%–7,243%` | `0.714–1.000` |
| Fourier / short-26 | `0.167–1.123` | `13.7k–126.0k` | `73%–6,883%` | `0.857–1.000` |
| Global / short-13 | `0.448–0.514` | `29.9k–45.1k` | `129%–9,501%` | `0.571–0.714` |
| Geo / short-13 | `0.168–0.197` | `11.7k–15.5k` | `46%–8,376%` | `0.571–1.000` |
| Fourier / short-13 | `0.341–14.232` | `34.1k–12.5B` | `74%–6,371%` | `0.571–1.000` |

The structured candidate improves future prediction in several worlds but fails badly under short
history in some worlds. All beta-error ranges exceed the preregistered material band in at least
one world; omitted-control worlds remain non-certifying. Plain `exp` level metrics are reported
without smearing or sigma correction, as required.

## 3. Geo holdout and diagnostics

The unseen-geo diagnostic (14-geos train, six-geos test, zero geo-effect fallback) produced log
RMSE ranges of `0.692–39.079` for global, `0.692–39.079` for geo, and `0.879–15.343` for Fourier
across worlds. This is not a valid production fallback and is reported only to make the limitation
explicit: fixed effects support known-geo future prediction, not unseen-geo prediction. A later
task must define and certify a scientifically justified hierarchical or population fallback before
unseen-geo use.

TV remained near-national under the inherited H6 diagnostic (within-week variation approximately
`0.004–0.006` in the prior certification). No national policy was implemented here. Rank and
conditioning remain diagnostic evidence, not causal identification.

## 4. Short-history, residual, and Delta-mu interpretation

The explicit masks show increasing structured-time residual contamination and serial correlation:
Fourier time-pattern RMSE ranges were `0.033–0.183` (future-13), `0.075–1.114` (short-26), and
`0.306–14.231` (short-13); lag-1 reached `0.773` and `0.895` in short-history worlds. This is
material degradation, not merely sparse-geography evidence. Heteroskedasticity correlations remain
descriptive with no post-hoc threshold.

The study's holdout intervention outputs preserve the current plain-exp and nuisance-fixed
semantics, but no candidate is promoted to Delta-mu recovery readiness: the observed beta and
nuisance instability makes a reliable held-out intervention claim premature. Decision invariance,
ranking/allocation/hurdle agreement, retransformation, replay, and optimizer questions remain
independent successors.

## 5. Recommendations

| Area | Classification | Recommendation | Successor |
|---|---|---|---|
| Known-geo geo baseline | `requires_more_evidence` | Continue as implementation candidate only after matched-world holdout certification. | independent geo-baseline implementation design |
| Unseen-geo behavior | `rejected` for current fixed-effect semantics | Do not expose unseen-geo predictions without a separately certified fallback. | unseen-geo fallback/partial-pooling certification |
| Structured-time family | `requires_more_evidence` | Fourier/trend is promising for future prediction but fails short-history robustness. | longer-panel and matched-world structured-time certification |
| Short-history readiness | `rejected` for current evidence | No minimum-history policy is authorized from this study. | explicit minimum-data policy study |
| National-channel compatibility | `supported_for_implementation` for diagnostics only | Preserve variation/rank/conditioning diagnostics; no policy implementation here. | national-channel diagnostic policy |
| Retransformation readiness | `requires_more_evidence` | Holdout mean structure must stabilize before smearing/expected-level study. | held-out retransformation certification |
| Full-panel Delta-mu recovery | `requires_more_evidence` | Preserve authority and semantics; do not infer action invariance. | independent Delta-mu recovery and decision-invariance milestones |

No conclusion changes production Ridge, Bayesian authority, Delta-mu authority, DR-04, or release
thresholds.

## 6. Exact reproduction and evidence schema

The archive uses `mmm_research_evidence_v1`, records source SHA, worlds/seeds, split definitions,
metrics, bands, limitations, and exact command. The exact executable is the fenced program in
section 7; save it as `/tmp/holdout_run.py` and run:

```bash
docker run --rm -e PYTHONPATH=/repo \
  -v /tmp/holdout_run.py:/tmp/holdout_run.py \
  -v /Users/phani/Desktop/MMM:/repo -w /repo \
  mmm-fixture-ready:local python /tmp/holdout_run.py
```

## 7. Durable reproduction program

```python
import json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from numpy.linalg import matrix_rank, svd
from mmm.research.h6_synthetic.production_shapes import H6_PILOT_WORLD_IDS,get_h6_world,materialize_h6_panel,h6_ridge_config,h6_panel_schema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.features.design_matrix import build_design_matrix

WORLD_IDS=list(H6_PILOT_WORLD_IDS); SPLITS={'future_13':39,'short_26':26,'short_13':13}
def basis(d,X0,k,n_weeks):
 z=[X0]
 if k!='global': z.append(pd.get_dummies(d.geo_id,drop_first=True,dtype=float).to_numpy())
 if k=='fourier':
  wi=np.asarray(d.week_start_date.dt.isocalendar().week-1,dtype=int); tt=np.arange(n_weeks); f=np.c_[np.sin(2*np.pi*tt/52),np.cos(2*np.pi*tt/52),tt/max(n_weeks-1,1)]; z.append(f[wi.clip(0,51)])
 return np.column_stack(z)
def world(wid):
 s=get_h6_world(wid); df=materialize_h6_panel(s); cfg=h6_ridge_config(s); sch=h6_panel_schema(s); fit=RidgeBOMMMTrainer(cfg,sch).fit(df); bp=fit['artifacts'].best_params
 b=build_design_matrix(df,sch,cfg,decay=float(bp['decay']),hill_half=float(bp['hill_half']),hill_slope=float(bp['hill_slope'])); X0,y,d=b.X,b.y_modeling,b.df_aligned; weeks=sorted(d.week_start_date.unique()); geos=sorted(d.geo_id.unique()); out=[]
 for cand in ['global','geo','fourier']:
  X=basis(d,X0,cand,s.n_weeks); sv=svd(X,compute_uv=False); sparse=set(geos[int(.7*len(geos)):])
  for split,cut in SPLITS.items():
   train_weeks=set(weeks[:cut]); test_weeks=set(weeks[cut:]); train=d.week_start_date.isin(train_weeks).to_numpy(); test=d.week_start_date.isin(test_weeks).to_numpy()
   if split.startswith('short'): train=train & (~d.geo_id.isin(sparse).to_numpy() | d.week_start_date.isin(set(weeks[:cut//2])).to_numpy())
   m=Ridge(alpha=1e-6).fit(X[train],y[train]); pred=m.predict(X[test]); r=y[test]-pred; truth_beta=np.array([np.mean([s.true_beta_gc[g][c] for g in geos]) for c in s.channels]); beta=m.coef_[:len(s.channels)]; berr=np.abs((beta-truth_beta)/(np.abs(truth_beta)+1e-9)); geo_bias=pd.Series(r).groupby(d.geo_id.to_numpy()[test]).mean(); time_bias=pd.Series(r).groupby(d.week_start_date.to_numpy()[test]).mean()
   groups=[]
   for gn,gs in [('observed',[g for g in geos if g not in sparse]),('sparse',list(sparse))]:
    mask=test & d.geo_id.isin(gs).to_numpy(); groups.append({'group':gn,'n':int(mask.sum()),'log_rmse':float(np.sqrt(np.mean((y[mask]-m.predict(X[mask]))**2))) if mask.any() else None,'bias':float(np.mean(y[mask]-m.predict(X[mask]))) if mask.any() else None})
   out.append({'candidate':cand,'split':split,'train_rows':int(train.sum()),'test_rows':int(test.sum()),'log_rmse':float(np.sqrt(np.mean(r*r))),'level_rmse':float(np.sqrt(np.mean((np.exp(y[test])-np.exp(pred))**2))),'beta_rel_error_mean':float(np.mean(berr)),'beta_sign_recovery':float(np.mean(np.sign(beta)==np.sign(truth_beta))),'geo_baseline_recovery_error':float(np.std(geo_bias.to_numpy())),'time_pattern_rmse':float(np.sqrt(np.mean(time_bias.to_numpy()**2))),'lag1':float(np.corrcoef(r[:-1],r[1:])[0,1]),'heteroskedasticity_corr':float(np.corrcoef(np.abs(pred),r*r)[0,1]),'groups':groups,'rank':int(matrix_rank(X)),'columns':int(X.shape[1]),'condition':float(sv[0]/max(sv[-1],1e-12))})
  train=d.geo_id.isin([g for g in geos if g not in sparse]).to_numpy(); test=~train; m=Ridge(alpha=1e-6).fit(X[train],y[train]); pred=m.predict(X[test]); r=y[test]-pred
  out.append({'candidate':cand,'split':'geo_unseen_diagnostic_zero_geo_effect','train_rows':int(train.sum()),'test_rows':int(test.sum()),'log_rmse':float(np.sqrt(np.mean(r*r))),'level_rmse':float(np.sqrt(np.mean((np.exp(y[test])-np.exp(pred))**2))),'beta_sign_recovery':float(np.mean(np.sign(m.coef_[:len(s.channels)])==np.sign([np.mean([s.true_beta_gc[g][c] for g in geos]) for c in s.channels]))),'limitation':'fixed geo effects cannot predict unseen geos; zero-effect fallback is diagnostic only'})
 return {'world_id':wid,'seed':int(s.panel_seed),'n_geos':len(geos),'n_weeks':s.n_weeks,'sparse_geos':geos[int(.7*len(geos)):],'results':out}
print(json.dumps({'source_worlds':WORLD_IDS,'splits':SPLITS,'results':[world(w) for w in WORLD_IDS]},sort_keys=True))
```

**Unresolved execution-blocking design questions:** none.
