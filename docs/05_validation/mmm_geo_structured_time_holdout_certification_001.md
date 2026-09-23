# MMM Geo-Structured-Time Holdout Certification 001 — Correction

**Task:** `MMM_GEO_STRUCTURED_TIME_HOLDOUT_CERTIFICATION_001`  
**Rejected head:** `128da874a8b8db2273f3d3a7c54c27b3391176b5`
**Source revision:** `bc4473c3ed4c3faca2ce377b21f30f1c100e26d5`  
**Status:** research-only; no production numerical behavior changed

## Correction summary

The prior evidence was rejected because hyperparameters were tuned on the full panel, the short
history labels changed the common test window, held-out Delta-mu was absent, and media/national
diagnostics were incomplete. This correction fixes all four defects. Every split now runs
`RidgeBOMMMTrainer.fit` on its training rows only; the common future test is always weeks 39–51;
sparse-tail history is masked to 26 or 13 weeks while other geos retain weeks 0–38; and each
eligible candidate/world reports held-out Delta-mu, media contribution, and national diagnostics.

## 1. Preregistered leakage boundary and splits

Worlds and seeds are unchanged: five existing H6 worlds (`6600`–`6604`), 20 geos, 52 weeks, and
seven channels. Candidates remain global, geo, and existing Fourier/trend; saturated weeks and
alternate bases are not competitive candidates.

| Split | All-geos future test | Observed-geos training | Sparse-tail training |
|---|---|---|---|
| `future_13` | weeks `39:52` | weeks `0:39` | weeks `0:39` |
| `short_26` | weeks `39:52` | weeks `0:39` | weeks `0:26` |
| `short_13` | weeks `39:52` | weeks `0:39` | weeks `0:13` |

The sparse tail is the existing H6 final six geos (`DMA_014`–`DMA_019`). For every row above,
hyperparameters (`decay`, `hill_half`, `hill_slope`, and `log_alpha`) are selected by the existing
trainer using only the split's training dataframe. The resulting parameters are then used to build
train/test design matrices. Held-out outcomes never enter tuning, nuisance fitting, model fitting,
or candidate selection. Media values and pre-holdout recursive state may cross the boundary because
they are known inputs; no future outcomes or future-derived parameters cross it. Calendar Fourier
features are constructed from week indices only. No random row split or holdout tuning is used.

## 2. Leakage-clean results

Ranges are across all five worlds. Plain `exp` is used for level error; no smearing or sigma
correction is introduced. Recovery and stability, not RMSE alone, determine interpretation.

| Candidate / split | Log RMSE | Level RMSE | Beta relative error | Sign recovery | Held-out Delta-mu abs. error |
|---|---:|---:|---:|---:|---:|
| Global / future-13 | `0.421–0.516` | `27.1k–45.1k` | `112%–3,110%` | `0.571–0.857` | `0.0025–0.0216` |
| Geo / future-13 | `0.152–0.168` | `11.1k–13.0k` | `38%–2,954%` | `0.857–1.000` | `0.0009–0.0127` |
| Fourier / future-13 | `0.143–0.234` | `11.8k–16.9k` | `56%–2,443%` | `0.857–1.000` | `0.0017–0.0179` |
| Global / short-26 | `0.421–0.518` | `27.1k–45.1k` | `115%–3,660%` | `0.714–0.857` | `0.0031–0.0220` |
| Geo / short-26 | `0.155–0.172` | `11.2k–13.1k` | `39%–3,461%` | `0.714–1.000` | `0.0008–0.0136` |
| Fourier / short-26 | `0.143–0.246` | `11.8k–18.1k` | `56%–2,863%` | `0.714–1.000` | `0.0018–0.0206` |
| Global / short-13 | `0.420–0.516` | `27.0k–45.2k` | `100%–3,684%` | `0.714` | `0.0034–0.0164` |
| Geo / short-13 | `0.153–0.169` | `11.3k–13.0k` | `42%–3,441%` | `0.857–1.000` | `0.0005–0.0145` |
| Fourier / short-13 | `0.145–0.264` | `11.8k–18.9k` | `69%–2,746%` | `0.857–1.000` | `0.0013–0.0222` |

The corrected future prediction advantage remains, but pooled media recovery remains materially
unstable in some worlds. Delta-mu recovery is generally within the preregistered high/intermediate
bands for the well-specified worlds, but the upper ranges cross the material boundary. This is
recovery evidence only and does not establish ranking, allocation, or hurdle invariance.

## 3. Held-out Delta-mu and media/contribution evidence

For each held-out candidate/world, the intervention multiplies every media column by `1.10` on
weeks 39–51. The fitted Delta-mu is the equal-row mean of candidate minus baseline fitted log means;
the known-truth Delta-mu uses H6 true pooled geo/channel effects applied to the transformed-media
delta. Nuisance columns and fitted coefficients are held fixed between baseline and intervention.
Per-channel fitted and true contribution means plus absolute contribution error are retained in the
JSON archive. No optimizer or canonical Delta-mu authority was changed.

## 4. Geo holdout and national-channel diagnostics

The 14-geos-train/six-geos-test diagnostic produces log-RMSE ranges of `0.756–13.188` (global),
`1.444–9.859` (geo), and `1.449–13.114` (Fourier). Fixed geo effects have no identified effect
for unseen geos; this diagnostic uses zero geo effects only to expose the limitation and is not a
production fallback. Known-geo future prediction and unseen-geo behavior remain separate claims.

For relevant future holdout designs, TV within-week variation is `0.0044–0.0090`; holdout matrix
ranks are `9–12` (global), `28–31` (geo), and `31–34` (Fourier), with condition ranges `68–414`,
`88–506`, and `452–8,398`, respectively. These are estimability diagnostics, not causal
identification. No national-channel policy was implemented.

## 5. Conclusions and successors

| Area | Classification | Conclusion |
|---|---|---|
| Known-geo geo baseline | `requires_more_evidence` | Predictive advantage persists, but media recovery is not uniformly stable. |
| Unseen-geo behavior | `rejected` | Fixed geo effects require a separately certified fallback. |
| Structured-time family | `requires_more_evidence` | Fourier/trend remains promising, but no production promotion follows. |
| Short-history readiness | `requires_more_evidence` | Fixed-window masking is now correct; no minimum-history policy is authorized. |
| National-channel compatibility | `supported_for_implementation` for diagnostics only | Preserve variation/rank/conditioning diagnostics. |
| Retransformation readiness | `requires_more_evidence` | Mean-structure recovery must stabilize first. |
| Full-panel Delta-mu recovery | `requires_more_evidence` | Recovery is measured; decision invariance remains independent. |

No production candidate, DR-04 threshold, Bayesian authority, replay, retransformation, partial
pooling, optimizer, or economic behavior changed.

## 6. Exact reproduction

The machine-readable archive uses `mmm_research_evidence_v1`. Exact command:

```bash
docker run --rm -e PYTHONPATH=/repo \
  -v /tmp/holdout_correction_run.py:/tmp/holdout_correction_run.py \
  -v /Users/phani/Desktop/MMM:/repo -w /repo \
  mmm-fixture-ready:local python /tmp/holdout_correction_run.py
```

The executable is preserved verbatim below and is the only source for the correction numbers.

```python
import json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from numpy.linalg import matrix_rank, svd
from mmm.research.h6_synthetic.production_shapes import H6_PILOT_WORLD_IDS,get_h6_world,materialize_h6_panel,h6_ridge_config,h6_panel_schema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.features.design_matrix import build_design_matrix

WORLD_IDS=list(H6_PILOT_WORLD_IDS)
SPLITS={'future_13':39,'short_26':39,'short_13':39}
def basis(d,X0,k,n_weeks):
 z=[X0]
 if k!='global': z.append(pd.get_dummies(d.geo_id,drop_first=True,dtype=float).to_numpy())
 if k=='fourier':
  wi=np.asarray(d.week_start_date.dt.isocalendar().week-1,dtype=int); tt=np.arange(n_weeks); f=np.c_[np.sin(2*np.pi*tt/52),np.cos(2*np.pi*tt/52),tt/max(n_weeks-1,1)]; z.append(f[wi.clip(0,51)])
 return np.column_stack(z)
def run_world(wid):
 s=get_h6_world(wid); df=materialize_h6_panel(s); cfg=h6_ridge_config(s); sch=h6_panel_schema(s); weeks=sorted(df.week_start_date.unique()); geos=sorted(df.geo_id.unique()); sparse=set(geos[int(.7*len(geos)):]); out=[]
 for cand in ['global','geo','fourier']:
  for split,cut in SPLITS.items():
   test_weeks=set(weeks[39:]); train_weeks=set(weeks[:39]); train=df.week_start_date.isin(train_weeks).to_numpy(); test=df.week_start_date.isin(test_weeks).to_numpy()
   if split=='short_26': train=train & (~df.geo_id.isin(sparse).to_numpy() | df.week_start_date.isin(set(weeks[:26])).to_numpy())
   if split=='short_13': train=train & (~df.geo_id.isin(sparse).to_numpy() | df.week_start_date.isin(set(weeks[:13])).to_numpy())
   train_df=df.loc[train].copy(); fit=RidgeBOMMMTrainer(cfg,sch).fit(train_df); bp=fit['artifacts'].best_params
   base=build_design_matrix(df,sch,cfg,decay=float(bp['decay']),hill_half=float(bp['hill_half']),hill_slope=float(bp['hill_slope'])); X=basis(base.df_aligned,base.X,cand,s.n_weeks); m=Ridge(alpha=1e-6).fit(X[train],base.y_modeling[train]); pred=m.predict(X[test]); y=base.y_modeling; r=y[test]-pred
   truth_beta=np.array([np.mean([s.true_beta_gc[g][c] for g in geos]) for c in s.channels]); beta=m.coef_[:len(s.channels)]; berr=np.abs((beta-truth_beta)/(np.abs(truth_beta)+1e-9)); test_df=df.loc[test].copy(); int_df=test_df.copy()
   for c in s.channels: int_df[c]=int_df[c]*1.1
   int_base=build_design_matrix(int_df,sch,cfg,decay=float(bp['decay']),hill_half=float(bp['hill_half']),hill_slope=float(bp['hill_slope'])); Xint=basis(int_base.df_aligned,int_base.X,cand,s.n_weeks); fitted_delta=float(np.mean(Xint@m.coef_-X[test]@m.coef_)); true_delta=float(np.mean(np.sum((int_base.X[:,:len(s.channels)]-base.X[test,:len(s.channels)])*np.array([np.mean([s.true_beta_gc[g][c] for g in geos]) for c in s.channels]),axis=1)))
   contrib_true=[]; contrib_fit=[]
   for j,c in enumerate(s.channels): contrib_true.append(float(np.mean(base.X[test,j]*np.array([s.true_beta_gc[g][c] for g in test_df.geo_id])))); contrib_fit.append(float(np.mean(X[test,j]*beta[j])))
   tv=df.loc[test].groupby('week_start_date')['tv'].transform(lambda z:z-z.mean()); tv_ratio=float(np.var(tv)/(np.var(df.loc[test,'tv'])+1e-12)); sv=svd(X[test],compute_uv=False); geo_bias=pd.Series(r).groupby(test_df.geo_id).mean(); time_bias=pd.Series(r).groupby(test_df.week_start_date).mean(); groups=[]
   for gn,gs in [('observed',[g for g in geos if g not in sparse]),('sparse',list(sparse))]:
    mask=test_df.geo_id.isin(gs).to_numpy(); groups.append({'group':gn,'n':int(mask.sum()),'log_rmse':float(np.sqrt(np.mean(r[mask]**2))) if mask.any() else None,'bias':float(np.mean(r[mask])) if mask.any() else None})
   out.append({'candidate':cand,'split':split,'train_rows':int(train.sum()),'test_rows':int(test.sum()),'train_window':'weeks[0:39] except sparse masked tail','sparse_history_window':('0:39' if split=='future_13' else ('0:26' if split=='short_26' else '0:13')),'test_window':'weeks[39:52]','fitted_hyperparameters':bp,'log_rmse':float(np.sqrt(np.mean(r*r))),'level_rmse':float(np.sqrt(np.mean((np.exp(y[test])-np.exp(pred))**2))),'beta_rel_error_mean':float(np.mean(berr)),'beta_sign_recovery':float(np.mean(np.sign(beta)==np.sign(truth_beta))),'contribution_true':contrib_true,'contribution_fitted':contrib_fit,'contribution_abs_error_mean':float(np.mean(np.abs(np.array(contrib_fit)-np.array(contrib_true)))),'heldout_delta_mu_true':true_delta,'heldout_delta_mu_fitted':fitted_delta,'heldout_delta_mu_abs_error':abs(fitted_delta-true_delta),'aggregation':'equal-row mean over held-out geo-week rows','intervention':'all media columns multiplied by 1.10; nuisance columns and fitted coefficients held fixed','geo_baseline_recovery_error':float(np.std(geo_bias.to_numpy())),'time_pattern_rmse':float(np.sqrt(np.mean(time_bias.to_numpy()**2))),'lag1':float(np.corrcoef(r[:-1],r[1:])[0,1]),'heteroskedasticity_corr':float(np.corrcoef(np.abs(pred),r*r)[0,1]),'tv_within_week_variation_ratio':tv_ratio,'holdout_rank':int(matrix_rank(X[test])),'holdout_columns':int(X[test].shape[1]),'holdout_condition':float(sv[0]/max(sv[-1],1e-12)),'groups':groups})
  train=df.geo_id.isin([g for g in geos if g not in sparse]).to_numpy(); test=~train; fit=RidgeBOMMMTrainer(cfg,sch).fit(df.loc[train].copy()); bp=fit['artifacts'].best_params; base=build_design_matrix(df,sch,cfg,decay=float(bp['decay']),hill_half=float(bp['hill_half']),hill_slope=float(bp['hill_slope'])); X=basis(base.df_aligned,base.X,cand,s.n_weeks); m=Ridge(alpha=1e-6).fit(X[train],base.y_modeling[train]); r=base.y_modeling[test]-m.predict(X[test]); out.append({'candidate':cand,'split':'geo_unseen_diagnostic_zero_geo_effect','train_rows':int(train.sum()),'test_rows':int(test.sum()),'log_rmse':float(np.sqrt(np.mean(r*r))),'level_rmse':float(np.sqrt(np.mean((np.exp(base.y_modeling[test])-np.exp(m.predict(X[test])))**2))),'beta_sign_recovery':float(np.mean(np.sign(m.coef_[:len(s.channels)])==np.sign([np.mean([s.true_beta_gc[g][c] for g in geos]) for c in s.channels]))),'limitation':'fixed geo effects cannot predict unseen geos; zero-effect fallback is diagnostic only'})
 return {'world_id':wid,'seed':int(s.panel_seed),'n_geos':len(geos),'n_weeks':s.n_weeks,'sparse_geos':sorted(sparse),'results':out}
print(json.dumps({'source_worlds':WORLD_IDS,'splits':SPLITS,'results':[run_world(w) for w in WORLD_IDS]},sort_keys=True))
```

**Unresolved execution-blocking design questions:** none.
