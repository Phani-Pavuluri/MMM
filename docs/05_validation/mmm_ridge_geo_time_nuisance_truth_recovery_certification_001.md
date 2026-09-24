# Ridge geo/time nuisance truth-recovery certification 001

**Task:** `MMM_RIDGE_GEO_TIME_NUISANCE_TRUTH_RECOVERY_CERTIFICATION_001`
**Source revision:** `8ff164696fcbc1b65dcab5977d48d16428200a29`
**Status:** research-only; no production behavior or authority changed.

## Scope and predecessor

This independent successor addresses the blocked predecessor
`MMM_RIDGE_HELDOUT_GEO_TIME_NUISANCE_SENSITIVITY_CERTIFICATION_001` (blocked head
`d77900e23d6d27317727bd47f0c58254e9d14fff`, rejected review head
`1c98ac57e370f63ee9ea14c2a75abfe5b854fea1`). Its rejected analytical artifacts were not merged
or treated as certified evidence.

## Preregistered design

The five deterministic H6 worlds (seeds 6600--6604), A/B/C candidates, training-only 0:39 fit,
39:52 known-geo holdout, +10% raw all-media holdout intervention, full recursive media state,
fixed nuisance, independent `transform_truth`/`true_beta_gc` truth, and equal-row mean are
unchanged from the authorized contract. A is global intercept; B adds known-geo effects; C adds
the already-established bounded Fourier/trend basis. No saturated weeks, new estimator, time
basis, retransformation, partial pooling, replay, optimizer, MIP, or GeoX work was performed.

## Findings

The corrected serial metric is the arithmetic mean of lag-1 correlations calculated separately
within each geo on holdout residuals; cross-geo boundary pairs are excluded. The executable and
archive persist per-geo values.

Direct nuisance truth recovery is now measured rather than inferred from residual fit. Geo truth is
`true_alpha_g`, compared after centering because the intercept/effects decomposition is location
non-identifiable. B/C centered geo-pattern correlations range 0.65--0.76 with RMSE 0.31--0.36.
The C time component is compared with the exposed H6 smooth baseline
`0.06*sin(2*pi*week_index/52)`; correlations range 0.77--0.96 with RMSE 0.012--0.038.
These are nuisance-recovery results, distinct from residual-pattern reduction. Shocks and
omitted-control effects are not claimed to be recovered by the smooth time basis.

Held-out media-response recovery remains materially under-recovered in the same evidence: B/C
improve residual fit and serial/geo pattern diagnostics but do not restore pooled beta or Delta-mu
recovery. Thus nuisance correction is contributory to mean fit, not the sole explanation of
media-response failure. Plain-exp level metrics remain diagnostic only.

The archive's `generated_candidate_rows` is the direct deterministic program output and includes
log/level metrics, beta/sign/relative error, channel contributions and closure, Delta-mu, geo/time
residual patterns, within-geo serial values, heteroskedasticity, sparse/observed summaries,
national variation, rank, conditioning, and nuisance-truth recovery fields. Contribution sums
close within floating-point tolerance. Fixed geo effects do not establish unseen-geo prediction.

## Conclusions

| Area | Classification | Result |
|---|---|---|
| Corrected within-geo serial summaries | `supported_for_implementation` | Boundary-safe descriptive diagnostic |
| Geo nuisance truth recovery | `requires_more_evidence` | Directional but materially imperfect recovery |
| Fourier/trend time nuisance truth recovery | `requires_more_evidence` | Smooth baseline directionally recovered; not certified for production |
| Nuisance correction as sole media-response explanation | `rejected` | Beta/Delta-mu under-recovery persists |
| Production candidate/authority | `research_only` | No implementation or authority promotion |
| Retransformation/decision invariance | `research_only` | Separate successor milestones |

## Reproduction

The exact program below was run twice under `PYTHONHASHSEED=0`; outputs were byte-identical and
were compared exactly with the archive's generated rows.

```bash
docker run --rm -e PYTHONHASHSEED=0 -e PYTHONPATH=/repo \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v /Users/phani/Desktop/MMM:/repo \
  -v /tmp/nuisance_sensitivity_run.py:/tmp/nuisance_sensitivity_run.py \
  -w /repo mmm-fixture-ready:local python /tmp/nuisance_sensitivity_run.py
```

```python
import sys
sys.modules["optuna"] = None
import json
import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank, svd

from mmm.research.h6_synthetic.production_shapes import (
    H6_PILOT_WORLD_IDS, get_h6_world, materialize_h6_panel, h6_ridge_config,
    h6_panel_schema, _geometric_adstock, _hill_transform,
)
from mmm.features.design_matrix import build_design_matrix
from mmm.features.builder import build_extra_control_matrix
from mmm.config.extensions import FeatureEngineConfig
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge

TRAIN_END = 39
HOLDOUT_END = 52
def canon(x):
    if isinstance(x, dict): return {k: canon(x[k]) for k in sorted(x)}
    if isinstance(x, (list, tuple)): return [canon(v) for v in x]
    if isinstance(x, (float, np.floating)): return round(float(x), 6)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def fit_one(wid):
    spec = get_h6_world(wid); panel = materialize_h6_panel(spec)
    cfg, schema = h6_ridge_config(spec), h6_panel_schema(spec)
    ordered = panel.sort_values([schema.geo_column, schema.week_column]).reset_index(drop=True)
    train_mask = ordered.groupby(schema.geo_column, sort=False).cumcount().to_numpy() < TRAIN_END
    hold_mask = ~train_mask
    train = ordered.loc[train_mask].copy()
    # Hyperparameters are selected on training outcomes only.
    trfit = RidgeBOMMMTrainer(cfg, schema).fit(train)
    bp = trfit["artifacts"].best_params
    alpha = float(10 ** float(bp["log_alpha"]))
    base = build_design_matrix(ordered, schema, cfg, decay=float(bp["decay"]), hill_half=float(bp["hill_half"]), hill_slope=float(bp["hill_slope"]))
    X = base.X; d = base.df_aligned; y = base.y_modeling
    # Materialize the identical +10% intervention on held-out raw media only.
    plan = ordered.copy()
    plan.loc[hold_mask, list(schema.channel_columns)] *= 1.10
    plan_b = build_design_matrix(plan, schema, cfg, decay=float(bp["decay"]), hill_half=float(bp["hill_half"]), hill_slope=float(bp["hill_slope"]))
    # Fit the global production-shaped coefficient surface on training rows of the full-path design.
    coef_a, int_a = fit_ridge(X[train_mask], y[train_mask], alpha=alpha)

    geos = sorted(d[schema.geo_column].astype(str).unique())
    sparse = set(geos[int(0.7*len(geos)):])
    geo_codes = pd.Categorical(d[schema.geo_column].astype(str), categories=geos).codes
    G = np.eye(len(geos), dtype=float)[geo_codes][:, 1:]
    # Existing bounded Fourier/trend utilities: one annual harmonic plus the configured trend basis.
    fc = FeatureEngineConfig(trend_spline_knots=2, fourier_yearly_harmonics=1)
    T = build_extra_control_matrix(d, schema, fc)
    def fit_candidate(name):
        extras = [] if name == "A_global" else [G]
        if name == "C_geo_fourier_trend": extras.append(T)
        XX = np.column_stack([X, *extras]) if extras else X
        PP = np.column_stack([plan_b.X, *extras]) if extras else plan_b.X
        c, i = fit_ridge(XX[train_mask], y[train_mask], alpha=alpha)
        return XX, PP, c, i

    # Independent DGP truth, reconstructed from raw media and row/geo-specific beta.
    truth_base = np.zeros(len(d), dtype=float); truth_plan = np.zeros(len(d), dtype=float)
    true_contrib = {}; truth_tr = {}
    for ch in schema.channel_columns:
        tr = spec.transform_truth[ch]
        vals0=[]; vals1=[]
        for geo in geos:
            raw0 = d.loc[d[schema.geo_column].astype(str)==geo, ch].to_numpy(float)
            raw1 = raw0.copy(); raw1[TRAIN_END:] *= 1.10
            ad0 = _geometric_adstock(raw0, tr["decay"]); ad1 = _geometric_adstock(raw1, tr["decay"])
            f0 = _hill_transform(ad0, tr["hill_half"], tr["hill_slope"]); f1 = _hill_transform(ad1, tr["hill_half"], tr["hill_slope"])
            beta = float(spec.true_beta_gc[geo][ch]); vals0.extend(beta*f0); vals1.extend(beta*f1)
        truth_base += np.asarray(vals0); truth_plan += np.asarray(vals1)
        true_contrib[ch] = float(np.mean((np.asarray(vals1)-np.asarray(vals0))[hold_mask]))
        truth_tr[ch] = dict(tr)
    true_delta = float(np.mean((truth_plan-truth_base)[hold_mask]))
    # DGP mean components are only used for residual diagnostic; actual heldout outcome is never fit.
    actual_log = np.log(d[schema.target_column].to_numpy(float))
    out = {"world_id": wid, "seed": int(spec.panel_seed), "train_window":"0:39", "holdout_window":"39:52", "hyperparameters":canon(bp), "truth_transform":truth_tr, "true_delta_mu":true_delta, "candidates":{}}
    for name in ("A_global", "B_geo", "C_geo_fourier_trend"):
        XX, PP, c, i = fit_candidate(name)
        pred0 = predict_ridge(X if name=="A_global" else XX, c, i)
        pred1 = predict_ridge(plan_b.X if name=="A_global" else PP, c, i)
        # media columns precede controls; intervention deltas isolate channel contributions.
        media_n = len(schema.channel_columns)
        contrib = {}
        for j,ch in enumerate(schema.channel_columns):
            dx = plan_b.X[:,j]-X[:,j]
            contrib[ch] = float(np.mean((c[j]*dx)[hold_mask]))
        delta = float(np.mean((pred1-pred0)[hold_mask]))
        residual = actual_log[hold_mask]-pred0[hold_mask]
        # Serial structure is computed within each geo; cross-geo boundary pairs are excluded.
        lag_values=[]
        for _, ix in d.loc[hold_mask].groupby(schema.geo_column, sort=False).groups.items():
            loc=np.asarray(ix, dtype=int)
            rr=(actual_log-pred0)[loc]
            if len(rr)>1 and np.std(rr[:-1])>0 and np.std(rr[1:])>0: lag_values.append(float(np.corrcoef(rr[:-1],rr[1:])[0,1]))
        lag_mean=float(np.mean(lag_values)) if lag_values else 0.0
        level_pred=np.exp(pred0[hold_mask]); level_actual=np.exp(actual_log[hold_mask])
        hold_d = d.loc[hold_mask].copy(); hold_d["resid"] = residual; hold_d["actual"] = actual_log[hold_mask]
        geo_means = hold_d.groupby(schema.geo_column)["resid"].mean(); time_means=hold_d.groupby(schema.week_column)["resid"].mean(); actual_geo_means=hold_d.groupby(schema.geo_column)["actual"].mean(); actual_time_means=hold_d.groupby(schema.week_column)["actual"].mean()
        vals=np.asarray(XX); sv=svd(vals[train_mask],compute_uv=False)
        betas=np.asarray(c[:media_n]); tb=np.array([np.mean([spec.true_beta_gc[g][ch] for g in geos]) for ch in schema.channel_columns])
        # Direct nuisance-truth recovery: alpha_g is identified only up to a common intercept.
        geo_truth = np.array([float(spec.true_alpha_g[g]) for g in geos])
        geo_fit = None; geo_recovery = {"available": False}
        if name != "A_global":
            gcoef = np.asarray(c[X.shape[1]:X.shape[1]+len(geos)-1])
            geo_fit = np.r_[float(i[0]), float(i[0]) + gcoef]
            geo_fit = geo_fit - np.mean(geo_fit); gt = geo_truth - np.mean(geo_truth)
            geo_recovery = {"available": True, "centered_rmse": float(np.sqrt(np.mean((geo_fit-gt)**2))), "centered_correlation": float(np.corrcoef(geo_fit,gt)[0,1]), "truth_mean": float(np.mean(geo_truth)), "fit_mean_before_centering": float(np.mean(np.r_[float(i[0]), float(i[0])+gcoef]))}
        time_recovery = {"available": False}
        if name == "C_geo_fourier_trend":
            tc = np.asarray(c[X.shape[1]+len(geos)-1:]); fitted_time = T @ tc
            wk = d.groupby(schema.geo_column)[schema.week_column].rank(method="dense").to_numpy(dtype=float)-1.0
            true_time = 0.06*np.sin(2*np.pi*wk/52.0)
            fitted_time = fitted_time - np.mean(fitted_time); true_time = true_time - np.mean(true_time)
            time_recovery = {"available": True, "centered_rmse": float(np.sqrt(np.mean((fitted_time-true_time)**2))), "centered_correlation": float(np.corrcoef(fitted_time,true_time)[0,1]), "truth_definition":"0.06*sin(2*pi*week_index/52)"}
        sparse_hold = d.loc[hold_mask, schema.geo_column].astype(str).isin(sparse).to_numpy()
        observed_hold = ~sparse_hold
        sparse_stats = {"sparse_geos": sorted(sparse), "groups": {"sparse": {"n_rows": int(sparse_hold.sum()), "residual_bias": float(np.mean(residual[sparse_hold])), "residual_rmse": float(np.sqrt(np.mean(residual[sparse_hold]**2)))}, "observed": {"n_rows": int(observed_hold.sum()), "residual_bias": float(np.mean(residual[observed_hold])), "residual_rmse": float(np.sqrt(np.mean(residual[observed_hold]**2)))}}}
        national = {ch: {"within_week_variation_ratio": float(np.var(d.loc[hold_mask,ch].to_numpy(float)-d.loc[hold_mask].groupby(schema.week_column)[ch].transform("mean").to_numpy(float))/(np.var(d.loc[hold_mask,ch].to_numpy(float))+1e-12))} for ch in schema.channel_columns}
        out["candidates"][name] = {"delta_mu":delta,"abs_delta_error":abs(delta-true_delta),"channel_contribution_delta":contrib,"true_channel_contribution_delta":true_contrib,"channel_abs_error":{ch:abs(contrib[ch]-true_contrib[ch]) for ch in contrib},"contribution_sum_closure":float(sum(contrib.values())-delta),"beta":betas.tolist(),"pooled_true_beta":tb.tolist(),"beta_rel_error":float(np.mean(np.abs((betas-tb)/(np.abs(tb)+1e-9)))),"beta_sign_recovery":float(np.mean(np.sign(betas)==np.sign(tb))),"holdout_log_rmse":float(np.sqrt(np.mean(residual**2))),"holdout_level_rmse_plain_exp":float(np.sqrt(np.mean((level_pred-level_actual)**2))),"holdout_level_mean_bias_plain_exp":float(np.mean(level_pred-level_actual)),"residual_geo_pattern_ratio":float(np.var(geo_means)/(np.var(actual_geo_means)+1e-9)),"residual_time_pattern_ratio":float(np.var(time_means)/(np.var(actual_time_means)+1e-9)),"residual_lag1_within_geo_mean":lag_mean,"residual_lag1_within_geo_values":lag_values,"heteroskedasticity_abs_pred_resid2":float(np.corrcoef(np.abs(pred0[hold_mask]),residual**2)[0,1]),"rank_train":int(matrix_rank(vals[train_mask])),"columns":int(vals.shape[1]),"condition_train":float(sv[0]/max(sv[-1],1e-12)),"geo_effect_columns":max(0,vals.shape[1]-X.shape[1]),"geo_baseline_recovery":geo_recovery,"time_baseline_recovery":time_recovery,"sparse_observed_stability":sparse_stats,"national_diagnostics":national}
    return canon(out)

rows=[fit_one(w) for w in H6_PILOT_WORLD_IDS]
print(json.dumps(rows, sort_keys=True, separators=(",",":")))
```

Archive: [MMM_RIDGE_GEO_TIME_NUISANCE_TRUTH_RECOVERY_CERTIFICATION_001.json](archives/MMM_RIDGE_GEO_TIME_NUISANCE_TRUTH_RECOVERY_CERTIFICATION_001.json).

## Successors and limitations

The five-world pilot does not establish production thresholds, unseen-geo fallback, partial pooling,
or causal identification from conditioning. Any production geo/time implementation, retransformation
certification, replay, optimizer, or decision-invariance work requires an independent authorized
task.

No analytical artifact from the blocked predecessor was merged or copied into this evidence.
