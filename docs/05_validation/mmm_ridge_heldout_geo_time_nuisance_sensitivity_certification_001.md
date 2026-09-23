# Ridge held-out geo/time nuisance sensitivity certification 001

**Task:** `MMM_RIDGE_HELDOUT_GEO_TIME_NUISANCE_SENSITIVITY_CERTIFICATION_001`
**Source revision:** `a0053e9331797ce8745ba7a82b1825d116e6f382`
**Status:** research-only; no production numerical behavior or authority changed.

## Result

The three bounded, leakage-clean candidates were evaluated on all five deterministic H6 pilot
worlds. Adding known-geo effects and the existing bounded Fourier/trend basis substantially lowers
held-out log error and serial/geo residual structure, but does **not** recover the independent H6
media response: Delta-mu remains materially under-recovered in four of five worlds and pooled-beta
relative error remains large. Therefore omitted geo/time nuisance is a real residual-structure
problem, but it is not the sole explanation of the predecessor's under-recovery. Transform,
pooled-coefficient, omitted-control, and identification failures remain.

The evidence concerns future weeks for geographies observed in training. Fixed geo effects do not
predict an unseen geo; no fallback was invented. The Fourier/trend candidate is a research
comparator, not a production recommendation.

## Preregistered matrix and boundary

| Candidate | Specification |
|---|---|
| A | Current global-intercept Ridge + pooled media + supplied controls |
| B | Known-geo one-hot effects + pooled media + supplied controls |
| C | B + existing Fourier (one annual harmonic) and bounded polynomial trend basis |

Worlds/seeds are the five existing H6 pilot worlds, 6600–6604. Every outcome-dependent transform,
hyperparameter, coefficient, and nuisance fit uses weeks `0:39`; evaluation is weeks `39:52`.
The intervention is +10% to every raw media channel only in weeks 39:52. Design matrices are built
over the complete geo-time path, preserving legitimate recursive pre-holdout state. Baseline and
plan hold nuisance columns fixed and use equal-row means. Truth is independently reconstructed
from raw media, each channel's `transform_truth`, full-path adstock/Hill recursion, and
row/geo-specific `true_beta_gc`.

The predecessor research interpretation band is retained: absolute Delta-mu error `<=0.02`
high/low discrepancy, `>0.02–0.05` intermediate/material, `>0.05` large. Candidate comparison,
residual, conditioning, and coefficient changes are descriptive; no release threshold is invented.

## Held-out results

| World | Truth | A Delta-mu (abs err) | B Delta-mu (abs err) | C Delta-mu (abs err) |
|---|---:|---:|---:|---:|
| Retail full controls | 0.044594 | 0.002481 (0.042113) | 0.001415 (0.043178) | 0.001513 (0.043081) |
| Retail omitted controls | 0.047182 | 0.003013 (0.044169) | 0.002763 (0.044419) | 0.002573 (0.044609) |
| Retail media-correlated controls | 0.038710 | 0.004935 (0.033774) | 0.004649 (0.034060) | 0.004520 (0.034189) |
| CPG full controls | 0.044084 | 0.005379 (0.038705) | 0.005342 (0.038742) | 0.004652 (0.039432) |
| Auto omitted controls | 0.039120 | 0.020342 (0.018778) | 0.016691 (0.022430) | 0.015809 (0.023312) |

Across the five worlds, known-geo nuisance correction improves held-out log RMSE (A ranges
0.424–0.525; B 0.205–0.439; C 0.203–0.435) and lowers lag-1 residual correlation (A
0.817–0.872; B 0.322–0.830; C 0.324–0.828). Residual geo-pattern ratios are approximately
one for A and approximately zero for B/C; time-pattern ratios remain nonzero, especially in
omitted-control worlds. These are fit/generalization diagnostics, not causal identification.

Pooled-beta relative error remains high: A `0.496–0.918`, B `0.521–0.919`, C `0.540–0.930`.
Thus nuisance correction does not cure coefficient/transform under-recovery. Contribution deltas
are computed per channel from the same holdout rows; their sums equal the candidate Delta-mu within
floating-point tolerance. The archive contains the per-world candidate rows and provenance.

## Identification and limitations

The existing H6 national/local diagnostic remains in force: TV is near-national while search,
social, radio, and local-flyer have materially greater within-week cross-geo variation. Rank and
conditioning are reported by the executable for each candidate design; structured-time
estimability is not causal identification. B/C are valid only for known geographies. Sparse/short
history, retransformation, decision invariance, replay, optimizer, and production mean-structure
implementation remain separate successors.

## Conclusions

| Area | Classification | Conclusion |
|---|---|---|
| Geo effects for known-geo future prediction | `requires_more_evidence` | Improves residual/generalization diagnostics; not production-authorized. |
| Fourier/trend nuisance | `requires_more_evidence` | Improves fit modestly, but does not restore response recovery. |
| Omitted nuisance as under-recovery driver | `rejected` as sole driver | Nuisance helps residuals while Delta-mu/beta errors persist. |
| Residual transform/coefficient failure | `supported_for_implementation` as diagnostic finding | Continue transform/coefficient/identification diagnosis; no fix here. |
| National-channel diagnostic dependency | `research_only` | Keep variation/rank/conditioning disclosure; no policy change. |
| Sparse/short-history readiness | `requires_more_evidence` | This task is not a short-history certification. |
| Retransformation readiness | `requires_more_evidence` | Residual suitability improved but omitted-control contamination remains. |
| Decision-invariance readiness | `research_only` | Ranking/allocation/hurdle testing is independent work. |

## Exact reproducibility

The exact deterministic program used for the numbers is preserved as the fenced program below;
it composes only existing H6 materialization, `RidgeBOMMMTrainer`, `build_design_matrix`, the
existing Fourier/trend feature builder, and repository `fit_ridge`. It emits normalized JSON for
the five worlds. Save the block as `/tmp/nuisance_sensitivity_run.py` and run it twice; the two
outputs were byte-identical under `PYTHONHASHSEED=0`.

```bash
docker run --rm -e PYTHONHASHSEED=0 -e PYTHONPATH=/repo \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v /Users/phani/Desktop/MMM:/repo \
  -v /tmp/nuisance_sensitivity_run.py:/tmp/nuisance_sensitivity_run.py \
  -w /repo mmm-fixture-ready:local python /tmp/nuisance_sensitivity_run.py
```

```python
+import sys
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
        hold_d = d.loc[hold_mask].copy(); hold_d["resid"] = residual; hold_d["actual"] = actual_log[hold_mask]
        geo_means = hold_d.groupby(schema.geo_column)["resid"].mean(); time_means=hold_d.groupby(schema.week_column)["resid"].mean(); actual_geo_means=hold_d.groupby(schema.geo_column)["actual"].mean(); actual_time_means=hold_d.groupby(schema.week_column)["actual"].mean()
        vals=np.asarray(XX); sv=svd(vals[train_mask],compute_uv=False)
        betas=np.asarray(c[:media_n]); tb=np.array([np.mean([spec.true_beta_gc[g][ch] for g in geos]) for ch in schema.channel_columns])
        out["candidates"][name] = {"delta_mu":delta,"abs_delta_error":abs(delta-true_delta),"channel_contribution_delta":contrib,"true_channel_contribution_delta":true_contrib,"channel_abs_error":{ch:abs(contrib[ch]-true_contrib[ch]) for ch in contrib},"beta":betas.tolist(),"pooled_true_beta":tb.tolist(),"beta_rel_error":float(np.mean(np.abs((betas-tb)/(np.abs(tb)+1e-9)))),"beta_sign_recovery":float(np.mean(np.sign(betas)==np.sign(tb))),"holdout_log_rmse":float(np.sqrt(np.mean(residual**2))),"residual_geo_pattern_ratio":float(np.var(geo_means)/(np.var(actual_geo_means)+1e-9)),"residual_time_pattern_ratio":float(np.var(time_means)/(np.var(actual_time_means)+1e-9)),"residual_lag1":float(np.corrcoef(residual[:-1],residual[1:])[0,1]),"heteroskedasticity_abs_pred_resid2":float(np.corrcoef(np.abs(pred0[hold_mask]),residual**2)[0,1]),"rank_train":int(matrix_rank(vals[train_mask])),"columns":int(vals.shape[1]),"condition_train":float(sv[0]/max(sv[-1],1e-12)),"geo_effect_columns":max(0,vals.shape[1]-X.shape[1]),"national_diagnostics":{ch:{"within_week_variation_ratio":float(np.var(d.loc[hold_mask,ch].to_numpy(float)-d.loc[hold_mask].groupby(schema.week_column)[ch].transform("mean").to_numpy(float))/(np.var(d.loc[hold_mask,ch].to_numpy(float))+1e-12))} for ch in schema.channel_columns}}
    return canon(out)

rows=[fit_one(w) for w in H6_PILOT_WORLD_IDS]
print(json.dumps(rows, sort_keys=True, separators=(",",":")))

```

Archive: [`MMM_RIDGE_HELDOUT_GEO_TIME_NUISANCE_SENSITIVITY_CERTIFICATION_001.json`](archives/MMM_RIDGE_HELDOUT_GEO_TIME_NUISANCE_SENSITIVITY_CERTIFICATION_001.json).

## Successors

1. Independently certify transform-response recovery and pooled-vs-heterogeneous coefficients.
2. Certify a leakage-clean retransformation method only after nuisance specification is resolved.
3. Separately certify decision invariance (ranking, allocation, and hurdle decisions).
4. Design, review, and authorize any production geo/time implementation independently.

No production behavior, Delta-mu authority, optimizer/economics, replay, MIP, GeoX, or DR-04
threshold changed.
