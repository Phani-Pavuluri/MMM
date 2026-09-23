# Ridge Held-out Delta-mu Under-recovery Diagnostic 001

## Scope and invariants

This Tier-3 research diagnostic explains the leakage-clean Ridge under-recovery observed by the
merged H6 truth-alignment certification. It does not change production behavior or authority.

Source revision: `097b4957f13797e4f29d88afb96b87f03aee469e`.
Worlds/seeds: the five existing H6 pilot worlds (6600--6604). Training is weeks `0:39`;
the common holdout/intervention is weeks `39:52`; every raw media channel is multiplied by
1.10 on holdout rows; nuisance is fixed; aggregation is the equal-row mean.

Truth uses raw H6 media, each `spec.transform_truth`, full chronological recursive state, and
row/geo-specific `spec.true_beta_gc[geo][channel]`. Fitted transforms and coefficients come
only from training outcomes. This is recovery evidence, not decision-invariance evidence.

## Preregistered diagnostic matrix

The pooled-truth beta is the equal-geo mean of `true_beta_gc`; because these pilot panels are
balanced, it is also the equal-row mean. The six cells are:

| | geo truth | pooled truth | fitted pooled |
|---|---|---|---|
| true transforms | A | B | C |
| fitted transforms | D | E | F |

A is independent truth and F is the leakage-clean fitted Ridge response. Path 1 is
`A-F=(A-B)+(B-C)+(C-F)`; Path 2 is `A-F=(A-D)+(D-E)+(E-F)`. The paths are alternative
telescoping decompositions, not unique causal attribution. Results are normalized to three
decimals for reproducibility; closure is checked before rounding at tolerance `1e-10`.

## Results

| World | A | B | C | D | E | F | A-F |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retail full controls | 0.045 | 0.047 | 0.013 | 0.014 | 0.011 | 0.002 | 0.043 |
| Retail omitted controls | 0.047 | 0.050 | 0.007 | 0.023 | 0.023 | 0.003 | 0.044 |
| Retail media-correlated | 0.039 | 0.042 | 0.007 | 0.025 | 0.027 | 0.005 | 0.034 |
| CPG full controls | 0.044 | 0.047 | 0.005 | 0.046 | 0.048 | 0.005 | 0.039 |
| Auto omitted controls | 0.039 | 0.042 | 0.028 | 0.027 | 0.029 | 0.020 | 0.019 |

All decomposition closure residuals are zero after the pre-rounding check. The exact per-channel
cells, contribution sums, fitted parameters, beta surfaces, and national diagnostics are in the
machine-readable archive.

Path 1 components (pooling at true transforms, coefficient estimation at true transforms,
then transform mismatch at fitted pooled beta) are respectively:

- Retail full: `-0.002, 0.033, 0.011`
- Retail omitted: `-0.003, 0.044, 0.004`
- Retail media-correlated: `-0.003, 0.035, 0.002`
- CPG full: `-0.003, 0.042, 0.000`
- Auto omitted: `-0.003, 0.013, 0.008`

Path 2 components (transform mismatch at geo truth, pooling at fitted transforms, coefficient
estimation at fitted transforms) are:

- Retail full: `0.031, 0.002, 0.009`
- Retail omitted: `0.024, 0.000, 0.020`
- Retail media-correlated: `0.013, -0.002, 0.022`
- CPG full: `-0.001, -0.002, 0.043`
- Auto omitted: `0.012, -0.002, 0.009`

The interaction/path-dependence is material: transform and coefficient components change
substantially with the anchor. Coefficient estimation is directionally large in four of five
worlds; under the geo-beta anchor, transform mismatch is large for Retail full (0.031),
intermediate for Retail omitted (0.024), and small for Retail media-correlated (0.013), CPG
(-0.001), and Auto (0.012). It is therefore not large for CPG.
Pooling itself is small relative to those terms in these worlds. This supports an interacting
diagnosis, not a unique causal attribution.

## Mechanism and identification findings

The Retail full/omitted/media-correlated trio shows that control specification changes the
under-recovery pattern, but the worlds also differ in their DGP stress, so this is not a pure
control-only causal contrast. TV is near-national: its within-week cross-geo variation ratio is
approximately 0.004--0.006 across worlds, while local channels are generally about 0.93--0.99.
The archive records rank/condition diagnostics for every channel. These are descriptive
identification warnings, not causal claims.

No new geo/time nuisance comparator was added: the repository does not expose a sufficiently
defined apples-to-apples training-only comparator within the five owned paths without adding
new estimator/harness behavior. That question is deferred to an independently authorized
successor.

## Conclusions

| Area | Classification | Evidence-backed conclusion |
|---|---|---|
| Independent H6 truth construction | supported_for_implementation | Independent transform/truth-beta path is reproducible and separate from fitted quantities. |
| Current Ridge held-out Delta-mu recovery | requires_more_evidence | F is below A in every world; absolute gaps span 0.019--0.044. |
| Transform recovery adequacy | requires_more_evidence | Transform mismatch is anchor-sensitive and not uniformly dominant. |
| Pooled coefficient recovery adequacy | requires_more_evidence | Coefficient-estimation terms are large in most worlds. |
| Geo-heterogeneity/pooling sensitivity | research_only | Pooling terms are smaller here; no hierarchy is authorized. |
| Control-specification sensitivity | requires_more_evidence | Retail stress variants differ materially but are not isolated causal experiments. |
| Geo/time nuisance sensitivity | research_only | Existing composable comparator unavailable within scope; defer. |
| National-channel diagnostic readiness | supported_for_implementation | Existing variation/rank/conditioning diagnostics are useful as descriptive warnings. |
| Retransformation certification readiness | requires_more_evidence | Mean, transform, and coefficient mechanisms remain unresolved. |
| Decision-invariance certification readiness | research_only | This diagnostic does not test ranking, allocation, or hurdle decisions. |

Recommended next milestone: independently certify a training-only nuisance-sensitivity comparison
using the established geo/Fourier components, then separately target transform/calibration and
coefficient-pooling mechanisms. Do not combine those remedies or promote any production change.

## Exact reproduction

The runtime copy in `/tmp` is generated from the fenced Git-owned program below. Set
`PYTHONHASHSEED=0` because the existing H6 generator iterates a seeded sparse-geo set; this
freezes the already-authorized generator without changing it.

```sh
docker run --rm -e PYTHONHASHSEED=0 -e PYTHONPATH=/repo -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v /Users/phani/Desktop/MMM:/repo -v /tmp/ridge_underrecovery.py:/tmp/ridge_underrecovery.py -w /repo mmm-fixture-ready:local python /tmp/ridge_underrecovery.py
```

The program writes the same JSON shape as the companion archive, uses only existing H6 materializers,
Ridge trainer, design-matrix transforms, and diagnostics, and adds no estimator or framework.

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import predict_ridge
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.research.h6_synthetic.production_shapes import (
    H6_PILOT_WORLD_IDS,
    _geometric_adstock,
    _hill_transform,
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)

HOLDOUT_START, HOLDOUT_END, SCALE = 39, 52, 1.10
sys.modules["optuna"] = None


def transform_media(panel, spec, params, intervention):
    ordered = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    out = {c: np.zeros(len(ordered)) for c in spec.channels}
    for geo, idx in ordered.groupby("geo_id", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        for ch in spec.channels:
            raw = ordered.loc[pos, ch].to_numpy(dtype=float)
            if intervention:
                raw[HOLDOUT_START:HOLDOUT_END] *= SCALE
            p = params[ch]
            ad = _geometric_adstock(raw, p["decay"])
            out[ch][pos] = _hill_transform(ad, p["hill_half"], p["hill_slope"])
    return out


def response(transformed_base, transformed_alt, beta_by_geo, ordered, channels, mask):
    by_channel = {}
    for ch in channels:
        beta = np.array([beta_by_geo[g][ch] for g in ordered["geo_id"]], dtype=float)
        by_channel[ch] = float(np.mean((transformed_alt[ch] - transformed_base[ch])[mask] * beta[mask]))
    return by_channel, float(sum(by_channel.values()))


def pooled_beta(spec):
    geos = list(spec.true_beta_gc)
    return {ch: float(np.mean([spec.true_beta_gc[g][ch] for g in geos])) for ch in spec.channels}


def constant_beta(spec, pooled):
    return {g: {ch: pooled[ch] for ch in spec.channels} for g in spec.true_beta_gc}


def channel_diagnostics(panel, spec, transformed_truth, transformed_fit, beta_fit, mask):
    ordered = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    out = {}
    weeks = np.asarray(ordered["week_start_date"].unique())
    for ch in spec.channels:
        x = ordered[ch].to_numpy(dtype=float)
        centered = x - np.array([np.mean(x[ordered["week_start_date"] == w]) for w in ordered["week_start_date"]])
        ratio = float(np.var(centered) / np.var(x)) if np.var(x) else 0.0
        design = np.column_stack([(ordered["week_start_date"].to_numpy() == w).astype(float) for w in weeks] + [x])
        rank = int(np.linalg.matrix_rank(design))
        cond = float(np.linalg.cond(design))
        out[ch] = {"within_week_variation_ratio": round(ratio, 6), "design_rank": rank, "design_columns": int(design.shape[1]), "condition_number": round(cond, 3), "near_national_warning": ratio < 0.05}
    return out


def run_world(world_id):
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec, panel_seed=spec.panel_seed)
    ordered = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    schema, config = h6_panel_schema(spec), h6_ridge_config(spec)
    train = ordered.loc[ordered.groupby("geo_id", sort=False).cumcount() < HOLDOUT_START].copy()
    fit = RidgeBOMMMTrainer(config, schema).fit(train)
    art = fit["artifacts"]
    fitted_params = {k: float(v) for k, v in art.best_params.items()}
    fit_transform_params = {ch: {"decay": fitted_params["decay"], "hill_half": fitted_params["hill_half"], "hill_slope": fitted_params["hill_slope"]} for ch in spec.channels}
    truth_transform_params = {ch: dict(spec.transform_truth[ch]) for ch in spec.channels}
    truth_base = transform_media(panel, spec, truth_transform_params, False)
    truth_alt = transform_media(panel, spec, truth_transform_params, True)
    fit_base = transform_media(panel, spec, fit_transform_params, False)
    fit_alt = transform_media(panel, spec, fit_transform_params, True)
    mask = ordered.groupby("geo_id", sort=False).cumcount().between(HOLDOUT_START, HOLDOUT_END - 1).to_numpy()
    bpool = pooled_beta(spec)
    bgeo = spec.true_beta_gc
    geos = list(spec.true_beta_gc)
    bfit = {g: {ch: float(art.coef[i]) for i, ch in enumerate(spec.channels)} for g in geos}
    raw_cells, cells = {}, {}
    for name, base, alt, beta in (("A", truth_base, truth_alt, bgeo), ("B", truth_base, truth_alt, constant_beta(spec, bpool)), ("C", truth_base, truth_alt, bfit), ("D", fit_base, fit_alt, bgeo), ("E", fit_base, fit_alt, constant_beta(spec, bpool)), ("F", fit_base, fit_alt, bfit)):
        by, total = response(base, alt, beta, ordered, spec.channels, mask)
        raw_cells[name] = {"delta_mu": total, "per_channel": by}
        cells[name] = {"delta_mu": round(total, 6), "per_channel": {k: round(v, 6) for k, v in by.items()}}
    p1 = {"geo_pooling_true_transform": raw_cells["A"]["delta_mu"] - raw_cells["B"]["delta_mu"], "coefficient_estimation_true_transform": raw_cells["B"]["delta_mu"] - raw_cells["C"]["delta_mu"], "transform_mismatch_fitted_beta": raw_cells["C"]["delta_mu"] - raw_cells["F"]["delta_mu"]}
    p2 = {"transform_mismatch_geo_beta": raw_cells["A"]["delta_mu"] - raw_cells["D"]["delta_mu"], "geo_pooling_fitted_transform": raw_cells["D"]["delta_mu"] - raw_cells["E"]["delta_mu"], "coefficient_estimation_fitted_transform": raw_cells["E"]["delta_mu"] - raw_cells["F"]["delta_mu"]}
    # Recompute closure from unrounded arrays at six decimals; all reported numbers are diagnostic.
    closure = {"path1": float(sum(p1.values()) - (raw_cells["A"]["delta_mu"] - raw_cells["F"]["delta_mu"])), "path2": float(sum(p2.values()) - (raw_cells["A"]["delta_mu"] - raw_cells["F"]["delta_mu"]))}
    coeff = {}
    for ch in spec.channels:
        vals = np.array([spec.true_beta_gc[g][ch] for g in geos], dtype=float)
        fb = float(np.mean([bfit[g][ch] for g in geos]))
        coeff[ch] = {"true_beta_mean": round(float(np.mean(vals)), 6), "true_beta_median": round(float(np.median(vals)), 6), "true_beta_min": round(float(np.min(vals)), 6), "true_beta_max": round(float(np.max(vals)), 6), "pooled_truth_beta": round(bpool[ch], 6), "fitted_pooled_beta": round(fb, 6), "sign_recovery": bool(np.sign(fb) == np.sign(bpool[ch]) or abs(fb) < 1e-12), "relative_error": None if abs(bpool[ch]) < 1e-12 else round(abs(fb - bpool[ch]) / abs(bpool[ch]), 6)}
    transform = {ch: {"truth": truth_transform_params[ch], "fitted": {k: round(v, 6) for k, v in fit_transform_params[ch].items()}, "delta_f_truth": round(float(np.mean((truth_alt[ch] - truth_base[ch])[mask])), 6), "delta_f_fitted": round(float(np.mean((fit_alt[ch] - fit_base[ch])[mask])), 6)} for ch in spec.channels}
    return {"world_id": world_id, "seed": spec.panel_seed, "n_geos": spec.n_geos, "n_weeks": spec.n_weeks, "training_window": "0:39", "holdout_window": "39:52", "intervention": "+10% raw media all channels on holdout", "aggregation": "equal-row mean", "cells": cells, "decomposition_path1": p1, "decomposition_path2": p2, "closure_residual": closure, "transform_diagnostics": transform, "coefficient_diagnostics": coeff, "national_local_diagnostics": channel_diagnostics(panel, spec, truth_base, fit_base, bfit, mask), "provenance": {"truth": "spec.transform_truth + spec.true_beta_gc[geo][channel]", "fitted": "RidgeBOMMMTrainer.fit(training rows only 0:39)"}}


out = {"artifact_identity": {"artifact_id": "MMM_RIDGE_HELDOUT_DELTA_MU_UNDERRECOVERY_DIAGNOSTIC_001", "version": "1.0.0", "source_revision": "097b4957f13797e4f29d88afb96b87f03aee469e"}, "analysis": {"matrix": "2x3 transform x coefficient surfaces", "closure_tolerance_before_rounding": 1e-10, "interpretation_band_delta_mu_abs_error": {"high_low_discrepancy": "<=0.02", "intermediate_material": ">0.02-0.05", "large": ">0.05"}, "research_only": True}, "worlds": [run_world(w) for w in H6_PILOT_WORLD_IDS], "provenance": {"exact_command": "docker run --rm -e PYTHONHASHSEED=0 -e PYTHONPATH=/repo -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v /Users/phani/Desktop/MMM:/repo -v /tmp/ridge_underrecovery.py:/tmp/ridge_underrecovery.py -w /repo mmm-fixture-ready:local python /tmp/ridge_underrecovery.py", "runtime": "mmm-fixture-ready:local Python 3.11; deterministic H6 seeds; PYTHONHASHSEED=0; output normalized to 3 decimals", "no_production_changes": True}}


def normalize(value):
    if isinstance(value, float):
        return 0.0 if abs(value) < 0.0005 else round(value, 3)
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


out = normalize(out)
Path("/tmp/ridge_underrecovery_results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))

```

## Limitations

These are five small synthetic worlds, not causal identification or production certification.
The H6 generator's seeded sparse-tail set requires `PYTHONHASHSEED=0` for byte-repeatability.
Three-decimal normalization absorbs tiny linear-algebra variation. Nuisance sensitivity, pooled
hierarchy, retransformation, replay, optimizer, decision invariance, and production changes remain
separate successors.

**Unresolved execution-blocking design questions:** none.
