# H6 Holdout Truth Alignment Certification 001

## Status and scope

This research-only certification reconstructs H6 generative truth independently of fitted
transform parameters and pooled Ridge coefficients. It does not change production behavior,
retransformation, replay, partial pooling, optimizer/economics, decision invariance, Delta-mu
authority, MIP, GeoX, or release thresholds.

Source revision for the empirical run: `e39bd2b236c81c925357746f2cb461ba45066da1`.
The exact blocked predecessor is `MMM_GEO_STRUCTURED_TIME_HOLDOUT_CERTIFICATION_001`,
branch `feat/mmm-geo-structured-time-holdout-certification-001`, blocked head
`e4ea5f3da3f00a2d684604d3f06b0421041b48c5`; no predecessor analytical artifact is used.

## Method and invariants

Five existing H6 worlds and their deterministic seeds were evaluated. For each channel and
geo, raw materialized media were transformed over the complete geo-time path using that
world's `spec.transform_truth` (geometric adstock followed by Hill saturation). Generative
contribution used the row/geo-specific `spec.true_beta_gc[geo][channel]`. No fitted transform
parameter or pooled fitted beta entered the truth calculation.

The intervention multiplies every raw channel by 1.10 at week indices 39 through 51. Recursive
state is recomputed over the full path, so pre-holdout state is carried naturally. Nuisance
terms are held fixed. The estimand is equal-row mean log-space Delta-mu and per-channel
contribution delta. Fitted quantities use the existing `RidgeBOMMMTrainer`, its selected
parameters, design matrix, and pooled coefficients only for comparison.

## Results

| World | Seed | Independent true Delta-mu | Fitted Delta-mu | Absolute error | Max channel contribution error |
|---|---:|---:|---:|---:|---:|
| WORLD-H6-PILOT-RETAIL-FULL-CONTROLS | 6600 | 0.045 | 0.002 | 0.042 | 0.009 |
| WORLD-H6-PILOT-RETAIL-OMITTED-CONTROLS | 6601 | 0.047 | 0.003 | 0.044 | 0.011 |
| WORLD-H6-PILOT-RETAIL-MEDIA-CORRELATED-CONTROLS | 6602 | 0.039 | 0.005 | 0.034 | 0.007 |
| WORLD-H6-PILOT-CPG-FULL-CONTROLS | 6603 | 0.044 | 0.005 | 0.039 | 0.008 |
| WORLD-H6-PILOT-AUTO-OMITTED-CONTROLS | 6604 | 0.039 | 0.020 | 0.019 | 0.007 |

The fitted Delta-mu is materially below independent truth in all five worlds. The fitted
comparator is now leakage-clean: every transform parameter and pooled coefficient is selected
by `RidgeBOMMMTrainer.fit(train)` on weeks `0:39` only. Full-path design construction uses those
training-derived parameters and carries only legitimate recursive media state into weeks `39:52`;
held-out outcomes never enter tuning, coefficients, nuisance fitting, or candidate selection.
The truth and fitted paths are independent, and this is recovery evidence rather than decision
invariance or production authority.

## Conclusions

| Area | Classification | Conclusion |
|---|---|---|
| H6 generative truth reconstruction | supported_for_implementation | Use `transform_truth` and row/geo-specific `true_beta_gc` for future certification truth. |
| Held-out contribution recovery | requires_more_evidence | The independent truth path is now defined, but broader worlds and channel-specific acceptance bars remain. |
| Held-out Delta-mu recovery | requires_more_evidence | Current five-world evidence shows material divergence; no promotion follows. |
| Production model behavior | rejected | No production numerical change is authorized by this artifact. |
| Retransformation, replay, optimizer, decision invariance | research_only | Explicitly deferred successors. |

## Evidence schema

The companion JSON uses this internal research schema:
```text
artifact_identity: {artifact_id, version, source_revision}
analysis: {analysis_id, truth_method, fitted_method, metrics}
worlds[]: {
  world_id, seed, n_geos, n_weeks, channels, holdout,
  transform_truth, true_beta_gc_source, fitted_params, fitted_parameter_provenance,
  true_delta_mu, fitted_delta_mu, delta_mu_abs_error,
  true_contribution_delta, fitted_contribution_delta, contribution_abs_error
}
provenance: {exact_command, runtime, no_production_changes}
```
The JSON is diagnostic evidence only and is not a production/package contract.

## Exact reproduction

Run from source revision `e39bd2b236c81c925357746f2cb461ba45066da1` (the `/tmp` file is only a
runtime copy of the Git-owned fenced program):

```sh
docker run --rm -e PYTHONPATH=/repo -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v /Users/phani/Desktop/MMM:/repo -v /tmp/h6_truth_alignment.py:/tmp/h6_truth_alignment.py -w /repo mmm-fixture-ready:local python /tmp/h6_truth_alignment.py
```

The complete executable used for the reported numbers is preserved below; it composes only
existing H6 generators, transforms, Ridge trainer/design matrix, and prediction functions.

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

HOLDOUT_START = 39
HOLDOUT_END = 52
SCALE = 1.10
sys.modules["optuna"] = None  # Use trainer's existing seeded grid fallback for exact reproducibility.


def truth_media(panel, spec, *, intervention: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ordered = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    transformed: dict[str, np.ndarray] = {c: np.zeros(len(ordered)) for c in spec.channels}
    contribution: dict[str, np.ndarray] = {c: np.zeros(len(ordered)) for c in spec.channels}
    for geo, idx in ordered.groupby("geo_id", sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        for ch in spec.channels:
            raw = ordered.loc[positions, ch].to_numpy(dtype=float)
            if intervention:
                raw[HOLDOUT_START:HOLDOUT_END] *= SCALE
            truth = spec.transform_truth[ch]
            ad = _geometric_adstock(raw, truth["decay"])
            sat = _hill_transform(ad, truth["hill_half"], truth["hill_slope"])
            transformed[ch][positions] = sat
            contribution[ch][positions] = sat * np.asarray(spec.true_beta_gc[geo][ch])
    return transformed, contribution


def run_world(world_id: str) -> dict[str, object]:
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec, panel_seed=spec.panel_seed)
    schema = h6_panel_schema(spec)
    config = h6_ridge_config(spec)
    trainer = RidgeBOMMMTrainer(config, schema)
    ordered_panel = panel.sort_values(["geo_id", "week_start_date"]).reset_index(drop=True)
    train = ordered_panel.loc[
        ordered_panel.groupby("geo_id", sort=False).cumcount() < HOLDOUT_START
    ].copy()
    fit = trainer.fit(train)
    art = fit["artifacts"]
    params = {k: float(v) for k, v in art.best_params.items()}
    bundle = build_design_matrix(panel, schema, config, decay=params["decay"], hill_half=params["hill_half"], hill_slope=params["hill_slope"])
    baseline = predict_ridge(bundle.X, art.coef, art.intercept)
    altered = panel.copy()
    ordered = altered.sort_values(["geo_id", "week_start_date"]).reset_index()
    for ch in spec.channels:
        ordered.loc[ordered.groupby("geo_id", sort=False).cumcount().between(HOLDOUT_START, HOLDOUT_END - 1), ch] *= SCALE
    altered = ordered.sort_values("index").drop(columns="index")
    bundle_alt = build_design_matrix(altered, schema, config, decay=params["decay"], hill_half=params["hill_half"], hill_slope=params["hill_slope"])
    candidate = predict_ridge(bundle_alt.X, art.coef, art.intercept)
    fitted_delta = candidate - baseline
    _, truth_base = truth_media(panel, spec, intervention=False)
    _, truth_alt = truth_media(panel, spec, intervention=True)
    mask = bundle.df_aligned.groupby("geo_id", sort=False).cumcount().between(HOLDOUT_START, HOLDOUT_END - 1).to_numpy()
    truth_delta_by_channel = {ch: float(np.mean((truth_alt[ch] - truth_base[ch])[mask])) for ch in spec.channels}
    fitted_delta_by_channel = {ch: float(np.mean(((bundle_alt.X[:, i] - bundle.X[:, i]) * art.coef[i])[mask])) for i, ch in enumerate(spec.channels)}
    true_delta = float(np.mean(np.sum([truth_alt[ch][mask] - truth_base[ch][mask] for ch in spec.channels], axis=0)))
    fitted_delta_mu = float(np.mean(fitted_delta[mask]))
    return {
        "world_id": world_id,
        "seed": spec.panel_seed,
        "n_geos": spec.n_geos,
        "n_weeks": spec.n_weeks,
        "channels": list(spec.channels),
        "training_window": {"start_week_index": 0, "end_week_index_exclusive": HOLDOUT_START, "rows_per_geo": HOLDOUT_START, "fit_outcomes": "training rows only"},
        "holdout": {"start_week_index": HOLDOUT_START, "end_week_index_exclusive": HOLDOUT_END, "rows_per_geo": HOLDOUT_END - HOLDOUT_START, "intervention": "all channels multiplied by 1.10 on held-out weeks", "aggregation": "equal-row mean", "nuisance_fixed": True},
        "transform_truth": spec.transform_truth,
        "true_beta_gc_source": "spec.true_beta_gc[geo][channel]",
        "fitted_params": {k: round(v, 2) for k, v in params.items()},
        "fitted_parameter_provenance": "RidgeBOMMMTrainer.fit(training rows week indices 0:39 only); full-path design construction carries legitimate recursive media state",
        "truth_parameter_provenance": "spec.transform_truth and spec.true_beta_gc[geo][channel], independently reconstructed from raw H6 media",
        "true_delta_mu": round(true_delta, 3),
        "fitted_delta_mu": round(fitted_delta_mu, 3),
        "delta_mu_abs_error": round(abs(fitted_delta_mu - true_delta), 3),
        "true_contribution_delta": {ch: round(v, 3) for ch, v in truth_delta_by_channel.items()},
        "fitted_contribution_delta": {ch: (0.0 if abs(v) < 0.0005 else round(v, 3)) for ch, v in fitted_delta_by_channel.items()},
        "contribution_abs_error": {ch: round(abs(fitted_delta_by_channel[ch] - truth_delta_by_channel[ch]), 3) for ch in spec.channels},
    }


out = {
    "artifact_identity": {"artifact_id": "MMM_H6_HOLDOUT_TRUTH_ALIGNMENT_CERTIFICATION_001", "version": "1.0.0", "source_revision": "e39bd2b236c81c925357746f2cb461ba45066da1"},
    "analysis": {"analysis_id": "H6-TRUTH-ALIGN-001", "truth_method": "raw H6 media -> spec.transform_truth adstock/saturation over full geo-time path -> spec.true_beta_gc[geo][channel]", "fitted_method": "existing RidgeBOMMMTrainer and design-matrix transform parameters", "metrics": ["true_delta_mu", "fitted_delta_mu", "delta_mu_abs_error", "per_channel_contribution_abs_error"]},
    "worlds": [run_world(w) for w in H6_PILOT_WORLD_IDS],
    "provenance": {"exact_command": "docker run --rm -e PYTHONPATH=/repo -v /Users/phani/Desktop/MMM:/repo -v /tmp/h6_truth_alignment.py:/tmp/h6_truth_alignment.py -w /repo mmm-fixture-ready:local python /tmp/h6_truth_alignment.py", "runtime": "mmm-fixture-ready:local Python 3.11; deterministic H6 seeds", "no_production_changes": True},
}
Path("/tmp/h6_truth_alignment_results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
```

## Limitations and successors

H6 pilot worlds are small and synthetic; this is not broad certification, causal identification,
retransformation validation, or decision invariance. Fitted Ridge uses its existing BO selection,
while truth uses the independent generative specification. The next reviewable milestones are
broader independent-truth world coverage and separately authorized held-out calibration/replay
certification. No registry row, production authority, or DR-04 threshold is promoted here.

**Unresolved execution-blocking design questions:** none.
