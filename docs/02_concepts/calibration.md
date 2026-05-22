# Experiment calibration

## Legacy replay (default)

Path-level **`CalibrationUnit`** JSON (`calibration.replay_units_path`) with observed/counterfactual spend frames and explicit `replay_estimand`.

- Enable with `use_replay_calibration: true` and `replay_mode: legacy` (default).
- Ridge+BO loss: mean standardized squared error per unit (unweighted).
- **Transform path:** legacy ETL and evidence-registry replay share `build_full_panel_replay_frames` — full-panel adstock/saturation, counterfactual spend only inside the estimand window, lift aggregated on the estimand mask only. Canonical mode: `replay_transform_mode: full_panel_transform_estimand_mask`.
- **Deprecation:** window-slice units without `replay_estimand` cannot be upgraded; artifacts emit `legacy_replay_deprecated_use_evidence_registry`. Prefer `replay_mode: evidence_registry` for new calibration units.

## BO replay generalization (Ridge+BO, advisory)

Train replay loss in the BO objective uses a **full-panel refit** at the trial hyperparameters (unchanged). When time-series CV runs, a **holdout replay loss** is also computed using the **last CV-fold** coefficients (diagnostic only).

| Field | Meaning |
|-------|---------|
| `replay_train_loss` | Replay loss with full-panel refit coef |
| `replay_holdout_loss` | Replay loss with last-fold coef (if CV ran) |
| `replay_generalization_gap` | `holdout_loss − train_loss` when holdout exists |
| `replay_generalization_gap_severity` | `none` (&lt;0.1), `moderate` (0.1–threshold), `severe` (≥ threshold) |
| `replay_overfit_warning` | Human-readable advisory when gap is moderate/severe |

Config: `calibration.replay_generalization_gap_threshold` (default `0.25`), `calibration.block_on_severe_replay_gap` (default `false` — warn only; set `true` to invalidate `model_release` on severe gap).

**Replay does not establish causal lift** — only consistency between observed experiment lifts and model-implied lifts under explicit estimands.

## Evidence-registry replay (opt-in, PR 2)

**`ExperimentEvidence`** registry (`calibration.evidence_registry_path`) with compatibility + quality weighting.

Requires:

```yaml
calibration:
  use_replay_calibration: true
  replay_mode: evidence_registry
  evidence_weighting_enabled: true
  compatibility_resolver_enabled: true
  evidence_registry_path: path/to/evidence.json
```

Weighted objective:

```
weighted_replay_loss = sum(w_i * ((mmm_lift_i - lift_i) / se_i)^2) / sum(w_i)
```

Incompatible, diagnostic-only, expired, or missing-SE (prod) evidence is **excluded** from the objective but listed in `evidence_weighted_replay_summary`.

**Prod (Ridge):** `replay_mode: evidence_registry` requires a passing `evidence_weighted_replay_summary` gate (see [experiment_evidence.md](experiment_evidence.md)). Weighted evidence calibrates the model objective; it does not establish causal lift or DMA-level truth from national/allocated shocks.

**Bayesian (research):** `bayesian.use_experiment_likelihood` adds a posterior likelihood term; see [bayesian.md](bayesian.md). Does not enable prod budget decisioning.

See [experiment_evidence.md](experiment_evidence.md) for compatibility and aggregate-only rules.

## Row-level experiments (extensions)

`ExperimentObservation` records (CSV or JSON list) with lift and optional `lift_se` — used for matching traces and scheduler diagnostics, not the Ridge BO replay path unless converted to replay units.

- **Bayesian**: experiment likelihood on matched lift summaries is research-only (not prod decisioning).

Always document which uncertainty source (experiment SE vs posterior width) is active in a given report.
