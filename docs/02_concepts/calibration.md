# Experiment calibration

Replay calibration checks **internal consistency** between observed experiment lifts and model-implied lifts under explicit estimands. It does **not** prove causal validity, incrementality, or that the MMM is well identified.

## Full-panel replay semantics (legacy + evidence-registry)

Both replay modes share one transform path:

1. **Full-panel frames** — observed and counterfactual spend cover the sorted modeling panel (not a pre-window slice).
2. **Pre-window adstock** — geometric adstock/saturation run on the full panel so carryover before the experiment window is preserved.
3. **Counterfactual spend** — multiplier applied only inside the experiment geo/time window.
4. **Lift evaluation** — implied vs observed lift aggregated **only** on rows in the experiment estimand mask (`lift_evaluated_on_estimand_mask_only: true`).

Canonical serialized mode:

```yaml
replay_estimand:
  replay_transform_mode: full_panel_transform_estimand_mask
```

Artifact fields `replay_uses_full_panel_transform` and `replay_transform_mode` should match this mode before trusting replay metrics in release review.

## Legacy replay (default, deprecated for new work)

Path-level **`CalibrationUnit`** JSON (`calibration.replay_units_path`) with observed/counterfactual spend frames.

- Enable with `use_replay_calibration: true` and `replay_mode: legacy` (default).
- Ridge+BO loss: mean standardized squared error per unit (unweighted).
- **Upgrade:** window-slice JSON units **with** `replay_estimand` are rebuilt to full-panel frames at train time.
- **Cannot upgrade safely:** units **without** `replay_estimand` — emit warning `legacy_replay_deprecated_use_evidence_registry`, skipped from replay loss. Do not assume window-slice-only units are comparable to evidence-registry replay.
- **Deprecation:** new calibration should use `replay_mode: evidence_registry` and the experiment evidence registry (see below).

## BO replay generalization (Ridge+BO, advisory)

Train replay loss in the BO objective uses a **full-panel refit** at the trial hyperparameters (**unchanged**). When time-series CV runs, a **holdout replay loss** is computed on the **same units** using the **last CV-fold** coefficients (diagnostic only — not a separate unit file or holdout fraction config).

| Field | Meaning |
|-------|---------|
| `calibration_refit_mode` | `full_panel_same_hyperparameters` — replay train path matches shipped-model refit |
| `replay_uses_full_panel_refit` | `true` — train replay uses full-panel coef |
| `replay_train_loss` | Replay loss with full-panel refit coef |
| `replay_holdout_loss` | Replay loss with last-fold coef when `replay_holdout_available` |
| `replay_holdout_available` | `false` when CV did not produce a holdout replay (must be disclosed in review) |
| `replay_generalization_gap` | `holdout_loss − train_loss` when holdout exists |
| `replay_generalization_gap_severity` | `none` (&lt;0.1), `moderate` (0.1–threshold), `severe` (≥ threshold) |
| `replay_overfit_warning` | Human-readable advisory when gap is moderate/severe |

Config (see [config_yaml.md](../01_getting_started/config_yaml.md)):

- `calibration.replay_generalization_gap_threshold` (default `0.25`)
- `calibration.block_on_severe_replay_gap` (default `false` — **warning only**; set `true` only when the org wants severe gap to invalidate `model_release`)

**Not implemented:** `use_replay_holdout_split`, `replay_holdout_fraction`, `train_replay_units_path`, `holdout_replay_units_path`. Fold-aligned replay refit is planned as a follow-on design PR.

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
