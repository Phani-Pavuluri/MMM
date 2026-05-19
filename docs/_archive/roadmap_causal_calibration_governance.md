# Roadmap: causal contracts, calibration replay, governance, and services

Canonical vocabulary for targets, calibration estimands, contributions, and optimization safety lives in **`mmm/contracts/semantics.py`** (import these types in services and reports; avoid duplicate string literals).

This document merges the implementation sequence with **explicit causal contracts**, **replay semantics**, **normalization profiles**, **baseline stress tests**, **decomposition honesty**, **environment-aware governance**, **curve stress tests**, **feature lineage**, **Bayesian PPC**, and **orchestration vs services** split.

---

## Refined implementation order

1. **Single entry `build_design_matrix` + full-history adstock** with explicit masks (see §1).
2. **Calendar-aware CV** (or geo-blocked) using the same mask contract — no leakage of val targets into train-time state.
3. **Replay-based calibration** with treatment mapping semantics (see §2) — *before* hardening governance/optimizer so downstream logic anchors on the right quantity.
4. **Bayesian prior split** (media vs controls) + **prior predictive** and **posterior predictive** checks, persisted (see §9).
5. **Governance null semantics + policy layer** with **environment profiles** (see §6).
6. **Objective normalization** via **profiles**, not only raw config knobs (see §3) + baseline cleanup.
7. **Per-channel response curves** + **stress tests** (see §7) + optimizer reads **artifact-linked** curves/mROI.
8. **CLI and reporting** consume persisted lineage and flags (`safe_for_budgeting`, etc.).

---

## 1. Full-history adstock — explicit causal contract

The design-matrix builder must distinguish four roles (boolean masks per row, aligned to `df.index`):

| Mask | Meaning |
|------|--------|
| `history_mask` | Row may contribute **past** spend/state for **recursive** adstock / distributed lag state (includes pre-train history if present). |
| `train_loss_mask` | Row contributes to **training** loss (outcome observed, model uses only information allowed at that row’s decision time). |
| `val_loss_mask` | Row contributes to **validation** loss. |
| `calibration_mask` | Row participates in **calibration replay** windows (may differ from val for GeoX holdout design). |

**Do not** collapse these into a single `row_weights_for_loss` without retaining the four masks in **metadata** returned by `build_design_matrix`.

Suggested return shape:

```python
@dataclass
class DesignMatrixBundle:
    X: np.ndarray
    y_modeling: np.ndarray
    meta: dict  # history_mask, train_loss_mask, val_loss_mask, calibration_mask, feature_lineage (§8)
```

**Contract (normative):** features for row `i` may use information only from rows `j` with `history_mask[j]` and time ≤ allowed information time for `i` (define per `week_column`). Val loss rows must not use post-cutoff outcomes to fit train parameters (only for evaluation).

---

## 2. Replay-based calibration — treatment mapping semantics

Extend **`CalibrationUnit`** (new or in `mmm/calibration/contracts.py`) so each unit encodes:

- `treated_channel_ids` (or names + index map)
- `observed_spend_frame` — panel slice: geo × week × channel spend **as observed**
- `counterfactual_spend_frame` — same shape, **explicit** counterfactual path
- `post_window`, `ramp_window`, `washout_window` (week ranges or indices)
- `effect_horizon: Literal["short_run", "long_run"]` (or enum)
- `treatment_intensity` definition (e.g. delta spend vs % lift on log scale)

**Replay** must consume these frames + the **same** `predict`/μ path as production, then compare to **observed lift** with uncertainty. Geo/time/KPI match alone is insufficient without **path-level** treatment definition.

---

## 3. Objective normalization — profile-based policy

Add a **`NormalizationProfile`** (e.g. `strict_prod`, `research`, `debug`) under config, implemented in **`mmm/evaluation/normalization_policy.py`** (or similar):

| Profile | Behavior (example) |
|---------|---------------------|
| `strict_prod` | Predictive term scaled by **baseline metric**; calibration by **experiment SE**; stability by **cross-fold baseline instability**; plausibility bounded to `[0,1]` or fixed cap. |
| `research` | Looser defaults, more logging. |
| `debug` | Raw components, no normalization. |

**Goal:** weights stay interpretable; users cannot easily pick arbitrary destructive scaling in prod.

Config sketch:

```yaml
objective:
  normalization_profile: strict_prod   # research | debug
```

---

## 4. Baselines — causal stress test

Add at least one baseline that breaks **timing coincidence**:

- **Media-shuffled within geo**: permute spend columns **within** `(geo_id)` blocks (preserve marginal spend distribution, destroy temporal co-movement with outcome), refit same ridge template; or
- **Permuted-within-geo** channel time series (circular shift per channel with random offset).

If the main model does not beat this by a margin, **flag** `signal_may_be_spurious_timing` in baseline report.

Implement in **`mmm/evaluation/baselines.py`** + tests.

---

## 5. Decomposition — impossible to misread

Extend **`DecompositionResult`** (`mmm/decomposition/engine.py`):

- `scale: Literal["log_surrogate", "level_approx", ...]`
- `is_exact_additive: bool` (usually `False` for log link + nonlinear transforms)
- `safe_for_budgeting: bool`
- `notes: list[str]` (e.g. “not literal incremental dollars”)

Reports and **`GovernanceService`** must read these flags so log-scale rows cannot be presented as dollar truth without an explicit second step.

---

## 6. Governance — environment-aware policy

Formalize **`run_environment: Literal["dev", "research", "staging", "prod"]`** on top-level config (or env var `MMM_ENV`).

**`mmm/governance/policy.py`** maps environment → defaults:

| Environment | Example |
|-------------|--------|
| `dev` | Missing identifiability → warn only. |
| `prod` | Missing identifiability → **fail closed** for `approved_for_optimization`; stricter gates. |

Avoid ad-hoc boolean sprawl; **policy table** drives defaults; user overrides require explicit `override_unsafe: true` in prod (logged).

---

## 7. Response curves — stress tests before optimizer

Beyond `ResponseDiagnostics` (monotonicity / cliffs):

- **Local derivative smoothness** (second derivative or gradient Lipschitz proxy)
- **Finite-difference stability** under ε perturbation of spend grid
- **Monotonicity under posterior / bootstrap** draws (fraction of draws violating monotone media effect)
- **Optimizer sensitivity**: small spend perturbation → large ΔmROI ⇒ flag `numerically_unstable_for_sqp`

Implement **`mmm/decomposition/curve_stress.py`** (or extend `response_diagnostics.py`). Optimizer gate requires **diagnostics + stress** pass.

---

## 8. Design-matrix builder — single source of truth + lineage

`build_design_matrix` (proposed `mmm/features/design_matrix.py`) owns:

- Target transform (`semi_log` / `log_log`)
- Control preprocessing (scaling, missingness)
- Media pipeline: raw spend → adstock → saturation → optional extras
- **Feature lineage** dict / small DAG serialized to JSON artifact:

`raw_spend → geometric_adstock(decay=…) → hill(…) → column_7`

Persist with every run next to `resolved_config.yaml`.

---

## 9. Bayesian — prior and posterior predictive checks

After prior split fix:

- `pm.sample_prior_predictive` (or equivalent) → **`prior_predictive`** artifact
- **`posterior_predictive_check`** (observed vs replicated) → artifact
- Hard fail or warn in `strict_prod` if implied Y has absurd scales / mass at boundary

Wire into **`BayesianMMMTrainer.fit`** completion path and **`DiagnosticsService`**.

---

## 10. Extension runner — orchestration only

**`mmm/evaluation/extension_runner.py`** should remain thin:

| Service | Responsibility |
|---------|----------------|
| `CalibrationService` | match units, replay, loss vs experiments |
| `CurveService` | build curves, diagnose, stress-test, bundle for optimizer |
| `GovernanceService` | scorecard, env policy, approvals |
| `DiagnosticsService` | data quality, lag, geo, identifiability, falsification |

Runner: load config → call services → aggregate dict → `store.log_dict("extension_report", ...)`.

---

## Files to add (summary)

| Path | Role |
|------|------|
| `mmm/features/design_matrix.py` | `build_design_matrix` + `DesignMatrixBundle` + masks + lineage |
| `mmm/calibration/contracts.py` | `CalibrationUnit` + treatment frames/windows |
| `mmm/calibration/replay_lift.py` | Path-level implied lift |
| `mmm/evaluation/normalization_policy.py` | Profile-based objective scaling |
| `mmm/governance/policy.py` | Environment → rules |
| `mmm/decomposition/curve_stress.py` | FD / bootstrap monotonicity / sensitivity |
| `mmm/diagnostics/bayesian_ppc.py` | Prior/posterior predictive packaging |
| `mmm/services/*.py` | Calibration, Curve, Governance, Diagnostics services |

---

## Deprecations

- `week_index_per_geo`-only global splits: mark **`geo_rank`** split axis deprecated when `calendar_week` exists.
- Direct `build_channel_features_from_params` from trainers: thin wrapper to `build_design_matrix` only.

This document supersedes informal chat for the causal mask contract, replay semantics, profiles, and service boundaries.
