# Decision artifact contract (Ridge semi-log prod)

Prod full-panel Δμ simulate/optimize enforces train↔decide alignment before scoring scenarios.

## Fingerprint match

- Compare `extension_report.data_fingerprint` (prefer `sha256_combined`) to the fingerprint of `config.data.path` at decide time.
- Legacy artifacts without `sha256_combined` compare `sha256_panel_keycols_sorted_csv` only (warning emitted).
- Override: `governance.allow_decision_fingerprint_mismatch: true` plus `governance.decision_fingerprint_mismatch_waiver_path` (signed JSON). Emits a severe warning on the decision payload.

Implementation: `mmm/governance/decision_fingerprint.py`, wired in `mmm/decision/service.py`.

## Ridge fit summary completeness

Required on `extension_report` for prod decide:

| Field | Requirement |
|-------|-------------|
| `ridge_fit_summary.coef` | Non-empty |
| `ridge_fit_summary.intercept` | Present |
| `ridge_fit_summary.best_params` | `decay`, `hill_half`, `hill_slope` |
| `ridge_fit_summary.model_form` | `semi_log` |
| `transform_policy` | Geometric + Hill manifest (`mmm_transform_policy_v1`) |
| `data_fingerprint` | Training panel/config lineage |

Decide-time YAML `transforms` must match `transform_policy` from training.

Implementation: `mmm/governance/decision_ridge_summary.py`.

## Replay refit (prod training)

- Default `calibration.replay_refit_mode: full_panel_refit` remains for research/backward compatibility.
- **Prod** Ridge training with replay calibration requires `fold_aligned`, `holdout_only_diagnostic`, or `calibration.full_panel_replay_refit_prod_waiver_path` when using `full_panel_refit`.
- Waiver usage is surfaced on `extension_report.replay_refit_prod_governance` and in `model_card.md` calibration section.

Implementation: `mmm/governance/replay_refit_prod_policy.py`.

## Causal disclaimer

These checks prove **internal consistency** between training artifacts and decide-time inputs. They do **not** prove that observational MMM coefficients or optimized budgets will match field incremental lift.
