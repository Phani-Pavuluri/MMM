# Cross-validation modes

## `split_axis`

| Value | Use when |
|-------|-----------|
| `calendar_week` | Weekly geo panels share a **global** calendar; train/val masks use the same calendar week index across geos. |
| `geo_rank` | Legacy: dense week rank **within each geo**, synchronized cuts (deprecated for most production panels). |
| `geo_blocked` | Hold out **entire geos** in validation; no time overlap leakage across geos. |

## Purged / embargo time

`cv.gap_weeks` inserts a **gap** between the last training calendar week and the first validation week for rolling/expanding calendar strategies (purge-style separation).

## When to use rolling vs expanding

- **Rolling**: longer history; stable seasonality; want multiple folds near the end of the series.
- **Expanding**: short history; early regime learning.
- **Geo-blocked**: geo-level heterogeneity or staggered rollout; complements calendar splits.

## Design-matrix masks vs CV

CV split objects yield ``(train_loss_mask, val_loss_mask)`` row booleans aligned to the **sorted** panel. Use ``mmm.features.design_matrix.design_masks_from_cv_split`` to wrap those into a :class:`~mmm.features.design_matrix.DesignMatrixMasks` contract for ``build_design_matrix(..., masks=...)`` when a single fold should drive lineage or diagnostics.

**Ridge BO hyperparameter search (current behavior):** the design matrix is built on the **full** sorted panel so geometric adstock carryover uses only past spend within each geo. For each CV fold, :func:`~mmm.features.design_matrix.apply_masks_for_fit` passes only **training** rows to ``fit_ridge``; validation rows are scored with ``predict_ridge`` on held-out indices. Validation targets never enter the fold fit loss. Replay calibration in the BO objective uses a separate full-panel refit for the same hyperparameters as the shipped model.
