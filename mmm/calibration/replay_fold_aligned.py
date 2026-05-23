"""Fold-aligned replay calibration (train-only coef per CV fold)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec, _week_mask
from mmm.calibration.replay_lift import implied_lift_from_counterfactual
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import apply_masks_for_fit, build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge


def _estimand_overlaps_validation_window(
    panel: pd.DataFrame,
    schema: PanelSchema,
    spec: ReplayEstimandSpec,
    val_mask: np.ndarray,
) -> bool:
    """True when any validation-row week falls inside the unit estimand window."""
    wcol = schema.week_column
    val_weeks = panel.loc[val_mask, wcol]
    if len(val_weeks) == 0:
        return False
    em = _week_mask(val_weeks, spec.week_start, spec.week_end)
    return bool(np.any(em))


def compute_fold_aligned_replay_loss(
    *,
    panel_sorted: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    units: list[CalibrationUnit],
    splits: list[tuple[np.ndarray, np.ndarray]],
    decay: float,
    hill_half: float,
    hill_slope: float,
    alpha: float,
    weighted_entries: list[Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Aggregate replay loss across CV folds using fold-specific train-only coefficients.

    Units whose estimand window does not overlap a fold's validation period are skipped for that fold.
    Full-panel spend frames are preserved for adstock; only the coefficient object changes per fold.
    """
    if not splits:
        return 0.0, {
            "n_units": 0,
            "fold_replay_losses": [],
            "fold_replay_units_used": 0,
            "fold_replay_units_skipped": 0,
            "replay_fold_alignment_warnings": ["no_cv_splits_for_fold_aligned_replay"],
        }

    bundle = build_design_matrix(
        panel_sorted,
        schema,
        config,
        decay=decay,
        hill_half=hill_half,
        hill_slope=hill_slope,
    )
    panel_aligned = bundle.df_aligned
    fold_losses: list[float] = []
    fold_meta: list[dict[str, Any]] = []
    total_used = 0
    total_skipped = 0
    warnings: list[str] = []
    z2_all: list[float] = []

    for fold_idx, (train_mask, val_mask) in enumerate(splits):
        if len(train_mask) != len(bundle.df_aligned):
            raise RuntimeError("CV masks length mismatch for fold-aligned replay")
        X_tr, y_tr_t = apply_masks_for_fit(bundle, train_mask)
        coef_fold, intercept_fold = fit_ridge(X_tr, y_tr_t, alpha)

        def predict_level(
            dfp: pd.DataFrame,
            *,
            _coef: np.ndarray = coef_fold,
            _intercept: np.ndarray = intercept_fold,
        ) -> np.ndarray:
            b = build_design_matrix(
                dfp,
                schema,
                config,
                decay=decay,
                hill_half=hill_half,
                hill_slope=hill_slope,
            )
            ylog = predict_ridge(b.X, _coef, _intercept)
            return np.exp(ylog)

        fold_z2: list[float] = []
        fold_used = 0
        fold_skipped = 0

        entries: list[tuple[CalibrationUnit, float]] = []
        if weighted_entries:
            for ent in weighted_entries:
                entries.append((ent.unit, float(ent.evidence_weight)))
        else:
            for u in units:
                entries.append((u, 1.0))

        for unit, weight in entries:
            if (
                unit.observed_spend_frame is None
                or unit.counterfactual_spend_frame is None
                or unit.observed_lift is None
                or not unit.replay_estimand
            ):
                fold_skipped += 1
                continue
            spec = ReplayEstimandSpec.from_dict(unit.replay_estimand)
            if not _estimand_overlaps_validation_window(panel_aligned, schema, spec, val_mask):
                fold_skipped += 1
                continue
            r = implied_lift_from_counterfactual(
                panel_observed=unit.observed_spend_frame,
                panel_counterfactual=unit.counterfactual_spend_frame,
                predict_fn=predict_level,
                schema=schema,
                estimand=spec,
            )
            implied = float(r["implied_mean_delta"])
            se = float(unit.lift_se) if unit.lift_se is not None and unit.lift_se > 0 else 1.0
            diff = implied - float(unit.observed_lift)
            fold_z2.append(float(weight * (diff**2) / (se**2 + 1e-12)))
            fold_used += 1

        total_used += fold_used
        total_skipped += fold_skipped
        if fold_z2:
            fl = float(np.mean(fold_z2))
            fold_losses.append(fl)
            fold_meta.append(
                {
                    "fold_index": fold_idx,
                    "replay_loss": fl,
                    "n_units_used": fold_used,
                    "n_units_skipped": fold_skipped,
                }
            )
        else:
            warnings.append(f"fold_{fold_idx}: no replay units overlapped validation window")

    if not z2_all and fold_losses:
        z2_all = fold_losses
    mean_loss = float(np.mean(fold_losses)) if fold_losses else 0.0
    return mean_loss, {
        "n_units": len(units) or len(weighted_entries or []),
        "mean_standardized_sq_error": mean_loss,
        "fold_replay_losses": fold_losses,
        "fold_replay_meta": fold_meta,
        "fold_replay_units_used": int(total_used),
        "fold_replay_units_skipped": int(total_skipped),
        "replay_fold_alignment_warnings": warnings,
        "replay_transform_mode": "full_panel_transform_estimand_mask",
        "replay_uses_full_panel_transform": True,
    }
