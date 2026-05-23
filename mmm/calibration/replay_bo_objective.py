"""Replay calibration loss for Ridge+BO trials (refit mode aware)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.evidence_replay import (
    aggregate_weighted_evidence_replay_loss,
    uses_weighted_evidence_replay,
)
from mmm.calibration.replay_fold_aligned import compute_fold_aligned_replay_loss
from mmm.calibration.replay_generalization import build_replay_calibration_metadata
from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.calibration.replay_refit_mode import (
    ReplayRefitMode,
    build_replay_refit_disclosure,
    replay_refit_enters_objective,
)
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge


def evaluate_replay_calibration_for_trial(
    *,
    panel_df: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    bundle: Any,
    splits: list[tuple[np.ndarray, np.ndarray]],
    coef_rows: list[np.ndarray],
    intercept_rows: list[np.ndarray],
    replay_units: list,
    evidence_prepared: Any | None,
    legacy_warnings: list[str],
    replay_split_meta: dict[str, Any],
    refit_mode: ReplayRefitMode,
    decay: float,
    hill_half: float,
    hill_slope: float,
    alpha: float,
) -> tuple[float, dict[str, Any], str] | None:
    """
    Returns ``(objective_replay_loss, merged_meta, replay_mode_label)`` or ``None`` if inactive.
    """
    has_replay = bool(replay_units or evidence_prepared is not None)
    if not has_replay or not coef_rows:
        return None

    replay_mode = "evidence_registry" if uses_weighted_evidence_replay(config) else "legacy"
    holdout_loss: float | None = None
    rmeta: dict[str, Any] = {}
    rloss = 0.0

    if refit_mode == "fold_aligned":
        weighted = evidence_prepared.used if evidence_prepared is not None else None
        rloss, rmeta = compute_fold_aligned_replay_loss(
            panel_sorted=panel_df,
            schema=schema,
            config=config,
            units=replay_units,
            splits=splits,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
            alpha=alpha,
            weighted_entries=weighted,
        )
        refit_disc = build_replay_refit_disclosure("fold_aligned")
    elif refit_mode == "holdout_only_diagnostic":
        rloss = 0.0
        rmeta = {"replay_refit_mode": "holdout_only_diagnostic", "n_units": 0}
        refit_disc = build_replay_refit_disclosure("holdout_only_diagnostic")
    else:
        coef_full, intercept_full = fit_ridge(bundle.X, bundle.y_modeling, alpha)

        def predict_level(dfp: pd.DataFrame) -> np.ndarray:
            b = build_design_matrix(
                dfp,
                schema,
                config,
                decay=decay,
                hill_half=hill_half,
                hill_slope=hill_slope,
            )
            ylog = predict_ridge(b.X, coef_full, intercept_full)
            return np.exp(ylog)

        if uses_weighted_evidence_replay(config):
            assert evidence_prepared is not None
            rloss, rmeta = aggregate_weighted_evidence_replay_loss(
                evidence_prepared.used,
                predict_level,
                schema=schema,
                target_col=schema.target_column,
                config=config,
            )
        else:
            rloss, rmeta = aggregate_replay_calibration_loss(
                replay_units,
                predict_level,
                schema=schema,
                target_col=schema.target_column,
                config=config,
            )
        refit_disc = build_replay_refit_disclosure("full_panel_refit")

    if coef_rows and refit_mode != "holdout_only_diagnostic":

        def predict_holdout(dfp: pd.DataFrame) -> np.ndarray:
            b = build_design_matrix(
                dfp,
                schema,
                config,
                decay=decay,
                hill_half=hill_half,
                hill_slope=hill_slope,
            )
            ylog = predict_ridge(b.X, coef_rows[-1], intercept_rows[-1])
            return np.exp(ylog)

        if uses_weighted_evidence_replay(config):
            assert evidence_prepared is not None
            holdout_loss, _ = aggregate_weighted_evidence_replay_loss(
                evidence_prepared.used,
                predict_holdout,
                schema=schema,
                target_col=schema.target_column,
                config=config,
            )
        elif replay_units:
            holdout_loss, _ = aggregate_replay_calibration_loss(
                replay_units,
                predict_holdout,
                schema=schema,
                target_col=schema.target_column,
                config=config,
            )

    train_for_gap = float(rloss) if replay_refit_enters_objective(refit_mode, use_replay_calibration=True) else 0.0
    disclosure = build_replay_calibration_metadata(
        train_loss=train_for_gap,
        holdout_loss=float(holdout_loss) if holdout_loss is not None else None,
        n_units=int(rmeta.get("n_units", 0)),
        replay_mode_used=replay_mode,
        replay_transform_mode=rmeta.get("replay_transform_mode"),
        calibration_refit_mode=str(refit_disc.get("calibration_refit_mode", "")),
        replay_uses_full_panel_refit=bool(refit_disc.get("replay_uses_full_panel_refit", False)),
        replay_overfit_warning=str(refit_disc.get("replay_overfit_warning", "")),
        gap_severe_threshold=float(config.calibration.replay_generalization_gap_threshold),
        legacy_warnings=legacy_warnings,
        extra={
            k: v
            for k, v in refit_disc.items()
            if k
            not in (
                "calibration_refit_mode",
                "replay_uses_full_panel_refit",
                "replay_overfit_warning",
            )
        },
    )
    if replay_split_meta:
        disclosure["replay_split_meta"] = dict(replay_split_meta)
    merged = dict(rmeta) if isinstance(rmeta, dict) else {}
    merged.update(disclosure)
    merged.update(refit_disc)
    if uses_weighted_evidence_replay(config):
        merged["calibration_score_source"] = "evidence_registry_weighted_replay"
        merged["weighted_replay_loss"] = rmeta.get("weighted_replay_loss", rloss)
    elif refit_mode == "fold_aligned":
        merged["calibration_score_source"] = "fold_aligned_cv_replay"
    elif refit_mode == "holdout_only_diagnostic":
        merged["calibration_score_source"] = "predictive_only_replay_holdout_diagnostic"
    else:
        merged["calibration_score_source"] = "full_data_refit_same_hyperparameters_as_shipped_model"
    merged["calibration_refit_n_rows"] = int(bundle.X.shape[0])
    in_objective = replay_refit_enters_objective(refit_mode, use_replay_calibration=True)
    merged["train_vs_holdout_replay_loss"] = {
        "replay_loss_in_objective": float(rloss) if in_objective else 0.0,
        "replay_holdout_loss": holdout_loss,
        "replay_generalization_gap": merged.get("replay_generalization_gap"),
        "predictive_score_source": "time_series_cv_folds",
    }
    obj_loss = float(rloss) if replay_refit_enters_objective(refit_mode, use_replay_calibration=True) else 0.0
    return obj_loss, merged, replay_mode
