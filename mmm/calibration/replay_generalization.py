"""Holdout replay generalization diagnostics (advisory; does not change BO objective)."""

from __future__ import annotations

from typing import Any, Literal

ReplayGapSeverity = Literal["none", "moderate", "severe"]


def replay_generalization_gap_severity(
    gap: float | None,
    *,
    moderate_threshold: float = 0.1,
    severe_threshold: float = 0.25,
) -> ReplayGapSeverity:
    if gap is None:
        return "none"
    if gap >= severe_threshold:
        return "severe"
    if gap >= moderate_threshold:
        return "moderate"
    return "none"


def build_replay_calibration_metadata(
    *,
    train_loss: float,
    holdout_loss: float | None,
    n_units: int,
    replay_mode_used: str,
    replay_transform_mode: str | None,
    gap_moderate_threshold: float = 0.1,
    gap_severe_threshold: float = 0.25,
    legacy_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Explicit replay calibration disclosure for BO trial / extension artifacts."""
    gap = (
        float(holdout_loss - train_loss)
        if holdout_loss is not None and train_loss is not None
        else None
    )
    severity = replay_generalization_gap_severity(
        gap,
        moderate_threshold=gap_moderate_threshold,
        severe_threshold=gap_severe_threshold,
    )
    warning = ""
    if severity == "severe":
        warning = (
            f"Replay generalization gap {gap:.4f} >= {gap_severe_threshold} (train vs holdout coef): "
            "replay calibration may be optimistic vs CV holdout."
        )
    elif severity == "moderate":
        warning = (
            f"Replay generalization gap {gap:.4f} in [{gap_moderate_threshold}, {gap_severe_threshold}): "
            "monitor replay vs predictive score."
        )
    meta: dict[str, Any] = {
        "calibration_refit_mode": "full_panel_same_hyperparameters",
        "replay_uses_full_panel_refit": True,
        "replay_training_units": int(n_units),
        "replay_holdout_units": int(n_units) if holdout_loss is not None else 0,
        "replay_holdout_available": holdout_loss is not None,
        "replay_train_loss": float(train_loss),
        "replay_holdout_loss": float(holdout_loss) if holdout_loss is not None else None,
        "replay_generalization_gap": gap,
        "replay_generalization_gap_severity": severity,
        "replay_overfit_warning": warning,
        "replay_transform_mode": replay_transform_mode,
        "replay_mode_used": replay_mode_used,
        "predictive_score_source": "time_series_cv_folds",
    }
    if legacy_warnings:
        meta["legacy_replay_upgrade_warnings"] = list(legacy_warnings)
    return meta
