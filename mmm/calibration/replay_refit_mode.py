"""Replay calibration refit modes (BO objective honesty)."""

from __future__ import annotations

from typing import Any, Literal

ReplayRefitMode = Literal["full_panel_refit", "fold_aligned", "holdout_only_diagnostic"]

REPLAY_REFIT_MODES: frozenset[str] = frozenset(
    {"full_panel_refit", "fold_aligned", "holdout_only_diagnostic"}
)

FULL_PANEL_REPLAY_OPTIMISM_WARNING = (
    "replay_refit_mode=full_panel_refit: replay calibration uses full-panel refit coefficients; "
    "loss may be optimistic vs time-series CV. Consider replay_refit_mode=fold_aligned."
)


def validate_replay_refit_mode(mode: str) -> ReplayRefitMode:
    if mode not in REPLAY_REFIT_MODES:
        raise ValueError(
            f"calibration.replay_refit_mode must be one of {sorted(REPLAY_REFIT_MODES)!r}, got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def replay_refit_enters_objective(mode: ReplayRefitMode, *, use_replay_calibration: bool) -> bool:
    """Whether replay loss is added to the BO composite objective."""
    if not use_replay_calibration:
        return False
    return mode != "holdout_only_diagnostic"


def build_replay_refit_disclosure(
    mode: ReplayRefitMode,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "replay_refit_mode": mode,
        "calibration_refit_mode": (
            "full_panel_same_hyperparameters"
            if mode == "full_panel_refit"
            else ("fold_aligned_cv" if mode == "fold_aligned" else "holdout_diagnostic_only")
        ),
        "replay_uses_full_panel_refit": mode == "full_panel_refit",
    }
    if mode == "full_panel_refit":
        out["replay_overfit_warning"] = FULL_PANEL_REPLAY_OPTIMISM_WARNING
    if extra:
        out.update(extra)
    return out
