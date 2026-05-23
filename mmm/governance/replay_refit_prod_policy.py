"""Prod training policy for calibration replay refit mode (no optimistic full-panel refit without waiver)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mmm.calibration.replay_refit_mode import validate_replay_refit_mode
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError


class FullPanelReplayRefitWaiverArtifact(BaseModel):
    waiver_id: str = Field(min_length=4)
    created_at: str
    reason: str = Field(min_length=8)
    owner: str | None = None

    @field_validator("created_at")
    @classmethod
    def _parse_created(cls, v: str) -> str:
        from datetime import datetime

        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


def load_full_panel_replay_refit_waiver(path: str | Path) -> FullPanelReplayRefitWaiverArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return FullPanelReplayRefitWaiverArtifact.model_validate(raw)


def validate_prod_replay_refit_mode(cfg: MMMConfig) -> None:
    """
    Prod Ridge training: forbid ``replay_refit_mode=full_panel_refit`` unless an explicit waiver is on file.

    Research and non-prod environments keep the backward-compatible default.
    """
    if cfg.run_environment != RunEnvironment.PROD:
        return
    if cfg.framework != Framework.RIDGE_BO:
        return
    if not cfg.calibration.use_replay_calibration:
        return

    mode = validate_replay_refit_mode(cfg.calibration.replay_refit_mode)
    if mode != "full_panel_refit":
        return

    wpath = str(cfg.calibration.full_panel_replay_refit_prod_waiver_path or "").strip()
    if not wpath:
        raise PolicyError(
            "run_environment=prod with calibration.use_replay_calibration and "
            "calibration.replay_refit_mode=full_panel_refit requires "
            "calibration.full_panel_replay_refit_prod_waiver_path (signed waiver). "
            "Prefer replay_refit_mode=fold_aligned or holdout_only_diagnostic for production training."
        )
    load_full_panel_replay_refit_waiver(wpath)


def replay_refit_prod_governance_note(cfg: MMMConfig) -> dict[str, Any] | None:
    """Advisory block for extension reports / model cards when a waiver-backed full-panel refit is used."""
    if cfg.run_environment != RunEnvironment.PROD:
        return None
    if cfg.framework != Framework.RIDGE_BO:
        return None
    mode = validate_replay_refit_mode(cfg.calibration.replay_refit_mode)
    if mode != "full_panel_refit":
        return None
    wpath = str(cfg.calibration.full_panel_replay_refit_prod_waiver_path or "").strip()
    if not wpath:
        return None
    waiver = load_full_panel_replay_refit_waiver(wpath)
    return {
        "replay_refit_mode": mode,
        "prod_full_panel_refit_waiver": True,
        "waiver_id": waiver.waiver_id,
        "reason": waiver.reason,
        "severity": "warning",
        "message": (
            "PROD REPLAY REFIT OVERRIDE: BO objective uses full_panel_refit replay coefficients; "
            "replay loss may be optimistic vs calendar CV. Prefer fold_aligned for production."
        ),
    }
