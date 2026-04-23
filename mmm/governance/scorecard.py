"""E13: model scorecard and approval flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmm.config.extensions import GovernanceConfig


@dataclass
class GovernanceScorecard:
    fit_mae: float
    baseline_mae: float
    identifiability_score: float
    calibration_loss: float | None
    falsification_flags: list[str]
    approved_for_reporting: bool
    approved_for_optimization: bool
    notes: list[str] = field(default_factory=list)
    decision_safe_uncertainty: bool = False
    identifiability_limits_decision_safety: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "fit_mae": self.fit_mae,
            "baseline_mae": self.baseline_mae,
            "identifiability_score": self.identifiability_score,
            "calibration_loss": self.calibration_loss,
            "falsification_flags": self.falsification_flags,
            "approved_for_reporting": self.approved_for_reporting,
            "approved_for_optimization": self.approved_for_optimization,
            "notes": self.notes,
            "decision_safe_uncertainty": self.decision_safe_uncertainty,
            "identifiability_limits_decision_safety": self.identifiability_limits_decision_safety,
        }


def build_scorecard(
    *,
    cfg: GovernanceConfig,
    fit_mae: float,
    baseline_mae: float,
    identifiability_score: float,
    calibration_loss: float | None,
    falsification_flags: list[str],
    beats_baselines: bool,
    decision_api_freeze: bool = True,
    calibration_is_replay: bool = False,
) -> GovernanceScorecard:
    notes: list[str] = []
    ok_fit = fit_mae <= cfg.max_mae_ratio_vs_baseline * baseline_mae
    ok_id = identifiability_score <= cfg.max_identifiability_risk
    id_limits_safety = float(identifiability_score) > float(cfg.max_identifiability_risk) * float(
        cfg.identifiability_decision_safety_margin
    )
    if calibration_loss is None:
        ok_cal = True
    elif calibration_is_replay:
        ok_cal = float(calibration_loss) < cfg.max_replay_calibration_chi2
    else:
        ok_cal = float(calibration_loss) < cfg.max_legacy_calibration_loss
    n_false = len(falsification_flags)
    if cfg.require_falsification_pass:
        ok_false = n_false == 0
    elif cfg.falsification_max_allowed_flags_for_optimization is not None:
        ok_false = n_false <= int(cfg.falsification_max_allowed_flags_for_optimization)
    else:
        ok_false = True
    appr_rep = bool(ok_fit and ok_id and beats_baselines)
    if decision_api_freeze:
        appr_opt = False
        notes.append(
            "decision_safety_freeze: approved_for_optimization is false while "
            "allow_unsafe_decision_apis is false (analysis-only)."
        )
        notes.append(
            "calibration_governance: experiment-facing calibration scoring is not decision-safe yet; "
            "do not treat calibration_loss as experiment-aligned lift validation."
        )
    else:
        appr_opt = bool(appr_rep and ok_cal and ok_false and ok_id)
    if not beats_baselines:
        notes.append("main model does not beat baselines by configured margin")
    if not ok_id:
        notes.append("identifiability_risk exceeds governance.max_identifiability_risk")
    if id_limits_safety:
        notes.append(
            "identifiability_approaches_governance_cap: decision_safety_downgraded_for_interpretation"
        )
    if not ok_false and cfg.falsification_max_allowed_flags_for_optimization is not None:
        notes.append(
            f"falsification_flag_count={n_false} exceeds governance.falsification_max_allowed_flags_for_optimization"
        )
    return GovernanceScorecard(
        fit_mae=fit_mae,
        baseline_mae=baseline_mae,
        identifiability_score=identifiability_score,
        calibration_loss=calibration_loss,
        falsification_flags=falsification_flags,
        approved_for_reporting=appr_rep,
        approved_for_optimization=appr_opt,
        notes=notes,
        decision_safe_uncertainty=False,
        identifiability_limits_decision_safety=bool(id_limits_safety or not ok_id),
    )
