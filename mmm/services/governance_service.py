from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.evaluation.baselines import BaselineComparisonReport
from mmm.governance.decision_safety import report_decision_safety_section
from mmm.governance.policy import approved_for_optimization_with_policy, policy_for_environment
from mmm.governance.scorecard import GovernanceScorecard, build_scorecard


def build_governance_bundle(
    *,
    config: MMMConfig,
    panel: pd.DataFrame,
    schema: PanelSchema,
    yhat: np.ndarray,
    baselines: BaselineComparisonReport,
    identifiability_json: dict[str, Any],
    falsification_flags: list[str],
    calibration_loss: float | None,
    calibration_is_replay: bool = False,
    calibration_raw: dict[str, Any] | None = None,
    bayesian_decision_inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    id_score = float(identifiability_json.get("identifiability_score", 0.0))
    sc = build_scorecard(
        cfg=config.extensions.governance,
        fit_mae=float(np.mean(np.abs(panel[schema.target_column].to_numpy(dtype=float) - yhat))),
        baseline_mae=float(baselines.mae_no_media),
        identifiability_score=id_score,
        calibration_loss=calibration_loss,
        falsification_flags=falsification_flags,
        beats_baselines=bool(baselines.beats_baselines),
        decision_api_freeze=not config.allow_unsafe_decision_apis,
        calibration_is_replay=calibration_is_replay,
    )
    pol = policy_for_environment(config.run_environment)
    id_ok = id_score <= config.extensions.governance.max_identifiability_risk
    appr_opt, pol_notes = approved_for_optimization_with_policy(
        base_approved=sc.approved_for_optimization,
        env=config.run_environment,
        override_unsafe=config.override_unsafe,
        identifiability_risk_ok=id_ok,
    )
    notes = list(sc.notes) + pol_notes
    if bayesian_decision_inference is not None and config.framework == Framework.BAYESIAN:
        post_ok = bool(bayesian_decision_inference.get("posterior_diagnostics_ok"))
        ppc_ok = bool(bayesian_decision_inference.get("posterior_predictive_ok"))
        if config.extensions.governance.require_posterior_predictive_pass and not ppc_ok:
            appr_opt = False
            notes.append("governance_require_posterior_predictive_pass_failed")
        if config.run_environment == RunEnvironment.PROD and (not post_ok or not ppc_ok):
            appr_opt = False
            notes.append("prod_bayesian_decision_inference_not_ok_blocks_optimization")
    if config.run_environment == RunEnvironment.PROD and config.framework == Framework.BAYESIAN:
        notes.append("bayesian_prod_experimental_only_optimization_disabled")
        appr_opt = False
    if baselines.signal_may_be_spurious_timing:
        notes.append("signal_may_be_spurious_timing_vs_shuffled_media")
    enriched = GovernanceScorecard(
        fit_mae=sc.fit_mae,
        baseline_mae=sc.baseline_mae,
        identifiability_score=sc.identifiability_score,
        calibration_loss=sc.calibration_loss,
        falsification_flags=sc.falsification_flags,
        approved_for_reporting=sc.approved_for_reporting,
        approved_for_optimization=appr_opt,
        notes=notes,
        decision_safe_uncertainty=False,
    )
    js = enriched.to_json()
    js["environment_policy"] = {
        "run_environment": config.run_environment.value,
        "override_unsafe": config.override_unsafe,
        "log_level": pol.log_level,
    }
    if identifiability_json.get("skipped"):
        ident_status = "skipped"
    elif "identifiability_score" in identifiability_json:
        ident_status = "evaluated"
    else:
        ident_status = "unknown"
    js["identifiability_status"] = ident_status
    js["calibration_status"] = (
        "replay"
        if calibration_is_replay and calibration_loss is not None
        else ("none" if calibration_loss is None else "legacy")
    )
    if calibration_raw:
        js["calibration_raw"] = calibration_raw
    js["decision_safety"] = report_decision_safety_section(
        allow_unsafe_decision_apis=config.allow_unsafe_decision_apis,
    )
    if bayesian_decision_inference is not None:
        js["bayesian_decision_inference"] = bayesian_decision_inference
    return js
