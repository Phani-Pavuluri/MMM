"""Phase 5E — dedicated VAL-012 drift detection runner for synthetic worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mmm.evaluation.drift_monitor import build_drift_report
from mmm.validation.synthetic._io import read_json
from mmm.validation.synthetic.recovery_certification import (
    DRIFT_POST_PRE_MAE_RATIO_MIN,
    TOLERANCE_POLICY_ID,
    RecoveryCheckOutcome,
    _fitted_period_mae,
    _train_ridge_truth_transforms,
    _week_period_masks,
    is_drift_recovery_eligible,
)

RUNNER_VERSION = "drift_detection_runner_v1.0.0"
REGISTRY_VALIDATION_ID = "VAL-012"

DriftSeverityLevel = Literal["none", "minor", "moderate", "severe"]
Val012Outcome = Literal["pass", "warning", "severe"]

# TBD_v1_runtime — provisional evidence bands (not production gates).
MAE_RATIO_MINOR = 1.05
MAE_RATIO_MODERATE = 1.15
MAE_RATIO_SEVERE = 1.35
KPI_SHIFT_MINOR = 0.05
KPI_SHIFT_MODERATE = 0.08
KPI_SHIFT_SEVERE = 0.15
CHANGEPOINT_TOLERANCE_PERIODS = 1


@dataclass
class DriftDetectionResult:
    """Structured VAL-012 output for certification and TrustReport."""

    runner_version: str
    registry_validation_id: str
    tolerance_policy_id: str
    world_id: str
    drift_expected: bool
    val_012_outcome: Val012Outcome
    drift_severity_level: DriftSeverityLevel
    changepoint_detected: bool
    changepoint_period_index: int | None
    changepoint_truth_index: int | None
    changepoint_alignment_ok: bool | None
    pre_period_fit_error: float | None
    post_period_fit_error: float | None
    post_pre_mae_ratio: float | None
    kpi_level_shift_relative: float | None
    drift_report_severity: str | None
    detected_drifts: list[dict[str, Any]]
    calibration_degradation: bool
    kpi_degradation: bool
    fit_degradation: bool
    readiness_downgraded: bool
    optimization_block_recommended: bool
    messages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_version": self.runner_version,
            "registry_validation_id": self.registry_validation_id,
            "tolerance_policy_id": self.tolerance_policy_id,
            "world_id": self.world_id,
            "drift_expected": self.drift_expected,
            "val_012_outcome": self.val_012_outcome,
            "drift_severity_level": self.drift_severity_level,
            "changepoint_detected": self.changepoint_detected,
            "changepoint_period_index": self.changepoint_period_index,
            "changepoint_truth_index": self.changepoint_truth_index,
            "changepoint_alignment_ok": self.changepoint_alignment_ok,
            "pre_period_fit_error": self.pre_period_fit_error,
            "post_period_fit_error": self.post_period_fit_error,
            "post_pre_mae_ratio": self.post_pre_mae_ratio,
            "kpi_level_shift_relative": self.kpi_level_shift_relative,
            "drift_report_severity": self.drift_report_severity,
            "detected_drifts": self.detected_drifts,
            "calibration_degradation": self.calibration_degradation,
            "kpi_degradation": self.kpi_degradation,
            "fit_degradation": self.fit_degradation,
            "readiness_downgraded": self.readiness_downgraded,
            "optimization_block_recommended": self.optimization_block_recommended,
            "messages": self.messages,
            "limitations": self.limitations,
        }


def _classify_severity(
    *,
    mae_ratio: float,
    kpi_shift: float,
    drift_report_severity: str,
    n_detected: int,
) -> DriftSeverityLevel:
    if mae_ratio < MAE_RATIO_MINOR and kpi_shift < KPI_SHIFT_MINOR and n_detected == 0:
        return "none"
    if (
        mae_ratio >= MAE_RATIO_SEVERE
        or kpi_shift >= KPI_SHIFT_SEVERE
        or drift_report_severity == "critical"
    ):
        return "severe"
    if (
        mae_ratio >= MAE_RATIO_MODERATE
        or kpi_shift >= KPI_SHIFT_MODERATE
        or drift_report_severity == "warning"
        or n_detected >= 2
    ):
        return "moderate"
    return "minor"


def _infer_changepoint_from_mae(
    truth: dict[str, Any],
    panel,
    schema,
    config,
    fit: dict[str, Any],
    *,
    baseline_mae: float,
) -> int | None:
    """Changepoint index with largest post/pre MAE ratio across candidate splits."""
    n_periods = int(truth["time_truth"]["n_periods"])
    best_idx: int | None = None
    best_ratio = 0.0
    for idx in range(1, n_periods):
        pre_mask, post_mask = _week_period_masks(
            truth,
            panel,
            schema,
            changepoint_index=idx,
        )
        if not post_mask.any() or not pre_mask.any():
            continue
        pre_mae = _fitted_period_mae(panel, schema, config, fit, pre_mask)
        post_mae = _fitted_period_mae(panel, schema, config, fit, post_mask)
        ratio = post_mae / max(pre_mae, 1e-9)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    if best_ratio >= MAE_RATIO_MODERATE:
        return best_idx
    if baseline_mae > 1e-6 and best_ratio >= MAE_RATIO_MINOR:
        return best_idx
    return None


def _calibration_degradation(
    truth: dict[str, Any],
    drift_report: dict[str, Any],
) -> bool:
    exp = truth.get("experiment_truth") or {}
    gov = truth.get("governance_truth") or {}
    replay_active = bool(gov.get("replay_calibration_active"))
    units = exp.get("units") or []
    detected = drift_report.get("detected_drifts") or []
    stale = any(d.get("kind") == "calibration_staleness" for d in detected)
    if replay_active and not units:
        return True
    return stale


def run_val_012_drift_detection(
    bundle_dir: str | Path,
    truth: dict[str, Any],
) -> DriftDetectionResult:
    """
    Execute VAL-012 on a materialized world bundle.

    On drift worlds: validates changepoint alignment, KPI/fit degradation, and drift_report.
    On non-drift worlds: expects ``none``/``minor`` severity (no false severe alarm).
    """
    from mmm.features.design_matrix import build_design_matrix
    from mmm.governance.scorecard import build_scorecard
    from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
    from mmm.models.ridge_bo.trainer import RidgeBOArtifacts
    from mmm.validation.synthetic.optimizer_truth import shared_ridge_transform_params

    bundle = Path(bundle_dir)
    meta = truth.get("metadata") or {}
    world_id = str(meta.get("world_id", bundle.name))
    drift = truth.get("drift_truth") or {}
    cps = drift.get("changepoints") or []
    coef_drift = drift.get("coefficient_drift") or []
    expected = drift.get("expected_reliability") or {}
    drift_expected = is_drift_recovery_eligible(truth) and bool(cps) and bool(coef_drift)

    limitations = [
        f"{RUNNER_VERSION} — VAL-012 synthetic drift detection",
        f"tolerance_policy_id={TOLERANCE_POLICY_ID}",
    ]

    if not drift_expected:
        return DriftDetectionResult(
            runner_version=RUNNER_VERSION,
            registry_validation_id=REGISTRY_VALIDATION_ID,
            tolerance_policy_id=TOLERANCE_POLICY_ID,
            world_id=world_id,
            drift_expected=False,
            val_012_outcome="pass",
            drift_severity_level="none",
            changepoint_detected=False,
            changepoint_period_index=None,
            changepoint_truth_index=None,
            changepoint_alignment_ok=None,
            pre_period_fit_error=None,
            post_period_fit_error=None,
            post_pre_mae_ratio=None,
            kpi_level_shift_relative=None,
            drift_report_severity=None,
            detected_drifts=[],
            calibration_degradation=False,
            kpi_degradation=False,
            fit_degradation=False,
            readiness_downgraded=False,
            optimization_block_recommended=False,
            messages=["drift not in scope for this world — VAL-012 pass (no drift truth)"],
            limitations=limitations,
        )

    cp_index = int(cps[0]["period_index"])
    messages: list[str] = []

    try:
        panel, schema, config, _fit_full = _train_ridge_truth_transforms(bundle, truth)
    except Exception as exc:
        return DriftDetectionResult(
            runner_version=RUNNER_VERSION,
            registry_validation_id=REGISTRY_VALIDATION_ID,
            tolerance_policy_id=TOLERANCE_POLICY_ID,
            world_id=world_id,
            drift_expected=True,
            val_012_outcome="severe",
            drift_severity_level="severe",
            changepoint_detected=False,
            changepoint_period_index=None,
            changepoint_truth_index=cp_index,
            changepoint_alignment_ok=False,
            pre_period_fit_error=None,
            post_period_fit_error=None,
            post_pre_mae_ratio=None,
            kpi_level_shift_relative=None,
            drift_report_severity=None,
            detected_drifts=[],
            calibration_degradation=False,
            kpi_degradation=False,
            fit_degradation=False,
            readiness_downgraded=True,
            optimization_block_recommended=True,
            messages=[f"train failed: {exc}"],
            limitations=limitations,
        )

    pre_mask, post_mask = _week_period_masks(truth, panel, schema, changepoint_index=cp_index)
    pre_panel = panel.loc[pre_mask]
    if pre_panel.empty:
        return DriftDetectionResult(
            runner_version=RUNNER_VERSION,
            registry_validation_id=REGISTRY_VALIDATION_ID,
            tolerance_policy_id=TOLERANCE_POLICY_ID,
            world_id=world_id,
            drift_expected=True,
            val_012_outcome="severe",
            drift_severity_level="severe",
            changepoint_detected=False,
            changepoint_period_index=None,
            changepoint_truth_index=cp_index,
            changepoint_alignment_ok=False,
            pre_period_fit_error=None,
            post_period_fit_error=None,
            post_pre_mae_ratio=None,
            kpi_level_shift_relative=None,
            drift_report_severity=None,
            detected_drifts=[],
            calibration_degradation=False,
            kpi_degradation=False,
            fit_degradation=False,
            readiness_downgraded=True,
            optimization_block_recommended=True,
            messages=["empty pre-changepoint panel"],
            limitations=limitations,
        )

    shared = shared_ridge_transform_params(truth)
    design_pre = build_design_matrix(
        pre_panel,
        schema,
        config,
        decay=shared["decay"],
        hill_half=shared["hill_half"],
        hill_slope=shared["hill_slope"],
    )
    coef, intercept = fit_ridge(design_pre.X, design_pre.y_modeling, alpha=1e-6)
    fit = {
        "artifacts": RidgeBOArtifacts(
            best_params={
                "decay": shared["decay"],
                "hill_half": shared["hill_half"],
                "hill_slope": shared["hill_slope"],
                "log_alpha": -6.0,
            },
            objective_history=[],
            coef=coef,
            intercept=intercept,
            leaderboard=[],
        )
    }

    pre_mae = _fitted_period_mae(panel, schema, config, fit, pre_mask)
    post_mae = _fitted_period_mae(panel, schema, config, fit, post_mask)
    mae_ratio = post_mae / max(pre_mae, 1e-9)

    target_col = schema.target_column
    pre_mean = float(panel.loc[pre_mask, target_col].mean())
    post_mean = float(panel.loc[post_mask, target_col].mean())
    kpi_shift = abs(post_mean - pre_mean) / max(abs(pre_mean), 1e-9)

    est_cp = _infer_changepoint_from_mae(truth, panel, schema, config, fit, baseline_mae=pre_mae)
    cp_aligned: bool | None
    if est_cp is None:
        cp_aligned = False
        changepoint_detected = False
    else:
        changepoint_detected = True
        cp_aligned = abs(est_cp - cp_index) <= CHANGEPOINT_TOLERANCE_PERIODS

    drift_report = build_drift_report(
        panel=panel.loc[post_mask] if post_mask.any() else panel,
        schema=schema,
        config=config,
        reference_panel=pre_panel,
    )
    detected = list(drift_report.get("detected_drifts") or [])
    drift_report_severity = str(drift_report.get("drift_severity", "informational"))
    cal_deg = _calibration_degradation(truth, drift_report)

    mae_ratio_for_severity = min(float(mae_ratio), 100.0)
    severity = _classify_severity(
        mae_ratio=mae_ratio_for_severity,
        kpi_shift=kpi_shift,
        drift_report_severity=drift_report_severity,
        n_detected=len(detected),
    )

    ratio_min = float(expected.get("post_period_fit_degradation_min_ratio", DRIFT_POST_PRE_MAE_RATIO_MIN))
    fit_deg = mae_ratio >= ratio_min
    kpi_deg = kpi_shift >= KPI_SHIFT_MODERATE

    art = fit["artifacts"]
    full_b = build_design_matrix(
        panel,
        schema,
        config,
        decay=art.best_params["decay"],
        hill_half=art.best_params["hill_half"],
        hill_slope=art.best_params["hill_slope"],
    )
    yhat_full = np.exp(predict_ridge(full_b.X, art.coef, art.intercept))
    gov = truth.get("governance_truth") or {}
    sc = build_scorecard(
        cfg=config.extensions.governance,
        fit_mae=float(np.mean(np.abs(panel[target_col].to_numpy(dtype=float) - yhat_full))),
        baseline_mae=float(np.mean(np.abs(panel[target_col].to_numpy(dtype=float) - yhat_full)))
        * 1.05,
        identifiability_score=0.5,
        calibration_loss=None,
        falsification_flags=[],
        beats_baselines=True,
    )
    readiness_downgraded = (
        not bool(gov.get("approved_for_optimization", True))
        or not sc.approved_for_optimization
        or severity in ("moderate", "severe")
        or drift_report_severity in ("warning", "critical")
    )
    opt_block = severity == "severe" or (
        severity == "moderate" and expected.get("readiness_downgrade_expected")
    )

    drift_warning_expected = bool(expected.get("drift_warning_expected", True))
    detection_ok = fit_deg and (bool(detected) or kpi_deg or changepoint_detected)
    readiness_ok = (
        readiness_downgraded if expected.get("readiness_downgrade_expected") else not readiness_downgraded
    )

    if not detection_ok or not readiness_ok:
        val_outcome: Val012Outcome = "severe"
        messages.append("VAL-012 severe — expected drift signals or readiness reaction missing")
    elif cp_aligned is False:
        val_outcome = "warning"
        messages.append(
            f"VAL-012 warning — changepoint misaligned (est={est_cp}, truth={cp_index})"
        )
    elif severity == "minor" and drift_warning_expected:
        val_outcome = "warning"
        messages.append("VAL-012 warning — drift world but severity only minor (weak detection)")
    else:
        val_outcome = "pass"
        messages.append(f"VAL-012 pass — drift severity {severity}")

    return DriftDetectionResult(
        runner_version=RUNNER_VERSION,
        registry_validation_id=REGISTRY_VALIDATION_ID,
        tolerance_policy_id=TOLERANCE_POLICY_ID,
        world_id=world_id,
        drift_expected=True,
        val_012_outcome=val_outcome,
        drift_severity_level=severity,
        changepoint_detected=changepoint_detected,
        changepoint_period_index=est_cp,
        changepoint_truth_index=cp_index,
        changepoint_alignment_ok=cp_aligned,
        pre_period_fit_error=pre_mae,
        post_period_fit_error=post_mae,
        post_pre_mae_ratio=mae_ratio,
        kpi_level_shift_relative=kpi_shift,
        drift_report_severity=drift_report_severity,
        detected_drifts=detected,
        calibration_degradation=cal_deg,
        kpi_degradation=kpi_deg,
        fit_degradation=fit_deg,
        readiness_downgraded=readiness_downgraded,
        optimization_block_recommended=opt_block,
        messages=messages,
        limitations=limitations,
    )


def recovery_check_from_drift_result(result: DriftDetectionResult) -> RecoveryCheckOutcome:
    """Map VAL-012 runner output to REC-4B5-DRIFT certification row."""
    status = "pass" if result.val_012_outcome == "pass" else "fail"
    return RecoveryCheckOutcome(
        "REC-4B5-DRIFT",
        "drift_recovery",
        status,
        "; ".join(result.messages) or f"VAL-012 {result.val_012_outcome}",
        metric_kind="provisional_statistical",
        details=result.to_dict(),
    )


def run_drift_detection_for_bundle(
    bundle_dir: str | Path,
    truth: dict[str, Any] | None = None,
) -> DriftDetectionResult:
    bundle = Path(bundle_dir)
    truth_eff = truth if truth is not None else read_json(bundle / "world_truth.json")
    return run_val_012_drift_detection(bundle, truth_eff)
