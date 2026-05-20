"""Operational drift diagnostics from panel + artifact fingerprints (no auto-retrain)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.evaluation.drift_history import (
    compare_distribution_snapshots,
    compare_model_outputs,
    panel_distribution_snapshot,
)

Severity = Literal["none", "low", "medium", "high"]
DriftSeverity = Literal["informational", "warning", "critical"]
RecommendedAction = Literal["monitor", "retrain_recommended", "experiment_refresh_recommended"]
DriftTrend = Literal["stable", "gradual_drift", "sudden_drift"]

_SEVERITY_TO_DRIFT: dict[Severity, DriftSeverity] = {
    "none": "informational",
    "low": "informational",
    "medium": "warning",
    "high": "critical",
}


def _channel_spend_shift(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    schema: PanelSchema,
) -> list[dict[str, Any]]:
    drifts: list[dict[str, Any]] = []
    for ch in schema.channel_columns:
        c_mean = float(current[ch].mean())
        r_mean = float(reference[ch].mean())
        denom = max(abs(r_mean), 1e-9)
        rel = abs(c_mean - r_mean) / denom
        if rel > 0.25:
            drifts.append(
                {
                    "kind": "channel_spend_distribution",
                    "channel": ch,
                    "relative_mean_shift": rel,
                    "reference_mean": r_mean,
                    "current_mean": c_mean,
                }
            )
    return drifts


def _control_shift(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    schema: PanelSchema,
) -> list[dict[str, Any]]:
    drifts: list[dict[str, Any]] = []
    for col in schema.control_columns:
        if col not in current.columns or col not in reference.columns:
            continue
        c_mean = float(current[col].mean())
        r_mean = float(reference[col].mean())
        denom = max(abs(r_mean), 1e-9)
        rel = abs(c_mean - r_mean) / denom
        if rel > 0.2:
            drifts.append(
                {
                    "kind": "control_drift",
                    "column": col,
                    "relative_mean_shift": rel,
                }
            )
    return drifts


def infer_drift_trend(detected: list[dict[str, Any]]) -> DriftTrend:
    """Classify drift pace for historical monitoring (diagnostic only)."""
    if not detected:
        return "stable"
    if any(d.get("kind") == "panel_fingerprint_change" for d in detected):
        return "sudden_drift"
    dist_kinds = {
        "channel_spend_distribution",
        "control_drift",
        "media_distribution_drift",
        "control_distribution_drift",
        "model_output_drift",
    }
    if sum(1 for d in detected if d.get("kind") in dist_kinds) >= 2:
        return "gradual_drift"
    if detected:
        return "gradual_drift"
    return "stable"


def build_drift_report(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    reference_panel: pd.DataFrame | None = None,
    reference_fingerprint: dict[str, Any] | None = None,
    historical_reference: dict[str, Any] | None = None,
    current_model_outputs: dict[str, Any] | None = None,
    residuals: np.ndarray | None = None,
    reference_residuals: np.ndarray | None = None,
    calibration_summary: dict[str, Any] | None = None,
    seed_resolution: dict[str, Any] | None = None,
    current_run_id: str | None = None,
) -> dict[str, Any]:
    """
    Compare current panel to a reference fingerprint/panel and emit ``drift_report``.

    Does not trigger retraining or external monitoring.
    """
    current_fp = fingerprint_panel(panel, schema, config=config, seed_resolution=seed_resolution)
    current_dist = panel_distribution_snapshot(panel, schema)
    detected: list[dict[str, Any]] = []

    same_run_excluded = False
    if historical_reference is not None:
        hist_dir = historical_reference.get("source_run_dir")
        if (current_run_id and historical_reference.get("registry_run_id") == current_run_id) or (
            hist_dir and current_run_id and str(current_run_id) in str(hist_dir)
        ):
            historical_reference = None
            same_run_excluded = True

    if historical_reference is not None:
        hist_fp = historical_reference.get("panel_fingerprint") or {}
        if isinstance(hist_fp, dict) and hist_fp:
            reference_fingerprint = reference_fingerprint or hist_fp
        hist_dist = historical_reference.get("panel_distribution_snapshot") or {}
        if isinstance(hist_dist, dict) and hist_dist:
            detected.extend(compare_distribution_snapshots(current_dist, hist_dist))
        hist_model = historical_reference.get("model_outputs") or {}
        if isinstance(hist_model, dict) and current_model_outputs:
            detected.extend(compare_model_outputs(current_model_outputs, hist_model))

    if reference_fingerprint is not None:
        ref_combined = reference_fingerprint.get("sha256_combined") or reference_fingerprint.get(
            "sha256_panel_keycols_sorted_csv"
        )
        cur_combined = current_fp.get("sha256_combined") or current_fp.get("sha256_panel_keycols_sorted_csv")
        if ref_combined and cur_combined and ref_combined != cur_combined:
            detected.append(
                {
                    "kind": "panel_fingerprint_change",
                    "reference_sha256": ref_combined,
                    "current_sha256": cur_combined,
                }
            )

    if reference_panel is not None and len(reference_panel):
        detected.extend(_channel_spend_shift(panel, reference_panel, schema))
        detected.extend(_control_shift(panel, reference_panel, schema))

    if residuals is not None and reference_residuals is not None:
        r_cur = np.asarray(residuals, dtype=float).ravel()
        r_ref = np.asarray(reference_residuals, dtype=float).ravel()
        if r_cur.size and r_ref.size:
            std_ref = float(np.std(r_ref))
            std_cur = float(np.std(r_cur))
            if std_ref > 1e-12 and abs(std_cur - std_ref) / std_ref > 0.35:
                detected.append(
                    {
                        "kind": "residual_drift",
                        "reference_residual_std": std_ref,
                        "current_residual_std": std_cur,
                    }
                )

    cal = calibration_summary or {}
    if cal.get("replay_calibration_active") and cal.get("replay_loss") is None:
        detected.append({"kind": "calibration_staleness", "detail": "replay_active_without_loss"})

    n = len(detected)
    if n == 0:
        severity: Severity = "none"
        action: RecommendedAction = "monitor"
    elif n <= 2:
        severity = "low"
        action = "monitor"
    elif any(d.get("kind") == "panel_fingerprint_change" for d in detected):
        severity = "high"
        action = "retrain_recommended"
    else:
        severity = "medium"
        action = "experiment_refresh_recommended" if any(
            d.get("kind") == "calibration_staleness" for d in detected
        ) else "retrain_recommended"

    trend = infer_drift_trend(detected) if historical_reference is not None else "stable"

    return {
        "severity": severity,
        "drift_severity": _SEVERITY_TO_DRIFT[severity],
        "drift_trend": trend,
        "detected_drifts": detected,
        "recommended_action": action,
        "same_run_comparison_excluded": same_run_excluded,
        "current_panel_fingerprint": current_fp,
        "current_panel_distribution_snapshot": current_dist,
        "historical_reference": (
            {
                "loaded": True,
                "source_run_dir": historical_reference.get("source_run_dir"),
            }
            if historical_reference
            else {"loaded": False}
        ),
        "policy_note": "Drift is diagnostic only; no automatic retraining or experiment execution.",
    }
