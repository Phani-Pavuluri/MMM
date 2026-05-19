"""Operational drift diagnostics from panel + artifact fingerprints (no auto-retrain)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema

Severity = Literal["none", "low", "medium", "high"]
RecommendedAction = Literal["monitor", "retrain_recommended", "experiment_refresh_recommended"]


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


def build_drift_report(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    reference_panel: pd.DataFrame | None = None,
    reference_fingerprint: dict[str, Any] | None = None,
    residuals: np.ndarray | None = None,
    reference_residuals: np.ndarray | None = None,
    calibration_summary: dict[str, Any] | None = None,
    seed_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare current panel to a reference fingerprint/panel and emit ``drift_report``.

    Does not trigger retraining or external monitoring.
    """
    current_fp = fingerprint_panel(panel, schema, config=config, seed_resolution=seed_resolution)
    detected: list[dict[str, Any]] = []

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

    return {
        "severity": severity,
        "detected_drifts": detected,
        "recommended_action": action,
        "current_panel_fingerprint": current_fp,
    }
