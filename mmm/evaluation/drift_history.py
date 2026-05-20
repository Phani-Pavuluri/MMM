"""Load prior-run references for historical drift comparison (no auto-retrain)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema


def _distribution_summary(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan")}
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p50": float(s.quantile(0.5)),
    }


def panel_distribution_snapshot(panel: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    """Summarize media and control distributions for drift comparison."""
    media: dict[str, Any] = {}
    for ch in schema.channel_columns:
        if ch in panel.columns:
            media[ch] = _distribution_summary(panel[ch])
    controls: dict[str, Any] = {}
    for col in schema.control_columns:
        if col in panel.columns:
            controls[col] = _distribution_summary(panel[col])
    target = {}
    if schema.target_column in panel.columns:
        target = _distribution_summary(panel[schema.target_column])
    return {"media": media, "controls": controls, "target": target}


def load_historical_reference(run_dir: str | Path) -> dict[str, Any] | None:
    """
    Load a prior accepted run's fingerprint and summary stats from ``extension_report.json``.

    Returns ``None`` when the path is missing or unreadable.
    """
    root = Path(run_dir)
    report_path = root / "extension_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fp = report.get("data_fingerprint") or report.get("panel_fingerprint")
    cal = report.get("calibration_summary") or {}
    fit_summary: dict[str, Any] = {}
    ident = report.get("identifiability") or {}
    if isinstance(ident, dict) and ident.get("mean_cv_score") is not None:
        fit_summary["mean_cv_score"] = ident.get("mean_cv_score")
    post = report.get("post_fit_validation") or {}
    if isinstance(post, dict) and post.get("in_sample_rmse") is not None:
        fit_summary["in_sample_rmse"] = post.get("in_sample_rmse")
    drift = report.get("drift_report") or {}
    dist = {}
    if isinstance(drift, dict):
        dist = drift.get("current_panel_distribution_snapshot") or {}
    if not dist:
        raw = report.get("panel_distribution_snapshot")
        dist = raw if isinstance(raw, dict) else {}
    return {
        "source_run_dir": str(root.resolve()),
        "panel_fingerprint": fp if isinstance(fp, dict) else {},
        "calibration_summary": cal if isinstance(cal, dict) else {},
        "model_outputs": fit_summary,
        "panel_distribution_snapshot": dist,
    }


def compare_model_outputs(
    current: dict[str, Any] | None,
    historical: dict[str, Any] | None,
    *,
    rmse_rel_threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Compare scalar model-output summaries between runs."""
    if not current or not historical:
        return []
    drifts: list[dict[str, Any]] = []
    for key in ("in_sample_rmse", "mean_cv_score"):
        c_val = current.get(key)
        h_val = historical.get(key)
        if c_val is None or h_val is None:
            continue
        c_f, h_f = float(c_val), float(h_val)
        denom = max(abs(h_f), 1e-9)
        rel = abs(c_f - h_f) / denom
        if rel > rmse_rel_threshold:
            drifts.append(
                {
                    "kind": "model_output_drift",
                    "metric": key,
                    "relative_shift": rel,
                    "historical": h_f,
                    "current": c_f,
                }
            )
    return drifts


def compare_distribution_snapshots(
    current: dict[str, Any],
    historical: dict[str, Any],
    *,
    rel_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Compare stored media/control mean summaries."""
    drifts: list[dict[str, Any]] = []
    for section, kind in (("media", "media_distribution_drift"), ("controls", "control_distribution_drift")):
        cur_sec = current.get(section) or {}
        hist_sec = historical.get(section) or {}
        if not isinstance(cur_sec, dict) or not isinstance(hist_sec, dict):
            continue
        for col, cur_stats in cur_sec.items():
            if col not in hist_sec or not isinstance(cur_stats, dict):
                continue
            h_mean = float(hist_sec[col].get("mean", float("nan")))
            c_mean = float(cur_stats.get("mean", float("nan")))
            if not np.isfinite(h_mean) or not np.isfinite(c_mean):
                continue
            rel = abs(c_mean - h_mean) / max(abs(h_mean), 1e-9)
            if rel > rel_threshold:
                drifts.append(
                    {
                        "kind": kind,
                        "column": col,
                        "relative_mean_shift": rel,
                        "historical_mean": h_mean,
                        "current_mean": c_mean,
                    }
                )
    return drifts
