"""Panel quality checks above ``validate_panel`` — persisted on extension artifacts and optional PROD blocks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.extensions import PanelQAConfig
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema

Severity = Literal["info", "warn", "block"]


def _severity_rank(s: Severity) -> int:
    return {"info": 0, "warn": 1, "block": 2}[s]


def run_panel_qa(
    df: pd.DataFrame,
    schema: PanelSchema,
    cfg: PanelQAConfig | None = None,
) -> dict[str, Any]:
    """
    Structural QA on the sorted modeling panel: duplicates, missing (geo, week) keys, spend spikes, all-zero rows.

    Returns a JSON-serializable report with ``max_severity`` in ``info`` / ``warn`` / ``block``.
    """
    cfg = cfg or PanelQAConfig()
    issues: list[dict[str, Any]] = []
    gcol, wcol = schema.geo_column, schema.week_column

    if not cfg.enabled:
        return {
            "panel_qa_version": "mmm_panel_qa_v1",
            "enabled": False,
            "max_severity": "info",
            "issues": [],
            "metrics": {},
        }

    dup = df.groupby([gcol, wcol], observed=False).size()
    n_dup = int((dup > 1).sum())
    if n_dup > 0:
        issues.append(
            {
                "code": "duplicate_geo_week_rows",
                "severity": "block",
                "detail": f"{n_dup} (geo, week) keys have duplicate rows",
            }
        )

    geos = df[gcol].astype(str).unique()
    weeks = df[wcol].unique()
    n_geo, n_week = len(geos), len(weeks)
    expected_pairs = int(n_geo * n_week)
    actual_pairs = int(df[[gcol, wcol]].drop_duplicates().shape[0])
    missing_cells = max(expected_pairs - actual_pairs, 0)
    miss_frac = float(missing_cells / max(expected_pairs, 1))
    metrics: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_geos": int(n_geo),
        "n_distinct_weeks": int(n_week),
        "n_unique_geo_week_pairs": actual_pairs,
        "expected_full_grid_pairs": expected_pairs,
        "missing_geo_week_cell_fraction_vs_full_grid": miss_frac,
    }
    if miss_frac > float(cfg.missing_week_warn_fraction):
        issues.append(
            {
                "code": "sparse_or_missing_geo_week_coverage",
                "severity": "warn",
                "detail": (
                    f"missing_geo_week_cell_fraction {miss_frac:.3f} exceeds warn threshold "
                    f"{cfg.missing_week_warn_fraction} (full rectangular grid may be incomplete)"
                ),
            }
        )

    chans = list(schema.channel_columns)
    if chans:
        mat = df[chans].to_numpy(dtype=float)
        all_zero_frac = float(np.mean(np.all(mat <= 0, axis=1))) if mat.size else 0.0
        metrics["all_channel_zero_row_fraction"] = all_zero_frac
        if all_zero_frac > float(cfg.all_channel_zero_warn_fraction):
            issues.append(
                {
                    "code": "high_all_channel_zero_row_fraction",
                    "severity": "warn",
                    "detail": (
                        f"fraction of rows with all channel spends <= 0 is {all_zero_frac:.3f} "
                        f"(threshold {cfg.all_channel_zero_warn_fraction})"
                    ),
                }
            )

        spike_counts = 0
        total = 0
        z_thr = float(cfg.spend_spike_abs_z)
        for ch in chans:
            x = np.log1p(np.maximum(df[ch].to_numpy(dtype=float), 0.0))
            mu, sig = float(np.mean(x)), float(np.std(x) + 1e-12)
            z = np.abs((x - mu) / sig)
            spike_counts += int(np.sum(z > z_thr))
            total += len(x)
        metrics["spend_spike_cells_abs_z_gt_threshold"] = spike_counts
        metrics["spend_spike_cell_fraction"] = float(spike_counts / max(total, 1))
        if spike_counts > 0 and metrics["spend_spike_cell_fraction"] > 0.02:
            issues.append(
                {
                    "code": "spend_spike_outliers",
                    "severity": "warn",
                    "detail": (
                        f"{spike_counts} channel-week cells exceed |z|>{z_thr} on log1p(spend) "
                        f"({metrics['spend_spike_cell_fraction']:.3%} of cells)"
                    ),
                }
            )

    # Calendar continuity: large gaps between consecutive dated weeks within a geo (bounded scan).
    try:
        wtd = pd.to_datetime(df[wcol], errors="coerce")
        if bool(wtd.notna().any()):
            dfw = df.assign(__wk=wtd)
            gap_warned = False
            for g in list(geos)[:128]:
                sub = dfw[dfw[gcol] == g].sort_values("__wk")["__wk"].drop_duplicates()
                if len(sub) < 6:
                    continue
                gaps = sub.diff().dropna().dt.days.astype(float)
                med = float(np.nanmedian(gaps)) or 7.0
                if float(np.nanmax(gaps)) > max(21.0, med * 5.0):
                    issues.append(
                        {
                            "code": "large_inter_week_gap_in_geo_calendar",
                            "severity": "warn",
                            "detail": (
                                f"geo={g!s}: max gap between consecutive dated weeks is "
                                f"{float(np.nanmax(gaps)):.0f}d vs median {med:.0f}d"
                            ),
                        }
                    )
                    gap_warned = True
                    break
            metrics["calendar_gap_scan_geos"] = int(min(len(geos), 128))
            metrics["calendar_gap_warn_emitted"] = bool(gap_warned)
    except Exception:
        metrics["calendar_gap_scan"] = "skipped_due_to_exception"

    max_sev: Severity = "info"
    for it in issues:
        s = it.get("severity", "warn")
        if s in ("info", "warn", "block") and _severity_rank(s) > _severity_rank(max_sev):  # type: ignore[arg-type]
            max_sev = s  # type: ignore[assignment]
    return {
        "panel_qa_version": "mmm_panel_qa_v1",
        "enabled": True,
        "max_severity": max_sev,
        "issues": issues,
        "metrics": metrics,
    }


def panel_qa_should_fail_prod_training(config: MMMConfig, report: dict[str, Any]) -> bool:
    """PROD fail-closed when panel QA reports ``block`` and extension policy requests a block."""
    if config.run_environment != RunEnvironment.PROD:
        return False
    if report.get("max_severity") != "block":
        return False
    return config.extensions.panel_qa.prod_block_severity == "block"


def assert_panel_qa_allows_training(df_sorted: pd.DataFrame, schema: PanelSchema, config: MMMConfig) -> dict[str, Any]:
    """
    Run panel QA on the modeling-ordered frame (caller must sort with ``sort_panel_for_modeling``).

    Raises ``PermissionError`` in production when ``panel_qa_should_fail_prod_training`` is true.
    """
    report = run_panel_qa(df_sorted, schema, config.extensions.panel_qa)
    if panel_qa_should_fail_prod_training(config, report):
        raise PermissionError(
            "Production training blocked by panel QA (extensions.panel_qa.prod_block_severity=block and "
            f"max_severity=block). Issues: {report.get('issues')!r}"
        )
    return report
