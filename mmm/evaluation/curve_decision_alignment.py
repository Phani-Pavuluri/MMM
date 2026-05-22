"""Diagnostic: response curves vs full-panel Δμ (curves explain; simulation decides)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import ridge_context_from_fit
from mmm.planning.decision_simulate import simulate

WarningLevel = Literal["none", "info", "warning", "critical"]

CURVE_DECISION_DOC = "Curves explain; full-panel simulation decides."


def _curve_increment_at_spend(curve_bundle: dict[str, Any], baseline_s: float, plan_s: float) -> float:
    grid = np.asarray(curve_bundle.get("spend_grid") or curve_bundle.get("spend_grid_weekly"), dtype=float)
    resp = np.asarray(
        curve_bundle.get("response_on_modeling_scale")
        or curve_bundle.get("response")
        or curve_bundle.get("response_on_target_scale"),
        dtype=float,
    )
    if grid.size < 2 or resp.size != grid.size:
        return float("nan")
    return float(np.interp(plan_s, grid, resp) - np.interp(baseline_s, grid, resp))


def evaluate_curve_decision_alignment(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    curve_bundles: list[dict[str, Any]],
    channel: str | None = None,
    spend_delta_frac: float = 0.05,
) -> dict[str, Any]:
    """
    Compare univariate curve increment to full-panel ``simulate()`` Δμ for one channel bump.

    Governance-only diagnostic — does not change optimizer or planning logic.
    """
    ch = channel or (schema.channel_columns[0] if schema.channel_columns else "")
    if not ch or not curve_bundles:
        return {
            "policy_note": CURVE_DECISION_DOC,
            "expected_alignment": False,
            "observed_difference": None,
            "warning_level": "info",
            "reason": "missing_channel_or_curves",
        }

    ctx = ridge_context_from_fit(panel, schema, config, fit_out)
    baseline_spend = float(panel[ch].mean())
    plan_spend = baseline_spend * (1.0 + float(spend_delta_frac))
    cb = next((c for c in curve_bundles if c.get("channel") == ch), curve_bundles[0])
    curve_delta = _curve_increment_at_spend(cb, baseline_spend, plan_spend)

    base = bau_baseline_from_panel(panel, schema)
    plan = dict(base.spend_by_channel)
    plan[ch] = plan_spend
    sim = simulate(plan, ctx, baseline_plan=base)
    panel_delta = float(sim.delta_mu)

    if not np.isfinite(curve_delta) or not np.isfinite(panel_delta):
        rel = float("inf")
    else:
        denom = max(abs(panel_delta), 1e-9)
        rel = float(abs(curve_delta - panel_delta) / denom)

    decay = float(ctx.best_params.get("decay", config.transforms.adstock_params.get("decay", 0.5)))
    n_ch = len(schema.channel_columns)
    n_geo = int(panel[schema.geo_column].nunique())

    # Heuristic expectations (diagnostic labels only).
    simple = decay < 0.05 and n_ch == 1 and n_geo == 1
    expected_alignment = simple
    if simple and rel < 0.25:
        warning_level: WarningLevel = "none"
    elif simple:
        warning_level = "warning"
    elif rel < 0.5:
        warning_level = "info"
    else:
        warning_level = "warning" if rel < 2.0 else "critical"

    return {
        "policy_note": CURVE_DECISION_DOC,
        "channel": ch,
        "baseline_spend": baseline_spend,
        "plan_spend": plan_spend,
        "curve_delta_response": curve_delta,
        "full_panel_delta_mu": panel_delta,
        "relative_abs_difference": rel,
        "expected_alignment": expected_alignment,
        "observed_difference": float(curve_delta - panel_delta),
        "warning_level": warning_level,
        "context": {
            "adstock_decay": decay,
            "n_channels": n_ch,
            "n_geos": n_geo,
            "spend_delta_frac": spend_delta_frac,
        },
    }


DivergenceCategory = Literal["expected_divergence", "suspicious_divergence", "aligned"]


def categorize_alignment(report: dict[str, Any]) -> DivergenceCategory:
    """Label diagnostic divergence — not a gate."""
    if report.get("expected_alignment") and report.get("warning_level") in ("none", "info"):
        return "aligned"
    rel = report.get("relative_abs_difference")
    if rel is None or (isinstance(rel, float) and not np.isfinite(rel)):
        return "suspicious_divergence"
    if report.get("expected_alignment"):
        return "expected_divergence"
    if report.get("warning_level") == "critical":
        return "suspicious_divergence"
    return "expected_divergence"


def evaluate_alignment_scenarios(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    curve_bundles: list[dict[str, Any]],
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Run multiple curve-vs-Δμ probes (diagnostic matrix).

    Each scenario may set ``channel``, ``spend_delta_frac``, and optional ``label``.
    """
    default = [{"label": "default", "spend_delta_frac": 0.05}]
    rows: list[dict[str, Any]] = []
    for spec in scenarios or default:
        rep = evaluate_curve_decision_alignment(
            panel=panel,
            schema=schema,
            config=config,
            fit_out=fit_out,
            curve_bundles=curve_bundles,
            channel=spec.get("channel"),
            spend_delta_frac=float(spec.get("spend_delta_frac", 0.05)),
        )
        cat = categorize_alignment(rep)
        rows.append(
            {
                "label": spec.get("label") or spec.get("channel") or "default",
                "divergence_category": cat,
                "report": rep,
            }
        )
    suspicious = sum(1 for r in rows if r["divergence_category"] == "suspicious_divergence")
    return {
        "policy_note": CURVE_DECISION_DOC,
        "diagnostic_only": True,
        "scenarios": rows,
        "summary": {
            "n_scenarios": len(rows),
            "n_suspicious": suspicious,
            "n_expected_divergence": sum(
                1 for r in rows if r["divergence_category"] == "expected_divergence"
            ),
        },
    }
