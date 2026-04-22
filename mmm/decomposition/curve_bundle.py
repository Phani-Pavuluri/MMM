"""Serialized curve artifacts for optimizers (Sprint 5 / 6)."""

from __future__ import annotations

from typing import Any

from mmm.decomposition.curve_stress import CurveStressReport
from mmm.decomposition.curves import ResponseCurve
from mmm.decomposition.response_diagnostics import ResponseDiagnostics
from mmm.decomposition.roi_bridge import attach_level_roi_to_curve_artifact
from mmm.economics.canonical import attach_contract_to_curve_artifact


def curve_bundle_to_artifact(
    *,
    channel: str,
    curve: ResponseCurve,
    diagnostics: ResponseDiagnostics,
    stress: CurveStressReport,
    horizon_weeks: int,
    model_form: str,
    economics_contract: dict[str, Any],
    aggregation: str = "steady_state_last_week",
    y_level_scale: float | None = None,
    target_column: str = "revenue",
    anchor_spend: float | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "channel": channel,
        "horizon_weeks": horizon_weeks,
        "aggregation": aggregation,
        "model_form": model_form,
        "spend_grid": curve.spend_grid.tolist(),
        "response_on_modeling_scale": curve.response.tolist(),
        "marginal_roi_modeling_scale": curve.marginal_roi.tolist(),
        "diagnostics": diagnostics.to_json(),
        "stress": stress.to_json(),
        "marginal_roi_definition": (
            "marginal_roi_modeling_scale: np.gradient(response, spend_grid) on β·x(spend) "
            "(log-mean contribution). Use mroas_level_proxy when y_level_scale is set."
        ),
    }
    if y_level_scale is not None and y_level_scale > 0:
        art = attach_level_roi_to_curve_artifact(
            base,
            y_level_scale=float(y_level_scale),
            target_column=target_column,
            anchor_spend=anchor_spend,
        )
        return attach_contract_to_curve_artifact(art, economics_contract)
    base["roi_bridge"] = {
        "model_form": model_form,
        "y_level_scale": None,
        "notes": "Pass y_level_scale to curve_bundle_to_artifact for level-space mROAS proxy.",
    }
    base["roi_bridge"]["target_kpi_column"] = target_column
    return attach_contract_to_curve_artifact(base, economics_contract)
