"""ROI / mROAS summaries from extension curve bundles (Sprint 6 / reporting)."""

from __future__ import annotations

from typing import Any

from mmm.contracts.quantity_models import ROIApproxQuantityResult
from mmm.contracts.runtime_validation import validate_proxy_reporting_payload
from mmm.economics.canonical import ECONOMICS_CONTRACT_VERSION


def curve_bundles_to_roi_summary(curve_bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-channel spend grid midpoints and level-space mROAS proxy at mid grid (reporting only)."""
    rows: list[dict[str, Any]] = []
    for b in curve_bundles:
        ch = b.get("channel", "?")
        grid = b.get("spend_grid") or []
        mroas_cons = b.get("mroas_level_consistent")
        mroas_lin = b.get("mroas_level_proxy")
        if not isinstance(grid, list) or len(grid) < 1:
            continue
        mid = float(grid[len(grid) // 2])
        m_cons_mid: float | None = None
        m_lin_mid: float | None = None
        if isinstance(mroas_cons, list) and len(mroas_cons) == len(grid):
            m_cons_mid = float(mroas_cons[len(mroas_cons) // 2])
        if isinstance(mroas_lin, list) and len(mroas_lin) == len(grid):
            m_lin_mid = float(mroas_lin[len(mroas_lin) // 2])
        tr = b.get("typed_roi_quantity") if isinstance(b.get("typed_roi_quantity"), dict) else {}
        econ = tr.get("economics_notes") if isinstance(tr.get("economics_notes"), dict) else {}
        yls = econ.get("y_level_scale")
        if yls is None and isinstance(b.get("roi_bridge"), dict):
            yls = b["roi_bridge"].get("y_level_scale")
        rows.append(
            {
                "channel": ch,
                "spend_mid_grid": mid,
                "mroas_level_consistent_mid_grid": m_cons_mid,
                "mroas_level_proxy_mid_grid": m_lin_mid,
                "mroas_preferred_mid_grid": m_cons_mid if m_cons_mid is not None else m_lin_mid,
                "y_level_scale": yls,
            }
        )
    economics_notes = {
        "channels": rows,
        "source": "curve_bundles_to_roi_summary",
        "surface": "curve_diagnostic_reporting",
        "computation_mode": "approximate",
        "mroas_values_are_local_proxies_not_exact_business_truth": True,
        "artifact_tier": "diagnostic",
        "is_proxy_metric": True,
        "not_exact_business_value": True,
        "economics_contract_version": ECONOMICS_CONTRACT_VERSION,
        "kpi_column": None,
        "kpi_unit_semantics": "not_declared_curve_bundle_roi_summary",
        "baseline_type": "not_applicable_curve_mid_grid",
    }
    qty = ROIApproxQuantityResult(
        mroas_level_proxy=[],
        economics_notes=economics_notes,
        validity_diagnostics={"reporting": "curve_bundles_to_roi_summary"},
    )
    summary = qty.section_dict()
    validate_proxy_reporting_payload(
        {
            "channels": rows,
            "source": "curve_bundles_to_roi_summary",
            "surface": "curve_diagnostic_reporting",
            "computation_mode": "approximate",
            "mroas_values_are_local_proxies_not_exact_business_truth": True,
            "artifact_tier": "diagnostic",
            "decision_safe": False,
            "approximate": True,
            "not_for_budgeting": True,
            "is_proxy_metric": True,
            "not_exact_business_value": True,
            "economics_contract_version": ECONOMICS_CONTRACT_VERSION,
            "kpi_column": None,
            "kpi_unit_semantics": "not_declared_curve_bundle_roi_summary",
            "baseline_type": "not_applicable_curve_mid_grid",
        }
    )
    return summary
