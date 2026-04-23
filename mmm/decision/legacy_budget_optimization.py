"""
NON-CANONICAL legacy budget optimization (curve interpolation or placeholder marginal ROI).

**Not** governed by ``optimize_budget_decision`` / full-panel Δμ policy. Non-prod only; blocked in
prod at callers. Use ``mmm.decision.service.optimize_budget_decision`` for decision-grade paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from mmm.config.schema import MMMConfig
from mmm.decision.extension_gate import optimization_gate_result
from mmm.diagnostics.curve_optimizer import optimize_budget_from_curve_bundles
from mmm.economics.canonical import economics_contract_for_curve_bundles
from mmm.governance.policy import PolicyError
from mmm.optimization.budget.curve_bundles_io import gather_curve_bundles_from_dict, gather_curve_bundles_from_path
from mmm.optimization.budget.optimizer import BudgetOptimizer


def run_legacy_diagnostic_optimize_budget(
    *,
    cfg: MMMConfig,
    er_data: dict,
    curve_bundle: Path | None,
    allow_unsafe_decision_apis: bool,
) -> dict[str, Any]:
    """
    Legacy curve / placeholder optimizer. Caller must enforce prod forbid + unsafe API flags.

    Returns a JSON-serializable dict tagged ``legacy_diagnostic_curve_optimizer``.
    """
    if os.environ.get("MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET", "").strip() != "1":
        raise PolicyError(
            "Legacy diagnostic budget optimization is quarantined: set environment variable "
            "MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET=1 to acknowledge non-canonical, non-decision-safe behavior."
        )
    if not cfg.allow_unsafe_decision_apis or not allow_unsafe_decision_apis:
        raise PolicyError("legacy diagnostic optimize requires allow_unsafe_decision_apis on config and caller")

    gr = optimization_gate_result(cfg, er_data, extension_report_present=True)
    if not gr.allowed:
        raise PolicyError("optimization_gate_blocked: " + "; ".join(gr.reasons))

    names = list(cfg.data.channel_columns)
    n = len(names)
    total_budget = float(cfg.budget.total_budget or n * 1e5)
    channel_min = np.array([float(cfg.budget.channel_min.get(c, 0.0)) for c in names], dtype=float)
    channel_max = np.array([float(cfg.budget.channel_max.get(c, 1e6)) for c in names], dtype=float)
    current = np.ones(n, dtype=float) * 1e5

    gathered = None
    if curve_bundle is not None and curve_bundle.exists():
        gathered = gather_curve_bundles_from_path(curve_bundle)
    else:
        gathered = gather_curve_bundles_from_dict(er_data)

    if gathered is not None:
        g_names, bundles = gathered
        bmap = {ch: b for ch, b in zip(g_names, bundles, strict=True)}
        missing = [c for c in names if c not in bmap]
        if missing:
            raise ValueError(f"Curve data missing channels: {missing}")
        ordered_bundles = [bmap[c] for c in names]
        ec = economics_contract_for_curve_bundles(ordered_bundles, strict=False)
        res = optimize_budget_from_curve_bundles(
            names,
            ordered_bundles,
            config=cfg,
            current_spend=current,
            total_budget=total_budget,
            channel_min=channel_min,
            channel_max=channel_max,
            economics_contract=ec,
        )
        return {
            "legacy_diagnostic_curve_optimizer": True,
            "non_canonical": True,
            "mode": "curve_bundles",
            "optimization_gate": gr.to_json(),
            "result": str(res),
        }

    opt = BudgetOptimizer(
        channel_names=names,
        marginal_roi=np.linspace(1.0, 2.0, n),
        channel_min=channel_min,
        channel_max=channel_max,
    )
    res = opt.optimize(current, total_budget=total_budget)
    return {
        "legacy_diagnostic_curve_optimizer": True,
        "non_canonical": True,
        "mode": "marginal_roi_placeholder",
        "optimization_gate": gr.to_json(),
        "result": str(res),
    }
