"""Optimization safety gate over extension JSON (shared by CLI, service, and legacy paths)."""

from __future__ import annotations

from mmm.config.schema import MMMConfig
from mmm.optimization.safety_gate import GateResult, OptimizationSafetyGate


def optimization_gate_result(
    cfg: MMMConfig,
    er_data: dict | None,
    *,
    extension_report_present: bool,
) -> GateResult:
    """Run the same extension-aware gate as ``mmm decide optimize-budget``."""
    gov = er_data.get("governance", {}) if isinstance(er_data, dict) else {}
    resp = er_data.get("response_diagnostics") if isinstance(er_data, dict) else None
    idv = er_data.get("identifiability", {}) if isinstance(er_data, dict) else {}
    ident = float(idv.get("identifiability_score", 0.0)) if isinstance(idv, dict) else 0.0
    pq_er = er_data.get("panel_qa") if isinstance(er_data, dict) else None
    gate = OptimizationSafetyGate(cfg.extensions.optimization_gates)
    return gate.check(
        governance=gov if isinstance(gov, dict) else {},
        response_diag=resp,
        identifiability_score=ident,
        run_environment=cfg.run_environment,
        extension_report_present=extension_report_present,
        panel_qa=pq_er if isinstance(pq_er, dict) else None,
    )
