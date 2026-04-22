"""E11: gate budget optimization on diagnostics — production is fail-closed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mmm.config.extensions import OptimizationGateConfig
from mmm.config.schema import RunEnvironment


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    audit: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": self.reasons, "audit": self.audit}


class OptimizationSafetyGate:
    def __init__(self, cfg: OptimizationGateConfig) -> None:
        self.cfg = cfg

    def check(
        self,
        *,
        governance: dict[str, Any],
        response_diag: dict[str, Any] | None,
        identifiability_score: float,
        run_environment: RunEnvironment | None = None,
        extension_report_present: bool = False,
    ) -> GateResult:
        audit: list[str] = []
        if run_environment == RunEnvironment.PROD:
            if not self.cfg.enabled:
                return GateResult(
                    False,
                    ["prod_requires_optimization_gates_enabled"],
                    audit=["unsafe_disabled_gates_rejected_in_prod"],
                )
            if not extension_report_present:
                return GateResult(
                    False,
                    ["prod_requires_extension_report_for_optimization"],
                    audit=["missing_extension_report"],
                )
            if not governance:
                return GateResult(
                    False,
                    ["prod_requires_governance_section_in_extension_report"],
                    audit=["empty_governance"],
                )

        if not self.cfg.enabled:
            audit.append("optimization_gates_disabled_non_prod")
            return GateResult(True, ["gates_disabled"], audit=audit)

        reasons: list[str] = []
        if self.cfg.require_governance_optimization_flag and not governance.get("approved_for_optimization"):
            skip_governance_check = (
                (run_environment is None or run_environment != RunEnvironment.PROD)
                and self.cfg.allow_missing_extension_report
                and not governance
                and not extension_report_present
            )
            if not skip_governance_check:
                reasons.append("governance: not approved_for_optimization")
        if self.cfg.require_response_curve_safe and response_diag and not response_diag.get("safe_for_optimization"):
            reasons.append("response_curve_unsafe")
        if identifiability_score > self.cfg.max_identifiability_risk:
            reasons.append(f"identifiability_risk {identifiability_score:.3f} > {self.cfg.max_identifiability_risk}")
        return GateResult(allowed=len(reasons) == 0, reasons=reasons or ["ok"], audit=audit)
