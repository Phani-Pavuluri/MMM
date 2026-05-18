"""Planning policy guardrails for sensitive control columns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from mmm.config.extensions import PlanningPolicyConfig
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.planning.assumptions import ControlsAssumption

Severity = Literal["info", "warning", "block"]

_NAME_HEURISTIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "promo": re.compile(r"promo|promotion|discount|coupon|offer", re.I),
    "pricing": re.compile(r"price|pricing|cpi|inflation|cost_index", re.I),
    "macro": re.compile(r"macro|gdp|unemployment|consumer_sentiment|recession|inflation_rate", re.I),
    "seasonality": re.compile(r"season|holiday|christmas|easter|black_friday|fourier|sine|cos_", re.I),
}


@dataclass
class ControlScenarioPolicyResult:
    severity: Severity
    messages: list[str] = field(default_factory=list)
    sensitive_columns_matched: list[str] = field(default_factory=list)
    controls_assumption: str = "observed"

    def to_json(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "messages": self.messages,
            "sensitive_columns_matched": self.sensitive_columns_matched,
            "controls_assumption": self.controls_assumption,
        }


def _configured_sensitive_columns(cfg: MMMConfig) -> dict[str, set[str]]:
    pol: PlanningPolicyConfig = cfg.extensions.planning_policy
    out: dict[str, set[str]] = {
        "promo": set(pol.promo_columns),
        "pricing": set(pol.pricing_columns),
        "macro": set(pol.macro_columns),
        "seasonality": set(pol.seasonality_columns),
    }
    ctrl = list(cfg.data.control_columns or [])
    if pol.name_heuristic_warnings:
        for col in ctrl:
            for kind, pat in _NAME_HEURISTIC_PATTERNS.items():
                if pat.search(col):
                    out[kind].add(col)
    return out


def sensitive_control_columns(cfg: MMMConfig) -> list[str]:
    buckets = _configured_sensitive_columns(cfg)
    return sorted(set().union(*buckets.values()))


def evaluate_control_scenario_policy(
    cfg: MMMConfig,
    *,
    controls_assumption: ControlsAssumption,
    control_columns: list[str] | tuple[str, ...] | None = None,
) -> ControlScenarioPolicyResult:
    """
    Warn or block when sensitive controls use observed historical values without explicit scenario.
    """
    cols = list(control_columns if control_columns is not None else (cfg.data.control_columns or []))
    if not cols:
        return ControlScenarioPolicyResult(severity="info", controls_assumption=controls_assumption)
    sensitive = sensitive_control_columns(cfg)
    matched = [c for c in cols if c in sensitive]
    if not matched:
        return ControlScenarioPolicyResult(severity="info", controls_assumption=controls_assumption)

    pol: PlanningPolicyConfig = cfg.extensions.planning_policy
    msgs: list[str] = []
    sev: Severity = "info"
    if controls_assumption == "observed":
        msgs.append(
            f"Sensitive control columns {matched} use observed historical panel values; "
            "future promo/pricing/macro conditions are not simulated unless a PlanningScenario is supplied."
        )
        sev = "warning"
        if (
            cfg.run_environment == RunEnvironment.PROD
            and pol.strict_prod_requires_explicit_control_scenario
        ):
            sev = "block"
            msgs.append(
                "strict_prod_requires_explicit_control_scenario=True: supply PlanningScenario with control overlays."
            )
    return ControlScenarioPolicyResult(
        severity=sev,
        messages=msgs,
        sensitive_columns_matched=matched,
        controls_assumption=controls_assumption,
    )
