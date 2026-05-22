"""Canonical ``decision_safe`` contract for simulate, enrichment, and prod gates."""

from __future__ import annotations

from typing import Any


def scenario_decision_safe_from_simulation(sim_js: dict[str, Any]) -> bool:
    """
    Scenario / baseline safety from ``decision_simulate`` (before governance enrichment).

    Locked or non-BAU baselines must remain unsafe regardless of extension gates.
    """
    if sim_js.get("baseline_suitable_for_decisioning") is False:
        return False
    bt = str(sim_js.get("baseline_type") or sim_js.get("baseline_definition") or "").lower()
    if "locked" in bt or bt in ("locked_plan", "historical_average", "zero_spend"):
        return False
    v = sim_js.get("decision_safe")
    if isinstance(v, bool):
        return v
    return False


def canonical_decision_safe(
    *,
    scenario_safe: bool,
    governance_gate_allowed: bool,
    optimizer_internal_safe: bool | None = None,
    optimizer_success: bool | None = None,
    gates_enabled: bool | None = None,
) -> bool:
    """
    Single prod-facing safety bit: scenario suitable AND governance AND (optional) optimizer checks.
    """
    if not scenario_safe:
        return False
    if not governance_gate_allowed:
        return False
    if gates_enabled is False:
        return False
    if optimizer_success is not None and not optimizer_success:
        return False
    return not (optimizer_internal_safe is not None and not optimizer_internal_safe)


def compute_decision_safe(
    *,
    governance_gate_allowed: bool,
    scenario_suitable_for_decisioning: bool,
    baseline_is_bau: bool,
    run_environment: Any = None,
) -> bool:
    """
    Backward-compatible helper for ``decision_simulate`` and tests.

    Delegates to :func:`canonical_decision_safe` (scenario ∧ governance ∧ BAU baseline).
    """
    _ = run_environment
    return canonical_decision_safe(
        scenario_safe=bool(scenario_suitable_for_decisioning and baseline_is_bau),
        governance_gate_allowed=governance_gate_allowed,
    )


def require_bool_decision_safe(sim_js: dict[str, Any], *, context: str) -> bool:
    """Fail closed when ``decision_safe`` is missing or non-bool after enrichment."""
    v = sim_js.get("decision_safe")
    if not isinstance(v, bool):
        raise ValueError(f"{context}: decision_safe must be a bool (got {type(v).__name__})")
    return v
