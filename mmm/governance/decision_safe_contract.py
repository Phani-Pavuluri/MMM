"""Single definition of ``decision_safe`` for scenarios and persisted decision payloads."""

from __future__ import annotations

from mmm.config.schema import RunEnvironment


def compute_decision_safe(
    *,
    governance_gate_allowed: bool,
    scenario_suitable_for_decisioning: bool,
    baseline_is_bau: bool,
    run_environment: RunEnvironment | None = None,
) -> bool:
    """
    Unified contract:

    - Governance gate must pass (optimization safety / prod extension gate).
    - Scenario must use a BAU baseline suitable for decisioning.

    ``run_environment`` is accepted for forward-compatible policy hooks; semantics are
    identical across environments so the same scenario yields the same flag everywhere.
    """
    _ = run_environment
    return bool(governance_gate_allowed and scenario_suitable_for_decisioning and baseline_is_bau)
