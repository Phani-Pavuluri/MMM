"""Environment → governance defaults (roadmap §6)."""

from __future__ import annotations

from dataclasses import dataclass

from mmm.config.schema import RunEnvironment


@dataclass(frozen=True)
class EnvironmentPolicy:
    """Resolved rules for a run."""

    identifiability_fail_closed: bool
    require_explicit_override_for_unsafe: bool
    log_level: str


def policy_for_environment(env: RunEnvironment) -> EnvironmentPolicy:
    if env == RunEnvironment.DEV:
        return EnvironmentPolicy(
            identifiability_fail_closed=False,
            require_explicit_override_for_unsafe=False,
            log_level="debug",
        )
    if env == RunEnvironment.RESEARCH:
        return EnvironmentPolicy(
            identifiability_fail_closed=False,
            require_explicit_override_for_unsafe=False,
            log_level="info",
        )
    if env == RunEnvironment.STAGING:
        return EnvironmentPolicy(
            identifiability_fail_closed=False,
            require_explicit_override_for_unsafe=True,
            log_level="info",
        )
    # prod
    return EnvironmentPolicy(
        identifiability_fail_closed=True,
        require_explicit_override_for_unsafe=True,
        log_level="warning",
    )


def approved_for_optimization_with_policy(
    *,
    base_approved: bool,
    env: RunEnvironment,
    override_unsafe: bool,
    identifiability_risk_ok: bool,
) -> tuple[bool, list[str]]:
    """
    Apply environment gates on top of scorecard ``approved_for_optimization``.

    In ``prod``, high identifiability risk blocks optimization unless
    ``override_unsafe`` is set on config (must be logged by caller).
    """
    pol = policy_for_environment(env)
    notes: list[str] = []
    ok = base_approved
    if pol.identifiability_fail_closed and not identifiability_risk_ok:
        ok = False
        notes.append("identifiability_risk_blocked_in_prod")
        if override_unsafe:
            ok = base_approved
            notes.append("unsafe_override_acknowledged")
    return ok, notes
