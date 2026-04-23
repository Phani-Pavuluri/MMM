"""Strict validators for decision-tier artifacts (write + read paths)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import RunEnvironment
from mmm.contracts.lineage import assert_decision_tier_lineage_complete
from mmm.governance.policy import PolicyError
from mmm.governance.semantics import ArtifactTier, DecisionSemantics, SafetyFlags, Surface


def validate_decision_tier_lineage(bundle: dict[str, Any], *, run_environment: RunEnvironment) -> None:
    """Fail closed when decision-tier bundles omit required canonical lineage (read/write)."""
    assert_decision_tier_lineage_complete(bundle, _run_environment=run_environment)


def validate_decision_surface_bundle(bundle: dict[str, Any], *, surface: Surface) -> None:
    if surface != Surface.DECISION:
        return
    tier = bundle.get("artifact_tier") or bundle.get("tier")
    if str(tier) != ArtifactTier.DECISION.value:
        raise PolicyError(f"decision surface requires artifact_tier=decision (got {tier!r})")


def validate_approximate_not_decision_safe(section: dict[str, Any], *, section_name: str) -> None:
    """Approximate / proxy sections must not claim decision_safe (excludes the main decision summary)."""
    if section_name == "decision_summary":
        return
    if (
        section.get("quantity_contract_version") == "mmm_quantity_envelope_v1"
        and section.get("prod_decisioning_allowed") is True
        and section.get("estimand_kind") not in ("full_panel_delta_mu",)
    ):
        raise PolicyError(f"{section_name}: approximate quantity cannot claim prod_decisioning_allowed=True")
    safety = section.get("safety") or {}
    if isinstance(safety, dict):
        ds = safety.get("decision_safe")
        approx = safety.get("approximate")
        if approx is True and ds is True:
            raise PolicyError(f"{section_name}: approximate sections cannot set decision_safe=True")


def validate_section_semantics_dict(section: dict[str, Any], *, section_name: str) -> None:
    sem = section.get("semantics")
    if sem is None:
        return
    allowed = {e.value for e in DecisionSemantics}
    if str(sem) not in allowed:
        raise PolicyError(f"{section_name}: unknown semantics {sem!r}")


def safety_flags_from_dict(d: dict[str, Any] | None) -> SafetyFlags:
    if not isinstance(d, dict):
        return SafetyFlags()
    return SafetyFlags.model_validate(
        {
            "decision_safe": bool(d.get("decision_safe", False)),
            "prod_safe": bool(d.get("prod_safe", False)),
            "approximate": bool(d.get("approximate", False)),
            "unsupported_for": list(d.get("unsupported_for") or []),
        }
    )
