"""Machine-readable business / decision surface metadata (tiers, exactness, KPI contract)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.contracts.artifact_tier import DECISION_TIER_VALUE
from mmm.economics.canonical import ECONOMICS_CONTRACT_VERSION


class BusinessSurfaceMetadataError(ValueError):
    """Raised when a payload claiming to be business-facing is missing required metadata."""


REQUIRED_BUSINESS_DECISION_KEYS: frozenset[str] = frozenset(
    {
        "artifact_tier",
        "decision_safe",
        "approximate",
        "not_for_budgeting",
        "economics_contract_version",
        "kpi_column",
        "kpi_unit_semantics",
        "baseline_type",
    }
)


def validate_business_facing_payload(
    payload: dict[str, Any],
    *,
    require_decision_tier: bool,
    require_unsupported_questions: bool,
) -> None:
    """
    Fail closed when a dict is advertised as business-facing but omits tier / exactness / KPI semantics.

    ``require_decision_tier``: PROD decision CLI JSON (must match ``decision`` tier + not_for_budgeting False).
    ``require_unsupported_questions``: decision bundle-like payloads must carry the transparency list.
    """
    missing = sorted(REQUIRED_BUSINESS_DECISION_KEYS - set(payload.keys()))
    if missing:
        raise BusinessSurfaceMetadataError(f"business-facing payload missing keys: {missing}")
    if not isinstance(payload.get("decision_safe"), bool):
        raise BusinessSurfaceMetadataError("decision_safe must be bool")
    if not isinstance(payload.get("approximate"), bool):
        raise BusinessSurfaceMetadataError("approximate must be bool")
    if not isinstance(payload.get("not_for_budgeting"), bool):
        raise BusinessSurfaceMetadataError("not_for_budgeting must be bool")
    if not payload.get("economics_contract_version"):
        raise BusinessSurfaceMetadataError("economics_contract_version required")
    if not payload.get("kpi_column"):
        raise BusinessSurfaceMetadataError("kpi_column required")
    if not payload.get("kpi_unit_semantics"):
        raise BusinessSurfaceMetadataError("kpi_unit_semantics required")
    if not payload.get("baseline_type"):
        raise BusinessSurfaceMetadataError("baseline_type required")
    if require_decision_tier:
        if payload.get("artifact_tier") != DECISION_TIER_VALUE:
            raise BusinessSurfaceMetadataError(
                f"prod business-facing artifact requires artifact_tier={DECISION_TIER_VALUE!r}"
            )
        if payload.get("not_for_budgeting") is not False:
            raise BusinessSurfaceMetadataError("prod decision payload requires not_for_budgeting=False")
        if payload.get("decision_safe") is not True:
            raise BusinessSurfaceMetadataError("prod decision payload requires decision_safe=True")
    if require_unsupported_questions and not isinstance(payload.get("unsupported_questions"), list):
        raise BusinessSurfaceMetadataError("unsupported_questions must be a list on decision bundle")


def enrich_decision_simulation_json(
    sim_js: dict[str, Any],
    *,
    cfg: MMMConfig,
    unsupported_questions: list[str],
    governance_gate_allowed: bool,
) -> dict[str, Any]:
    """
    Attach stable business-surface fields to ``decision_simulate`` JSON (machine inspection).

    Tier is ``decision`` only when prod gate passed and environment is prod; otherwise ``research``.
    """
    out = dict(sim_js)
    prod = cfg.run_environment == RunEnvironment.PROD
    tier = DECISION_TIER_VALUE if prod and governance_gate_allowed else "research"
    approx = str(out.get("uncertainty_mode", "point")) != "point"
    out["artifact_tier"] = tier
    # CLI prod path: ``decision_safe`` on persisted JSON follows the optimization safety gate, not only BAU flags.
    out["decision_safe"] = bool(governance_gate_allowed)
    out["approximate"] = approx
    out["not_for_budgeting"] = tier != DECISION_TIER_VALUE
    out["economics_contract_version"] = str(out.get("economics_version") or ECONOMICS_CONTRACT_VERSION)
    out["kpi_column"] = str(out.get("kpi_column") or cfg.data.target_column)
    out["kpi_unit_semantics"] = "same_units_as_training_target_column"
    out["baseline_type"] = str(out.get("baseline_type") or out.get("baseline_definition") or "bau")
    out["unsupported_questions"] = list(unsupported_questions)
    return out


def optimization_response_business_metadata(
    *,
    cfg: MMMConfig,
    bundle: dict[str, Any],
    governance_gate_allowed: bool,
) -> dict[str, Any]:
    """Top-level metadata block mirroring the decision bundle for optimization JSON consumers."""
    prod = cfg.run_environment == RunEnvironment.PROD
    tier = bundle.get("artifact_tier", "research")
    return {
        "artifact_tier": tier,
        "decision_safe": bool(bundle.get("decision_safe")) and governance_gate_allowed,
        "approximate": bool(bundle.get("approximate")),
        "not_for_budgeting": bool(bundle.get("not_for_budgeting")),
        "economics_contract_version": str(bundle.get("economics_version") or ECONOMICS_CONTRACT_VERSION),
        "kpi_column": str(bundle.get("target_kpi") or cfg.data.target_column),
        "kpi_unit_semantics": "same_units_as_training_target_column",
        "baseline_type": str(bundle.get("baseline_type") or "bau"),
        "unsupported_questions": list(bundle.get("unsupported_questions") or []),
        "optimization_surface": "full_model_simulation_slsqp",
        "prod_environment": prod,
    }
