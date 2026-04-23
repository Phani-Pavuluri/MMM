"""Central decision orchestration: semantic validation, tier checks, prod bundle audit (CLI + API)."""

from __future__ import annotations

from typing import Any

from mmm.artifacts.decision_bundle import validate_prod_decision_bundle
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.contracts.business_surface import (
    BusinessSurfaceMetadataError,
    validate_business_facing_payload,
)
from mmm.contracts.runtime_validation import (
    SemanticContractError,
    assert_decision_artifact_tier,
    validate_semantic_contract,
)
from mmm.governance.policy import PolicyError as GovernancePolicyError
from mmm.contracts.quantity_models import validate_typed_approximate_artifact_section
from mmm.governance.validation import (
    validate_approximate_not_decision_safe,
    validate_decision_tier_lineage,
    validate_section_semantics_dict,
)


def finalize_and_validate_cli_decision_bundle(
    bundle: dict[str, Any],
    cfg: MMMConfig,
    *,
    simulation_json: dict[str, Any] | None,
) -> None:
    """Fail closed on prod semantic / tier / completeness rules for ``mmm decide …`` outputs."""
    if cfg.run_environment != RunEnvironment.PROD:
        return
    try:
        validate_decision_tier_lineage(bundle, run_environment=cfg.run_environment)
        for _name, _sec in (bundle.get("artifact_sections") or {}).items():
            if isinstance(_sec, dict):
                validate_typed_approximate_artifact_section(_sec, section_name=str(_name))
                validate_section_semantics_dict(_sec, section_name=str(_name))
                validate_approximate_not_decision_safe(_sec, section_name=str(_name))
    except GovernancePolicyError as e:
        raise SemanticContractError(str(e)) from e
    validate_semantic_contract(bundle, simulation_json=simulation_json)
    assert_decision_artifact_tier(bundle, run_environment=cfg.run_environment)
    miss = validate_prod_decision_bundle(bundle, run_environment=cfg.run_environment, decision_cli_surface=True)
    if miss:
        raise SemanticContractError("prod decision bundle failed completeness: " + "; ".join(miss))
    try:
        validate_business_facing_payload(
            bundle,
            require_decision_tier=True,
            require_unsupported_questions=True,
        )
    except BusinessSurfaceMetadataError as e:
        raise SemanticContractError(str(e)) from e
    if simulation_json is not None:
        try:
            validate_business_facing_payload(
                simulation_json,
                require_decision_tier=True,
                require_unsupported_questions=True,
            )
        except BusinessSurfaceMetadataError as e:
            raise SemanticContractError(f"simulation JSON business metadata: {e}") from e
