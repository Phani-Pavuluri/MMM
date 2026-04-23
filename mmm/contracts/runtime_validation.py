"""Machine-enforced semantic contracts for decision-facing artifacts (prod fail-closed)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import RunEnvironment
from mmm.contracts.artifact_tier import DECISION_TIER_VALUE


class SemanticContractError(ValueError):
    """Raised when a decision bundle or payload violates semantic contracts."""


def validate_estimand_alignment(*, bundle_estimand: str | None) -> None:
    if not bundle_estimand:
        raise SemanticContractError("semantic_contract.estimand is required")
    if bundle_estimand != "delta_mu_full_panel":
        raise SemanticContractError(
            f"semantic_contract.estimand must be 'delta_mu_full_panel' (got {bundle_estimand!r})"
        )


def validate_aggregation_alignment(*, bundle_aggregation: str | None, simulation_aggregation: str | None) -> None:
    if not bundle_aggregation:
        raise SemanticContractError("semantic_contract.aggregation is required")
    if simulation_aggregation and bundle_aggregation != simulation_aggregation:
        raise SemanticContractError(
            f"semantic_contract.aggregation {bundle_aggregation!r} != simulation aggregation_semantics "
            f"{simulation_aggregation!r}"
        )


def validate_scale_alignment(*, bundle_scale: str | None, model_form: str | None, target_column: str | None) -> None:
    if not bundle_scale:
        raise SemanticContractError("semantic_contract.scale is required")
    if model_form and model_form not in bundle_scale:
        raise SemanticContractError(
            f"semantic_contract.scale {bundle_scale!r} must reference model_form={model_form!r}"
        )
    if target_column and target_column not in bundle_scale:
        raise SemanticContractError(
            f"semantic_contract.scale {bundle_scale!r} must reference target_column={target_column!r}"
        )


def validate_baseline_consistency(*, baseline_type: str | None, economics_baseline: str | None) -> None:
    if baseline_type in (None, "", "unspecified"):
        raise SemanticContractError("baseline_type must be explicit for decision bundles")
    if economics_baseline in (None, "", "unspecified"):
        raise SemanticContractError("economics_output_metadata.baseline_type must be explicit")
    if baseline_type != economics_baseline:
        raise SemanticContractError(
            f"baseline mismatch: bundle baseline_type={baseline_type!r} vs economics {economics_baseline!r}"
        )


def validate_semantic_contract(bundle: dict[str, Any], *, simulation_json: dict[str, Any] | None = None) -> None:
    """
    Validate required semantic fields on a decision bundle.

    Raises ``SemanticContractError`` when any required field is missing or inconsistent.
    """
    sem = bundle.get("semantic_contract")
    if not isinstance(sem, dict):
        raise SemanticContractError("decision_bundle.semantic_contract must be a dict")
    for key in ("estimand", "aggregation", "scale", "baseline_definition"):
        if not sem.get(key):
            raise SemanticContractError(f"semantic_contract.{key} is required")

    econ = bundle.get("economics_output_metadata")
    if not isinstance(econ, dict):
        raise SemanticContractError("economics_output_metadata must be a dict")
    econ_ver = bundle.get("economics_version") or econ.get("economics_contract_version")
    if not econ_ver:
        raise SemanticContractError("economics_contract_version / economics_version is required")

    sim_agg = simulation_json.get("aggregation_semantics") if simulation_json else None
    validate_aggregation_alignment(bundle_aggregation=sem.get("aggregation"), simulation_aggregation=sim_agg)

    validate_estimand_alignment(bundle_estimand=sem.get("estimand"))

    rc = bundle.get("resolved_config_snapshot") or {}
    mf = rc.get("model_form") if isinstance(rc, dict) else None
    tgt = None
    if isinstance(rc, dict) and isinstance(rc.get("data"), dict):
        tgt = rc["data"].get("target_column")

    validate_scale_alignment(
        bundle_scale=sem.get("scale"),
        model_form=str(mf) if mf else None,
        target_column=str(tgt) if tgt else None,
    )

    validate_baseline_consistency(
        baseline_type=str(bundle.get("baseline_type") or sem.get("baseline_definition") or ""),
        economics_baseline=str(econ.get("baseline_type") or ""),
    )
    if str(sem.get("baseline_definition", "")) != str(bundle.get("baseline_type", "")):
        raise SemanticContractError(
            "semantic_contract.baseline_definition must match top-level baseline_type on the decision bundle"
        )


def assert_decision_artifact_tier(bundle: dict[str, Any], *, run_environment: RunEnvironment) -> None:
    """Prod: only ``decision`` tier may be used for decision CLI outputs."""
    if run_environment != RunEnvironment.PROD:
        return
    tier = bundle.get("artifact_tier")
    if tier != DECISION_TIER_VALUE:
        raise SemanticContractError(
            f"artifact_tier must be {DECISION_TIER_VALUE!r} for prod decision outputs (got {tier!r})"
        )
    if bundle.get("not_for_budgeting") is not False:
        raise SemanticContractError("prod decision bundle requires not_for_budgeting=False")
    if bundle.get("decision_safe") is not True:
        raise SemanticContractError("prod decision bundle requires decision_safe=True")


def validate_proxy_reporting_payload(payload: dict[str, Any]) -> None:
    """Ensure diagnostic/research ROI payloads carry misuse-prevention flags."""
    if payload.get("is_proxy_metric") is not True:
        raise SemanticContractError("reporting payload missing is_proxy_metric=True")
    if payload.get("not_exact_business_value") is not True:
        raise SemanticContractError("reporting payload missing not_exact_business_value=True")
    if payload.get("not_for_budgeting") is not True:
        raise SemanticContractError("proxy reporting payload requires not_for_budgeting=True")


def tier_for_surface(surface: str) -> str:
    if surface in ("curve_diagnostic", "decomposition"):
        return "diagnostic"
    if surface in ("replay_calibration",):
        return "research"
    if surface == "full_model_simulation":
        return "decision"
    return "research"
