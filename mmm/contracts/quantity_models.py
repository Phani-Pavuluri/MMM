"""Typed approximate / diagnostic quantity envelopes (machine-checkable; not decision truth)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mmm.contracts.estimands import EstimandKind
from mmm.governance.semantics import (
    BAN_BUDGETING,
    BAN_CHANNEL_ALLOCATION,
    BAN_EXACT_PROFIT_FORECAST,
    BAN_FINANCIAL_COMMITMENT,
    ArtifactTier,
    DecisionSemantics,
    SafetyFlags,
    Surface,
)

QUANTITY_CONTRACT_VERSION = "mmm_quantity_envelope_v1"


class ApproximateQuantityEnvelope(BaseModel):
    """Shared metadata for non–full-panel-Δμ quantities."""

    quantity_contract_version: Literal["mmm_quantity_envelope_v1"] = QUANTITY_CONTRACT_VERSION
    tier: ArtifactTier
    surface: Surface
    semantics: DecisionSemantics
    estimand_kind: EstimandKind
    safety: SafetyFlags
    prod_decisioning_allowed: bool = False
    allowed_surfaces: tuple[str, ...] = ("diagnostic", "research")
    reason_blocked: str | None = Field(
        default=None,
        description="Why this quantity cannot drive prod decisioning (always set for approximate kinds).",
    )

    @model_validator(mode="after")
    def _enforce_safety_invariants(self) -> ApproximateQuantityEnvelope:
        if self.safety.approximate and self.safety.decision_safe:
            raise ValueError("invariant violated: approximate=True cannot pair with decision_safe=True")
        if self.semantics == DecisionSemantics.POSTERIOR_EXPLORATION and self.safety.prod_safe:
            raise ValueError("posterior exploration quantities cannot set prod_safe=True under current policy")
        if self.safety.approximate and not self.safety.unsupported_for:
            raise ValueError(
                "approximate quantities must declare unsupported_for (non-empty) to block downstream misuse"
            )
        return self

    def section_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["safety"] = self.safety.model_dump()
        return d


class CurveQuantityResult(ApproximateQuantityEnvelope):
    estimand_kind: EstimandKind = EstimandKind.APPROX_CURVE
    semantics: DecisionSemantics = DecisionSemantics.APPROX_CURVE
    tier: ArtifactTier = ArtifactTier.DIAGNOSTIC
    surface: Surface = Surface.DIAGNOSTIC
    spend_grid: list[float] = Field(default_factory=list)
    response_on_modeling_scale: list[float] = Field(default_factory=list)
    marginal_roi_modeling_scale: list[float] = Field(default_factory=list)
    validity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    non_canonical_builder: Literal["none", "research_only_sparse_or_relaxed"] = "none"
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=False,
            prod_safe=False,
            approximate=True,
            unsupported_for=[
                BAN_BUDGETING,
                BAN_FINANCIAL_COMMITMENT,
                BAN_CHANNEL_ALLOCATION,
                BAN_EXACT_PROFIT_FORECAST,
            ],
        )
    )
    reason_blocked: str = Field(
        default="curve marginal ROI is diagnostic; not full-panel Δμ decision estimand",
    )


class DecompositionQuantityResult(ApproximateQuantityEnvelope):
    estimand_kind: EstimandKind = EstimandKind.DECOMPOSITION
    semantics: DecisionSemantics = DecisionSemantics.DECOMPOSITION
    tier: ArtifactTier = ArtifactTier.DIAGNOSTIC
    surface: Surface = Surface.DIAGNOSTIC
    channel_shares_modeling_scale: dict[str, float] = Field(default_factory=dict)
    validity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=False,
            prod_safe=False,
            approximate=True,
            unsupported_for=[BAN_BUDGETING, BAN_CHANNEL_ALLOCATION, BAN_EXACT_PROFIT_FORECAST],
        )
    )
    reason_blocked: str = Field(default="decomposition shares are not audited dollar decision truth")


class ROIApproxQuantityResult(ApproximateQuantityEnvelope):
    estimand_kind: EstimandKind = EstimandKind.ROI_FIRST_ORDER
    semantics: DecisionSemantics = DecisionSemantics.ROI_FIRST_ORDER
    tier: ArtifactTier = ArtifactTier.DIAGNOSTIC
    surface: Surface = Surface.DIAGNOSTIC
    mroas_level_proxy: list[float] = Field(default_factory=list)
    economics_notes: dict[str, Any] = Field(default_factory=dict)
    validity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=False,
            prod_safe=False,
            approximate=True,
            unsupported_for=[
                BAN_BUDGETING,
                BAN_FINANCIAL_COMMITMENT,
                BAN_CHANNEL_ALLOCATION,
                BAN_EXACT_PROFIT_FORECAST,
            ],
        )
    )
    reason_blocked: str = Field(
        default="first-order ROI bridge is a level proxy, not exact marginal profit for budgeting",
    )


class UncertaintyBucketsQuantityResult(ApproximateQuantityEnvelope):
    """Structured uncertainty bucket diagnostics (not channel share decomposition)."""

    estimand_kind: EstimandKind = EstimandKind.DIAGNOSTIC_UNCERTAINTY_BUCKETS
    semantics: DecisionSemantics = DecisionSemantics.DIAGNOSTIC_UNCERTAINTY_BUCKETS
    tier: ArtifactTier = ArtifactTier.DIAGNOSTIC
    surface: Surface = Surface.DIAGNOSTIC
    validity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=False,
            prod_safe=False,
            approximate=True,
            unsupported_for=[BAN_BUDGETING, BAN_FINANCIAL_COMMITMENT, BAN_EXACT_PROFIT_FORECAST],
        )
    )
    reason_blocked: str = Field(
        default="uncertainty bucket report is diagnostic; not calibrated intervals or decision Δμ",
    )


class PosteriorExplorationQuantityResult(ApproximateQuantityEnvelope):
    estimand_kind: EstimandKind = EstimandKind.POSTERIOR_EXPLORATION
    semantics: DecisionSemantics = DecisionSemantics.POSTERIOR_EXPLORATION
    tier: ArtifactTier = ArtifactTier.RESEARCH
    surface: Surface = Surface.RESEARCH
    draw_summary: dict[str, Any] = Field(default_factory=dict)
    validity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=False,
            prod_safe=False,
            approximate=True,
            unsupported_for=[BAN_BUDGETING, BAN_FINANCIAL_COMMITMENT, BAN_EXACT_PROFIT_FORECAST],
        )
    )
    prod_decisioning_allowed: bool = False
    allowed_surfaces: tuple[str, ...] = ("diagnostic", "research")
    reason_blocked: str = Field(
        default="current policy disallows Bayesian posterior draws as prod decision truth",
    )


def roi_approx_quantity_from_level_bridge_artifact(
    *,
    mroas_level_proxy: list[float],
    economics_notes: dict[str, Any],
    validity_diagnostics: dict[str, Any] | None = None,
) -> ROIApproxQuantityResult:
    """Canonical typed ROI section for curve artifacts (level-bridge outputs)."""
    return ROIApproxQuantityResult(
        mroas_level_proxy=[float(x) for x in mroas_level_proxy],
        economics_notes=dict(economics_notes),
        validity_diagnostics=dict(validity_diagnostics or {}),
    )


def validate_typed_approximate_artifact_section(section: Any, *, section_name: str) -> None:
    """
    Fail closed when a named approximate artifact section is not a typed quantity envelope.

    ``decision_summary`` is exempt (decision estimand metadata, not an approximate envelope).
    """
    from mmm.governance.policy import PolicyError

    if section_name == "decision_summary":
        return
    approx_sections = {
        "roi_bridge": EstimandKind.ROI_FIRST_ORDER.value,
        "decomposition": EstimandKind.DECOMPOSITION.value,
        "curve_diagnostic": EstimandKind.APPROX_CURVE.value,
    }
    if section_name not in approx_sections:
        return
    if not isinstance(section, dict):
        raise PolicyError(
            f"artifact_sections.{section_name}: expected dict typed section, got {type(section).__name__}"
        )
    if section.get("quantity_contract_version") != QUANTITY_CONTRACT_VERSION:
        raise PolicyError(
            f"artifact_sections.{section_name}: missing quantity_contract_version={QUANTITY_CONTRACT_VERSION!r} "
            "(canonical path must serialize via *QuantityResult.section_dict())."
        )
    want = approx_sections[section_name]
    if section.get("estimand_kind") != want:
        raise PolicyError(
            f"artifact_sections.{section_name}: estimand_kind must be {want!r} for this section "
            f"(got {section.get('estimand_kind')!r})."
        )


def curve_quantity_result_from_response_curve(
    curve: Any,
    *,
    channel: str,
    non_canonical_builder: Literal["none", "research_only_sparse_or_relaxed"] = "none",
    validity_diagnostics: dict[str, Any] | None = None,
) -> CurveQuantityResult:
    """
    Wrap a strict ``ResponseCurve`` as a typed approximate quantity (canonical curve path).

    ``curve`` must be a :class:`mmm.decomposition.curves.ResponseCurve` instance.
    """
    from mmm.decomposition.curves import ResponseCurve

    if not isinstance(curve, ResponseCurve):
        raise TypeError(f"curve must be ResponseCurve (got {type(curve).__name__})")
    vd = dict(validity_diagnostics or {})
    vd.setdefault("channel", channel)
    return CurveQuantityResult(
        spend_grid=[float(x) for x in np.asarray(curve.spend_grid, dtype=float).ravel()],
        response_on_modeling_scale=[float(x) for x in np.asarray(curve.response, dtype=float).ravel()],
        marginal_roi_modeling_scale=[float(x) for x in np.asarray(curve.marginal_roi, dtype=float).ravel()],
        validity_diagnostics=vd,
        non_canonical_builder=non_canonical_builder,
    )


def parse_legacy_curve_bundle_dict(d: dict[str, Any]) -> CurveQuantityResult:
    """
    Adapter: coerce a legacy curve-bundle dict into ``CurveQuantityResult``.

    Fails fast if required numeric series are missing (prevents silent partial semantics).
    """
    required = ("spend_grid", "response_on_modeling_scale", "marginal_roi_modeling_scale")
    missing = [k for k in required if k not in d or d[k] is None]
    if missing:
        raise ValueError(f"legacy curve bundle missing required keys: {missing}")
    sg = np.asarray(d["spend_grid"], dtype=float).ravel()
    resp = np.asarray(d["response_on_modeling_scale"], dtype=float).ravel()
    mroi = np.asarray(d["marginal_roi_modeling_scale"], dtype=float).ravel()
    if sg.size != resp.size or sg.size != mroi.size:
        raise ValueError("legacy curve bundle: spend_grid / response / mroi length mismatch")
    ch = str(d.get("channel") or "unknown_channel")
    return CurveQuantityResult(
        spend_grid=[float(x) for x in sg],
        response_on_modeling_scale=[float(x) for x in resp],
        marginal_roi_modeling_scale=[float(x) for x in mroi],
        validity_diagnostics={"adapter": "parse_legacy_curve_bundle_dict", "channel": ch},
        non_canonical_builder="none",
    )


def reject_if_approximate_quantity_dict(payload: dict[str, Any], *, context: str) -> None:
    """Guard decision paths against typed approximate envelopes passed as opaque dicts."""
    if payload.get("quantity_contract_version") == QUANTITY_CONTRACT_VERSION:
        ek = payload.get("estimand_kind")
        if ek != EstimandKind.FULL_PANEL_DELTA_MU.value:
            from mmm.governance.policy import PolicyError

            raise PolicyError(
                f"{context}: received typed approximate quantity (estimand_kind={ek!r}); "
                "only full_panel_delta_mu is admissible for prod decision optimizers."
            )


def reject_approximate_quantity_subtrees_in_payload(obj: Any, *, context: str) -> None:
    """Depth-first: fail on any typed approximate quantity envelope under ``context``."""
    if isinstance(obj, dict):
        if obj.get("quantity_contract_version") == QUANTITY_CONTRACT_VERSION:
            reject_if_approximate_quantity_dict(obj, context=context)
        for k, v in obj.items():
            reject_approximate_quantity_subtrees_in_payload(v, context=f"{context}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            reject_approximate_quantity_subtrees_in_payload(v, context=f"{context}[{i}]")


__all__ = [
    "QUANTITY_CONTRACT_VERSION",
    "ApproximateQuantityEnvelope",
    "CurveQuantityResult",
    "curve_quantity_result_from_response_curve",
    "DecompositionQuantityResult",
    "parse_legacy_curve_bundle_dict",
    "PosteriorExplorationQuantityResult",
    "ROIApproxQuantityResult",
    "roi_approx_quantity_from_level_bridge_artifact",
    "UncertaintyBucketsQuantityResult",
    "reject_approximate_quantity_subtrees_in_payload",
    "reject_if_approximate_quantity_dict",
    "validate_typed_approximate_artifact_section",
]
