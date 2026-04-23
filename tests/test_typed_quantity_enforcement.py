"""Typed approximate quantity models: invariants and decision-path rejection."""

import pytest

from mmm.contracts.estimands import EstimandKind
from mmm.contracts.quantity_models import (
    ROIApproxQuantityResult,
    parse_legacy_curve_bundle_dict,
    reject_approximate_quantity_subtrees_in_payload,
    reject_if_approximate_quantity_dict,
)
from mmm.governance.policy import PolicyError
from mmm.governance.semantics import SafetyFlags


def test_approximate_cannot_claim_decision_safe() -> None:
    with pytest.raises(ValueError, match="approximate=True"):
        ROIApproxQuantityResult(
            safety=SafetyFlags(
                decision_safe=True,
                prod_safe=False,
                approximate=True,
                unsupported_for=["budgeting"],
            ),
        )


def test_parse_legacy_curve_bundle_dict_requires_series() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        parse_legacy_curve_bundle_dict({"channel": "c1", "spend_grid": [1.0]})
    q = parse_legacy_curve_bundle_dict(
        {
            "channel": "c1",
            "spend_grid": [1.0, 2.0],
            "response_on_modeling_scale": [0.1, 0.2],
            "marginal_roi_modeling_scale": [0.05, 0.05],
        }
    )
    assert q.estimand_kind.value == "approx_curve"


def test_reject_subtrees_nested_under_scenario() -> None:
    bad = {
        "candidate_spend": {
            "c1": 1.0,
            "nested": {
                "quantity_contract_version": "mmm_quantity_envelope_v1",
                "estimand_kind": "approx_curve",
                "tier": "diagnostic",
                "surface": "diagnostic",
                "semantics": "approx_curve",
                "safety": {
                    "decision_safe": False,
                    "prod_safe": False,
                    "approximate": True,
                    "unsupported_for": ["budgeting"],
                },
            },
        }
    }
    with pytest.raises(PolicyError, match="typed approximate quantity"):
        reject_approximate_quantity_subtrees_in_payload(bad, context="scenario_yaml")


def test_reject_if_approximate_quantity_dict_blocks_decision_context() -> None:
    bad = {
        "quantity_contract_version": "mmm_quantity_envelope_v1",
        "estimand_kind": EstimandKind.APPROX_CURVE.value,
        "tier": "diagnostic",
    }
    with pytest.raises(PolicyError, match="typed approximate quantity"):
        reject_if_approximate_quantity_dict(bad, context="optimizer")
