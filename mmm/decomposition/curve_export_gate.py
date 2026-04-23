"""Fail-closed checks for curve bundle dicts crossing export / strict contract surfaces."""

from __future__ import annotations

from typing import Any

from mmm.contracts.quantity_models import QUANTITY_CONTRACT_VERSION, CurveQuantityResult
from mmm.governance.policy import PolicyError


def validate_curve_bundle_typed_curve_quantity(bundle: dict[str, Any], *, context: str) -> None:
    """
    Require ``typed_curve_quantity`` envelope (canonical export contract).

    Internal optimizers may still consume bare spend/response series in research-only paths;
    strict economics / prod-style ingestion must call this gate.
    """
    raw = bundle.get("typed_curve_quantity")
    if not isinstance(raw, dict):
        raise PolicyError(
            f"{context}: curve bundle is missing typed_curve_quantity "
            "(export surfaces require CurveQuantityResult.section_dict())."
        )
    if raw.get("quantity_contract_version") != QUANTITY_CONTRACT_VERSION:
        raise PolicyError(
            f"{context}: typed_curve_quantity must declare quantity_contract_version="
            f"{QUANTITY_CONTRACT_VERSION!r}."
        )
    try:
        CurveQuantityResult.model_validate(raw)
    except Exception as e:
        raise PolicyError(f"{context}: typed_curve_quantity failed contract validation: {e}") from e


__all__ = ["validate_curve_bundle_typed_curve_quantity"]
