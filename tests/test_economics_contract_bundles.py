"""economics_contract_for_curve_bundles strict / lenient resolution."""

from __future__ import annotations

import pytest

from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.economics.canonical import build_economics_contract, economics_contract_for_curve_bundles


def _ec(**overrides: object) -> dict:
    base = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(path=None, channel_columns=["a", "b"], target_column="rev"),
        )
    )
    return {**base, **overrides}


def test_non_strict_returns_first_contract() -> None:
    ec = _ec()
    b_no_ec = {"channel": "a", "spend_grid": [1.0, 2.0], "response_on_modeling_scale": [0.0, 1.0]}
    b_ec = {
        "channel": "b",
        "economics_contract": ec,
        "spend_grid": [1.0, 2.0],
        "response_on_modeling_scale": [0.0, 1.0],
    }
    out = economics_contract_for_curve_bundles([b_no_ec, b_ec], strict=False)
    assert out == ec


def test_strict_requires_contract_on_every_bundle() -> None:
    ec = _ec()
    with pytest.raises(ValueError, match="contract_version"):
        economics_contract_for_curve_bundles(
            [
                {"channel": "a", "economics_contract": ec},
                {"channel": "b"},
            ],
            strict=True,
        )


def test_strict_rejects_mismatched_contracts() -> None:
    ec_a = _ec()
    ec_b = {**ec_a, "target_kpi_column": "other_kpi"}
    with pytest.raises(ValueError, match="differs"):
        economics_contract_for_curve_bundles(
            [
                {"channel": "a", "economics_contract": ec_a},
                {"channel": "b", "economics_contract": ec_b},
            ],
            strict=True,
        )


def test_strict_ok_when_all_match() -> None:
    ec = _ec()
    got = economics_contract_for_curve_bundles(
        [
            {"channel": "a", "economics_contract": ec},
            {"channel": "b", "economics_contract": ec},
        ],
        strict=True,
    )
    assert got == ec
