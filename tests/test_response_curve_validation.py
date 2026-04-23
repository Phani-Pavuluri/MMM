"""Curve construction validity: grid density and related diagnostics (non-decision surface)."""

import numpy as np
import pytest

from mmm.decomposition.curves import build_curve_for_channel, build_curve_for_channel_research_only


def test_build_curve_rejects_too_sparse_spend_grid() -> None:
    grid = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="sparse_spend_grid"):
        build_curve_for_channel(
            grid,
            decay=0.5,
            hill_half=1.0,
            hill_slope=2.0,
            beta=0.5,
            model_form="semi_log",
        )


def test_research_only_curve_builder_sparse_grid_typed_envelope() -> None:
    grid = np.array([1.0, 2.0])
    _, env = build_curve_for_channel_research_only(
        grid,
        decay=0.5,
        hill_half=1.0,
        hill_slope=2.0,
        beta=0.5,
        model_form="semi_log",
        bypass_reason="unit_test_sparse_grid",
    )
    assert env["quantity_contract_version"] == "mmm_quantity_envelope_v1"
    assert env["non_canonical_builder"] == "research_only_sparse_or_relaxed"
    assert env["safety"]["decision_safe"] is False
