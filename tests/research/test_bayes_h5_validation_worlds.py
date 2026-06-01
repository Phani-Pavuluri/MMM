"""Bayes-H5 validation world catalog (research only)."""

from __future__ import annotations

from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    H5_WORLD_IDS,
    WORLD_BAYES_H5_ADSTOCK_ALIGNED,
    WORLD_BAYES_H5_ADSTOCK_MISMATCH,
    WORLD_BAYES_H5_SATURATION_ALIGNED,
    WORLD_BAYES_H5_SATURATION_MISMATCH,
    h5_world_catalog_metadata,
    h5_world_production_flags,
    list_h5_validation_worlds,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import validate_h5_world_catalog
from mmm.research.bayes_h3_sandbox.recovery_worlds import get_recovery_world, list_all_recovery_world_ids


def test_h5_world_catalog_contains_required_worlds() -> None:
    required = {
        WORLD_BAYES_H5_ADSTOCK_ALIGNED,
        WORLD_BAYES_H5_SATURATION_ALIGNED,
        WORLD_BAYES_H5_ADSTOCK_MISMATCH,
        WORLD_BAYES_H5_SATURATION_MISMATCH,
        "WORLD-BAYES-H5-CORRELATED-CHANNELS",
        "WORLD-BAYES-H5-WEAK-SIGNAL",
        "WORLD-BAYES-H5-SPARSE-RECOVERY",
    }
    assert required <= set(H5_WORLD_IDS)
    assert len(list_h5_validation_worlds()) == len(H5_WORLD_IDS)


def test_h5_worlds_registered_in_recovery_registry() -> None:
    for wid in H5_WORLD_IDS:
        assert wid in list_all_recovery_world_ids()
        spec = get_recovery_world(wid)
        assert spec.world_id == wid


def test_h5_world_production_flags_false() -> None:
    flags = h5_world_production_flags()
    assert flags["approved_for_prod"] is False
    assert flags["prod_decisioning_allowed"] is False
    assert flags["hard_gate"] is False
    assert flags["production_promotion"] is False
    for row in h5_world_catalog_metadata():
        assert row["approved_for_prod"] is False
        assert row["prod_decisioning_allowed"] is False


def test_mismatch_vs_aligned_classification() -> None:
    aligned = {
        get_recovery_world(wid).expected_diagnostic_behavior["transform_mismatch_mode"]
        for wid in (WORLD_BAYES_H5_ADSTOCK_ALIGNED, WORLD_BAYES_H5_SATURATION_ALIGNED)
    }
    mismatch = {
        get_recovery_world(wid).expected_diagnostic_behavior["transform_mismatch_mode"]
        for wid in (WORLD_BAYES_H5_ADSTOCK_MISMATCH, WORLD_BAYES_H5_SATURATION_MISMATCH)
    }
    assert aligned == {"aligned"}
    assert mismatch == {"intentional_mismatch"}


def test_h5_catalog_materializes_deterministically() -> None:
    result = validate_h5_world_catalog()
    assert result["status"] == "pass"
