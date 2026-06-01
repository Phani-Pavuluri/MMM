"""Bayes-H5 transform-alignment diagnostics (H5b polish)."""

from __future__ import annotations

from mmm.research.bayes_h3_sandbox.h5_transforms import (
    compute_transform_mismatch_detected,
    transforms_aligned,
)
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    WORLD_BAYES_H5_ADSTOCK_ALIGNED,
    WORLD_BAYES_H5_ADSTOCK_MISMATCH,
    WORLD_BAYES_H5_CORRELATED_CHANNELS,
    WORLD_BAYES_H5_SATURATION_ALIGNED,
    WORLD_BAYES_H5_SATURATION_MISMATCH,
    WORLD_BAYES_H5_SPARSE_RECOVERY,
    WORLD_BAYES_H5_WEAK_SIGNAL,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import compute_h5_diagnostic_warnings
from mmm.research.bayes_h3_sandbox.recovery_worlds import get_recovery_world


def _artifact(*, mismatch_detected: bool) -> dict:
    return {"h5_transform_diagnostics": {"transform_mismatch_detected": mismatch_detected}}


def test_transforms_aligned_identity_linear_worlds() -> None:
    assert transforms_aligned("linear", "identity")
    assert transforms_aligned("identity", "identity")
    assert transforms_aligned("correlated", "identity")
    assert transforms_aligned("weak_signal", "identity")


def test_transforms_aligned_media_probes() -> None:
    assert transforms_aligned("geometric_adstock", "geometric_adstock")
    assert transforms_aligned("hill_saturation", "hill_saturation")
    assert not transforms_aligned("geometric_adstock", "identity")
    assert not transforms_aligned("hill_saturation", "identity")


def test_intentional_mismatch_detected() -> None:
    assert compute_transform_mismatch_detected(
        "geometric_adstock",
        "identity",
        transform_mismatch_mode="intentional_mismatch",
    )


def test_sparse_recovery_no_unexpected_mismatch_warning() -> None:
    spec = get_recovery_world(WORLD_BAYES_H5_SPARSE_RECOVERY)
    warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=False), spec)
    assert not any("unexpected_transform_mismatch" in w for w in warnings)
    assert not any(w.startswith("h5:transform_mismatch:") for w in warnings)


def test_aligned_adstock_no_mismatch_warning() -> None:
    spec = get_recovery_world(WORLD_BAYES_H5_ADSTOCK_ALIGNED)
    warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=False), spec)
    assert not any("transform_mismatch" in w for w in warnings)


def test_aligned_saturation_no_mismatch_warning() -> None:
    spec = get_recovery_world(WORLD_BAYES_H5_SATURATION_ALIGNED)
    warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=False), spec)
    assert not any("transform_mismatch" in w for w in warnings)


def test_mismatch_worlds_emit_transform_mismatch_warning() -> None:
    for wid in (WORLD_BAYES_H5_ADSTOCK_MISMATCH, WORLD_BAYES_H5_SATURATION_MISMATCH):
        spec = get_recovery_world(wid)
        warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=True), spec)
        assert any(w.startswith("h5:transform_mismatch:") for w in warnings)
        assert not any("unexpected_transform_mismatch" in w for w in warnings)


def test_correlated_emits_collinearity_not_transform_mismatch() -> None:
    spec = get_recovery_world(WORLD_BAYES_H5_CORRELATED_CHANNELS)
    import pandas as pd

    rng_rows = []
    for t in range(20):
        rng_rows.append({"tv": float(t), "search": float(t) * 0.95})
    panel = pd.DataFrame(rng_rows)
    warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=False), spec, panel_df=panel)
    assert any("collinearity" in w for w in warnings)
    assert not any(w.startswith("h5:transform_mismatch:") for w in warnings)
    assert not any("unexpected_transform_mismatch" in w for w in warnings)


def test_weak_signal_emits_weak_id_not_transform_mismatch() -> None:
    spec = get_recovery_world(WORLD_BAYES_H5_WEAK_SIGNAL)
    warnings = compute_h5_diagnostic_warnings(_artifact(mismatch_detected=False), spec)
    assert any("weak_identification" in w for w in warnings)
    assert not any(w.startswith("h5:transform_mismatch:") for w in warnings)
    assert not any("unexpected_transform_mismatch" in w for w in warnings)
