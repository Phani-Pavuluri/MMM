"""Bayes-H5 research transform registry (deterministic)."""

from __future__ import annotations

import numpy as np

from mmm.research.bayes_h3_sandbox.h5_transforms import (
    TRANSFORM_IDS,
    apply_channel_transform,
    apply_media_transforms_matrix,
    list_transform_registry,
    transforms_aligned,
)


def test_transform_registry_lists_required_ids() -> None:
    reg = list_transform_registry()
    assert reg["research_only"] is True
    assert reg["wired_to_production_feature_engineering"] is False
    for tid in ("identity", "geometric_adstock", "hill_saturation", "adstock_then_saturation"):
        assert tid in TRANSFORM_IDS


def test_transforms_are_deterministic() -> None:
    x = np.array([1.0, 2.0, 3.0, 2.5, 4.0])
    for tid in TRANSFORM_IDS:
        a = apply_channel_transform(x, tid)
        b = apply_channel_transform(x, tid)
        np.testing.assert_allclose(a, b)


def test_media_matrix_shape_preserved() -> None:
    x = np.random.default_rng(0).uniform(0.5, 5.0, size=(12, 2))
    out = apply_media_transforms_matrix(
        x,
        ["tv", "search"],
        {"tv": "geometric_adstock", "search": "identity"},
    )
    assert out.shape == x.shape


def test_transforms_aligned_helper() -> None:
    assert transforms_aligned("geometric_adstock", "geometric_adstock")
    assert not transforms_aligned("geometric_adstock", "identity")
    assert transforms_aligned("hill_saturation", "hill_saturation")
