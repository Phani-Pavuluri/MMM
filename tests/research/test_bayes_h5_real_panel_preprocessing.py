"""Tests for H5 real-panel collinearity preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmm.data.schema import PanelSchema
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_DROP_SPARSE,
    H5RealPanelPreprocessingError,
    apply_channel_policy,
    build_composite_media_panel,
    build_drop_collinear_panel,
    build_sparse_channel_drop_panel,
    detect_collinear_channel_groups,
    validate_collinearity_config,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import load_panel_from_path, load_transform_config


def _schema_three_channel() -> PanelSchema:
    return PanelSchema("geo_id", "week_start_date", "revenue", ("search", "social", "tv"), ())


def test_collinear_groups_detected_on_sample_panel() -> None:
    df = load_panel_from_path("examples/sample_panel.csv")
    groups = detect_collinear_channel_groups(df, ("search", "social", "tv"), max_abs_corr_threshold=0.95)
    assert groups
    assert any("social" in g["channels"] and "tv" in g["channels"] for g in groups)


def test_drop_collinear_records_dropped_channels_and_reason() -> None:
    df = load_panel_from_path("examples/sample_panel.csv")
    schema = _schema_three_channel()
    _, _, record = build_drop_collinear_panel(df, schema, max_abs_corr_threshold=0.95)
    assert record["dropped_channels"]
    assert record["kept_channels"]
    assert len(record["kept_channels"]) >= 1
    for drop in record["dropped_channels"]:
        assert "reason" in drop
        assert "collinear" in drop["reason"]


def test_composite_records_source_channels() -> None:
    df = load_panel_from_path("examples/sample_panel.csv")
    schema = _schema_three_channel()
    _, out_schema, record = build_composite_media_panel(
        df,
        schema,
        source_channels=["social", "tv"],
        method="sum_scaled_media",
        output_channel="social_tv_sum",
        remaining_channels=["search"],
    )
    assert "social_tv_sum" in out_schema.channel_columns
    assert record["source_channels"] == ["social", "tv"]
    assert record["method"] == "sum_scaled_media"


def test_no_silent_dropping_requires_explicit_policy() -> None:
    df = load_panel_from_path("examples/sample_panel.csv")
    schema = _schema_three_channel()
    cfg = load_transform_config("docs/06_investigations/h5g_sample_panel_transform_config.json")
    _, _, _, record = apply_channel_policy(df, schema, cfg)
    assert record["mode"] == "keep_all_channels"
    assert len(record["kept_channels"]) == 3


def test_invalid_config_fail_closed() -> None:
    with pytest.raises(H5RealPanelPreprocessingError):
        validate_collinearity_config({"mode": "drop_collinear_channels"})
    with pytest.raises(H5RealPanelPreprocessingError):
        validate_collinearity_config({"mode": "single_channel"})
    with pytest.raises(H5RealPanelPreprocessingError):
        validate_collinearity_config(
            {"mode": "composite_media_channel", "source_channels": ["a"], "method": "bad", "output_channel": "x"}
        )
    with pytest.raises(H5RealPanelPreprocessingError, match="reason"):
        validate_collinearity_config(
            {
                "mode": CHANNEL_POLICY_DROP_SPARSE,
                "dropped_channels": ["radio"],
                "kept_channels": ["search"],
                "no_silent_dropping": True,
            }
        )


def test_sparse_channel_drop_requires_explicit_lists_and_reason() -> None:
    path = Path("examples/triangulation_geo_panel_v1.csv")
    if not path.is_file():
        pytest.skip("triangulation panel missing")
    df = load_panel_from_path(path)
    schema = PanelSchema(
        "geo_id",
        "week_start_date",
        "revenue",
        ("search", "social", "display", "radio"),
        (),
    )
    _, out_schema, record = build_sparse_channel_drop_panel(
        df,
        schema,
        dropped_channels=["radio"],
        kept_channels=["search", "social", "display"],
        reason="radio near_zero_share ~0.99 — sparse governed drop",
    )
    assert record["mode"] == CHANNEL_POLICY_DROP_SPARSE
    assert "radio" not in out_schema.channel_columns
    assert "sparse" in record["sparse_drop_reason"].lower()

