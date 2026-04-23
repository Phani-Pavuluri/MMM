"""Synthetic DGP invariants: coarse recovery checks (not brittle point estimates)."""

import numpy as np
import pytest

from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_synthetic_panel_main_channel_more_correlated_with_target() -> None:
    spec = SyntheticGeoPanelSpec(
        n_geos=4,
        n_weeks=100,
        channels=("high", "mid", "low"),
        betas=(0.5, 0.25, 0.08),
        noise=0.04,
    )
    df, schema = generate_geo_panel(spec, seed=7)
    y = df[schema.target_column].to_numpy(dtype=float)
    corrs = [abs(float(np.corrcoef(df[ch].to_numpy(dtype=float), y)[0, 1])) for ch in schema.channel_columns]
    assert corrs[0] >= corrs[2] - 0.05


def test_known_failure_unsupported_transform_raises() -> None:
    from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm, TransformConfig
    from mmm.config.validators import validate_transform_stack_for_framework

    cfg = MMMConfig.model_construct(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(channel_columns=["c1"], control_columns=[]),
        transforms=TransformConfig(adstock="weibull", saturation="hill"),
    )
    with pytest.raises(ValueError, match="Unsupported canonical media stack"):
        validate_transform_stack_for_framework(cfg)
