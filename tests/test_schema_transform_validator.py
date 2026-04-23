import pytest

from mmm.config.schema import Framework, MMMConfig, TransformConfig


def test_ridge_bo_rejects_weibull_adstock():
    with pytest.raises(ValueError, match="geometric"):
        MMMConfig(
            framework=Framework.RIDGE_BO,
            data={"channel_columns": ["c1"], "control_columns": []},
            transforms=TransformConfig(adstock="weibull"),
        )


def test_bayesian_rejects_unimplemented_adstock_at_parse():
    with pytest.raises(ValueError, match="geometric"):
        MMMConfig(
            framework=Framework.BAYESIAN,
            data={"channel_columns": ["c1"], "control_columns": []},
            transforms=TransformConfig(adstock="weibull"),
        )


def test_bayesian_rejects_unimplemented_saturation_at_parse():
    with pytest.raises(ValueError, match="hill"):
        MMMConfig(
            framework=Framework.BAYESIAN,
            data={"channel_columns": ["c1"], "control_columns": []},
            transforms=TransformConfig(saturation="logistic"),
        )
