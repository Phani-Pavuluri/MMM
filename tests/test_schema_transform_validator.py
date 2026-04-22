import pytest

from mmm.config.schema import Framework, MMMConfig, TransformConfig


def test_ridge_bo_rejects_weibull_adstock():
    with pytest.raises(ValueError, match="geometric"):
        MMMConfig(
            framework=Framework.RIDGE_BO,
            data={"channel_columns": ["c1"], "control_columns": []},
            transforms=TransformConfig(adstock="weibull"),
        )


def test_bayesian_allows_weibull_in_schema_only():
    """Bayesian path does not yet validate transforms the same way (future)."""
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        data={"channel_columns": ["c1"], "control_columns": []},
        transforms=TransformConfig(adstock="weibull"),
    )
    assert cfg.transforms.adstock == "weibull"
