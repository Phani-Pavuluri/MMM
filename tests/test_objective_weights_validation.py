import pytest

from mmm.config.schema import MMMConfig, ObjectiveWeights


def test_objective_weights_reject_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        ObjectiveWeights(predictive=-0.1)


def test_objective_weights_reject_zero_sum() -> None:
    with pytest.raises(ValueError, match="positive"):
        ObjectiveWeights(
            predictive=0.0,
            calibration=0.0,
            stability=0.0,
            plausibility=0.0,
            complexity=0.0,
        )


def test_nested_objective_weights_in_config() -> None:
    with pytest.raises(ValueError):
        MMMConfig(
            data={"channel_columns": ["c1"], "control_columns": []},
            objective={"weights": {"predictive": 1.0, "calibration": -1.0}},
        )
