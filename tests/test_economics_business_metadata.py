"""Economics metadata completeness for business-facing JSON."""

from __future__ import annotations

import pytest

from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.economics.canonical import economics_output_metadata, validate_business_economics_metadata


def _cfg() -> MMMConfig:
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            target_column="y",
            channel_columns=["c1"],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=5, horizon_weeks=2),
        ridge_bo={"n_trials": 1},
    )


def test_economics_output_metadata_has_required_business_keys() -> None:
    cfg = _cfg()
    meta = economics_output_metadata(
        cfg,
        uncertainty_mode="point",
        surface="full_model_simulation",
        baseline_type="bau",
        decision_safe=True,
    )
    validate_business_economics_metadata(
        meta,
        require_specific_baseline=True,
        require_decision_safe_bool=True,
    )
    assert meta["economics_version"] == meta["economics_contract_version"]
    assert meta["computation_mode"] == "exact"
    assert meta["baseline_type"] == "bau"


def test_validate_rejects_unspecified_baseline_when_required() -> None:
    cfg = _cfg()
    meta = economics_output_metadata(cfg, surface="full_model_simulation", uncertainty_mode="point")
    assert meta["baseline_type"] == "unspecified"
    with pytest.raises(ValueError, match="baseline_type"):
        validate_business_economics_metadata(meta, require_specific_baseline=True)


def test_validate_rejects_missing_decision_safe_when_required() -> None:
    cfg = _cfg()
    meta = economics_output_metadata(
        cfg,
        surface="full_model_simulation",
        uncertainty_mode="point",
        baseline_type="bau",
        decision_safe=None,
    )
    with pytest.raises(ValueError, match="decision_safe"):
        validate_business_economics_metadata(meta, require_decision_safe_bool=True)
