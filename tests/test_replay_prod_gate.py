"""Production replay unit requirements (estimand, lift scale, SE, frames)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_prod_gate import assert_replay_production_ready
from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.experiments.durable_registry import upsert_experiment_record
from mmm.experiments.registry import ApprovalState, ExperimentRecord


_OBS = pd.DataFrame({"g": [1], "w": [1], "y": [1.0], "c1": [1.0]})
_BASE_UNIT = CalibrationUnit(
    unit_id="u1",
    treated_channel_names=["c1"],
    observed_spend_frame=_OBS,
    counterfactual_spend_frame=_OBS.copy(),
    observed_lift=0.05,
    lift_se=0.02,
    target_kpi="y",
    geo_ids=["g"],
    estimand="geo_time_ATT",
    lift_scale="mean_kpi_level_delta",
    replay_estimand={
        "geo_scope": "listed",
        "geo_ids": ["1"],
        "week_start": 0,
        "week_end": 5,
        "aggregation": "mean",
        "target_kpi_column": "y",
        "lift_scale": "mean_kpi_level_delta",
    },
)


def _minimal_unit(**kwargs: object) -> CalibrationUnit:
    return replace(_BASE_UNIT, **kwargs)


def test_prod_replay_gate_ok_complete_unit() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    assert_replay_production_ready(cfg, [_minimal_unit()], schema=schema)


def test_prod_replay_gate_missing_estimand() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    with pytest.raises(ValueError, match="estimand"):
        assert_replay_production_ready(cfg, [_minimal_unit(estimand="")], schema=schema)


def test_prod_replay_gate_rejects_unknown_lift_scale() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    re = dict(_BASE_UNIT.replay_estimand or {})
    re["lift_scale"] = "legacy_relative_lift"
    bad = _minimal_unit(lift_scale="legacy_relative_lift", replay_estimand=re)
    with pytest.raises(ValueError, match="lift_scale"):
        assert_replay_production_ready(cfg, [bad], schema=schema)


def test_prod_replay_gate_non_prod_skips() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["c1"], target_column="y"),
        run_environment=RunEnvironment.RESEARCH,
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    assert_replay_production_ready(cfg, [_minimal_unit(estimand="")])


def _prod_schema_cfg(tmp_reg: Path | None) -> tuple[MMMConfig, PanelSchema]:
    cal_kw: dict = dict(use_replay_calibration=True, replay_units_path="x.json")
    if tmp_reg is not None:
        cal_kw["require_approved_experiment_registry"] = True
        cal_kw["experiment_registry_path"] = str(tmp_reg)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        calibration=CalibrationConfig(**cal_kw),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    return cfg, schema


def test_prod_replay_registry_requires_path(tmp_path: Path) -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={"require_approved_experiment_registry": True, "experiment_registry_path": None}
            )
        }
    )
    with pytest.raises(ValueError, match="experiment_registry_path"):
        assert_replay_production_ready(cfg, [_minimal_unit(experiment_id="e1")], schema=schema)


def test_prod_replay_registry_gate_ok(tmp_path: Path) -> None:
    reg_path = tmp_path / "reg.json"
    upsert_experiment_record(
        reg_path,
        ExperimentRecord(experiment_id="exp-approved", approval=ApprovalState.APPROVED),
    )
    cfg, schema = _prod_schema_cfg(reg_path)
    assert_replay_production_ready(
        cfg, [_minimal_unit(experiment_id="exp-approved")], schema=schema
    )


def test_prod_replay_registry_unknown_experiment_id(tmp_path: Path) -> None:
    reg_path = tmp_path / "reg.json"
    upsert_experiment_record(
        reg_path,
        ExperimentRecord(experiment_id="exp-approved", approval=ApprovalState.APPROVED),
    )
    cfg, schema = _prod_schema_cfg(reg_path)
    with pytest.raises(ValueError, match="not found"):
        assert_replay_production_ready(
            cfg, [_minimal_unit(experiment_id="exp-missing")], schema=schema
        )


def test_prod_replay_registry_not_approved(tmp_path: Path) -> None:
    reg_path = tmp_path / "reg.json"
    upsert_experiment_record(
        reg_path,
        ExperimentRecord(experiment_id="exp-draft", approval=ApprovalState.DRAFT),
    )
    cfg, schema = _prod_schema_cfg(reg_path)
    with pytest.raises(PermissionError, match="approval"):
        assert_replay_production_ready(cfg, [_minimal_unit(experiment_id="exp-draft")], schema=schema)


def test_prod_replay_registry_missing_experiment_id_on_unit(tmp_path: Path) -> None:
    reg_path = tmp_path / "reg.json"
    upsert_experiment_record(
        reg_path,
        ExperimentRecord(experiment_id="exp-approved", approval=ApprovalState.APPROVED),
    )
    cfg, schema = _prod_schema_cfg(reg_path)
    with pytest.raises(ValueError, match="experiment_id required"):
        assert_replay_production_ready(cfg, [_minimal_unit(experiment_id="")], schema=schema)
