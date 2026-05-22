"""LOG_LOG is research-only; blocked from prod config, prod decisions, and hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmm.artifacts.decision_bundle import compute_unsupported_questions
from mmm.config.schema import (
    CVConfig,
    Framework,
    HierarchyConfig,
    MMMConfig,
    ModelForm,
    RunEnvironment,
)
from mmm.data.schema import PanelSchema
from mmm.decision.service import _apply_runtime_policy_prechecks
from mmm.governance.model_form_policy import (
    LOG_LOG_HIERARCHY_POLICY_MESSAGE,
    LOG_LOG_PROD_POLICY_MESSAGE,
    LOG_LOG_UNSUPPORTED_QUESTIONS,
)
from mmm.governance.policy import PolicyError, runtime_policy_from_config
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition
from mmm.hierarchy.hierarchy_extension import load_and_validate_hierarchy
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def _minimal_prod_kwargs() -> dict:
    return {
        "run_environment": RunEnvironment.PROD,
        "framework": Framework.RIDGE_BO,
        "data": {"channel_columns": ["c1", "c2"], "control_columns": [], "data_version_id": "dv1"},
        "cv": {"mode": "rolling"},
        "objective": {
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        "extensions": {"optimization_gates": {"enabled": True}},
    }


def test_prod_config_log_log_fails() -> None:
    with pytest.raises(PolicyError, match="LOG_LOG is research-only"):
        MMMConfig(
            **_minimal_prod_kwargs(),
            model_form=ModelForm.LOG_LOG,
            prod_canonical_modeling_contract_id="ridge_bo_log_log_calendar_cv_v1",
        )


def test_research_config_log_log_passes() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.LOG_LOG,
        data={"channel_columns": ["c1", "c2"]},
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=5, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    assert cfg.model_form == ModelForm.LOG_LOG


def test_prod_decision_rejects_stale_log_log_extension_report() -> None:
    cfg = MMMConfig(
        **_minimal_prod_kwargs(),
        model_form=ModelForm.SEMI_LOG,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
    )
    er = {
        "ridge_fit_summary": {"coef": [0.1, 0.2], "best_params": {}, "intercept": [1.0]},
        "economics_contract": {"model_form": "log_log"},
        "model_release": {"state": "planning_allowed"},
        "panel_qa": {"max_severity": "info"},
    }
    policy = runtime_policy_from_config(cfg)
    with pytest.raises(PolicyError, match="stale LOG_LOG artifacts"):
        _apply_runtime_policy_prechecks(cfg, er, policy)


def test_hierarchy_enabled_with_log_log_fails(tmp_path: Path) -> None:
    defn = HierarchyDefinition(
        hierarchy_id="ch",
        hierarchy_type="channel",
        version="1",
        parent_nodes=["P"],
        child_nodes=["a", "b"],
        node_mapping={"a": "P", "b": "P"},
    )
    p = tmp_path / "h.json"
    p.write_text(defn.model_dump_json(), encoding="utf-8")
    with pytest.raises(PolicyError, match=LOG_LOG_HIERARCHY_POLICY_MESSAGE):
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.LOG_LOG,
            data={"channel_columns": ["P", "a", "b"]},
            hierarchy=HierarchyConfig(enabled=True, hierarchy_definition_path=str(p)),
        )


def test_semi_log_hierarchy_still_works(tmp_path: Path) -> None:
    defn = HierarchyDefinition(
        hierarchy_id="ch",
        hierarchy_type="channel",
        version="1",
        parent_nodes=["Paid_Social"],
        child_nodes=["Meta", "TikTok", "Reddit"],
        node_mapping={
            "Meta": "Paid_Social",
            "TikTok": "Paid_Social",
            "Reddit": "Paid_Social",
        },
    )
    p = tmp_path / "h.json"
    p.write_text(defn.model_dump_json(), encoding="utf-8")
    rows = []
    channels = ("Paid_Social", "Meta", "TikTok", "Reddit")
    for g in range(2):
        for w in range(20):
            row = {"geo_id": f"g{g}", "week_start_date": w, "revenue": 100.0 + w}
            for c in channels:
                row[c] = 5.0
            rows.append(row)
    import pandas as pd

    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", channels)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": list(channels),
        },
        hierarchy=HierarchyConfig(enabled=True, hierarchy_definition_path=str(p)),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    _def, report, pairs, _ = load_and_validate_hierarchy(cfg, schema, panel)
    assert report.valid
    assert len(pairs) == 3
    trainer = RidgeBOMMMTrainer(cfg, schema)
    trainer.fit(panel)
    assert trainer._hierarchy_pairs


def test_log_log_unsupported_questions_emitted() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.LOG_LOG,
        data={"channel_columns": ["c1"]},
    )
    qs = compute_unsupported_questions(cfg, {"economics_contract": {"model_form": "log_log"}})
    for expected in LOG_LOG_UNSUPPORTED_QUESTIONS:
        assert expected in qs


def test_prod_policy_message_constants() -> None:
    assert "SEMI_LOG" in LOG_LOG_PROD_POLICY_MESSAGE
    assert "validation" in LOG_LOG_HIERARCHY_POLICY_MESSAGE.lower()


def test_prod_model_form_contract_test_updated() -> None:
    """Former wrong-contract test: prod LOG_LOG now fails before contract id check."""
    with pytest.raises(PolicyError, match="research-only"):
        MMMConfig(
            **_minimal_prod_kwargs(),
            model_form=ModelForm.LOG_LOG,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        )
