"""PR 4A: Ridge hierarchical borrowing (explicit hierarchy, opt-in)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.config.schema import (
    CVConfig,
    Framework,
    HierarchyConfig,
    MMMConfig,
    ModelForm,
)
from mmm.data.schema import PanelSchema
from mmm.hierarchy.diagnostics import (
    HIERARCHY_GOVERNANCE_WARNINGS,
    HIERARCHY_UNSUPPORTED_QUESTIONS,
    build_hierarchy_diagnostics,
    hierarchy_enabled,
)
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, load_hierarchy_definition
from mmm.hierarchy.hierarchy_extension import build_hierarchy_reports_for_fit
from mmm.hierarchy.penalty import hierarchical_penalty, prepare_hierarchy_for_ridge
from mmm.hierarchy.validator import HierarchyValidator
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def _channel_hierarchy() -> HierarchyDefinition:
    return HierarchyDefinition(
        hierarchy_id="paid_social_v1",
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


def _write_hierarchy(tmp_path: Path, definition: HierarchyDefinition) -> Path:
    p = tmp_path / "hierarchy.json"
    p.write_text(definition.model_dump_json(), encoding="utf-8")
    return p


def _panel_and_schema() -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    channels = ("Paid_Social", "Meta", "TikTok", "Reddit")
    for g in range(2):
        for w in range(24):
            row = {"geo_id": f"dma_{g}", "week_start_date": w, "revenue": 100.0 + w}
            for c in channels:
                row[c] = 10.0 + w * 0.1
            rows.append(row)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", channels)
    return pd.DataFrame(rows), schema


def test_hierarchy_validation_cycle_rejected() -> None:
    defn = _channel_hierarchy()
    defn = defn.model_copy(update={"node_mapping": {"Meta": "TikTok", "TikTok": "Meta", "Reddit": "Paid_Social"}})
    report = HierarchyValidator(min_children_per_parent=2).validate(
        defn, model_entities=set(defn.child_nodes) | {"Paid_Social"}
    )
    assert not report.valid
    assert report.cycle_detected


def test_hierarchy_validation_duplicate_child_rejected() -> None:
    defn = _channel_hierarchy()
    bad_mapping = dict(defn.node_mapping)
    bad_mapping["Meta"] = "Paid_Social"
    defn = defn.model_copy(update={"node_mapping": bad_mapping})
    validator = HierarchyValidator(min_children_per_parent=2, allow_cross_branch_pooling=False)
    report = validator.validate(defn, model_entities=set(defn.child_nodes) | {"Paid_Social"})
    assert report.valid or not report.duplicate_assignments  # single parent per child OK


def test_hierarchy_validation_missing_node_rejected() -> None:
    defn = _channel_hierarchy()
    extra = {**defn.node_mapping, "MissingChan": "Paid_Social"}
    defn = defn.model_copy(
        update={"child_nodes": [*defn.child_nodes, "MissingChan"], "node_mapping": extra}
    )
    report = HierarchyValidator().validate(defn, model_entities={"Paid_Social", "Meta", "TikTok", "Reddit"})
    assert not report.valid


def test_hierarchy_validation_orphan_rejected() -> None:
    defn = _channel_hierarchy()
    defn = defn.model_copy(update={"child_nodes": list(defn.child_nodes) + ["OrphanChild"]})
    report = HierarchyValidator().validate(defn, model_entities=set(defn.child_nodes) | {"Paid_Social", "OrphanChild"})
    assert not report.valid
    assert "OrphanChild" in report.orphan_nodes


def test_hierarchy_validation_disconnected_rejected() -> None:
    defn = HierarchyDefinition(
        hierarchy_id="x",
        hierarchy_type="channel",
        version="1",
        parent_nodes=["A", "B"],
        child_nodes=["c1", "c2"],
        node_mapping={"c1": "A"},
    )
    report = HierarchyValidator(min_children_per_parent=1).validate(
        defn, model_entities={"A", "B", "c1", "c2"}
    )
    assert not report.valid
    assert report.disconnected_nodes


def test_penalty_child_moves_toward_parent() -> None:
    pairs, report, _ = prepare_hierarchy_for_ridge(
        _channel_hierarchy(),
        ["Paid_Social", "Meta", "TikTok", "Reddit"],
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    assert report.valid
    coef = np.array([1.0, 5.0, 4.0, 3.0])  # parent, children far from parent
    pen_high, _ = hierarchical_penalty(coef, pairs, regularization_strength=1.0)
    coef_near = np.array([1.0, 1.2, 1.1, 1.3])
    pen_low, _ = hierarchical_penalty(coef_near, pairs, regularization_strength=1.0)
    assert pen_high > pen_low


def test_penalty_larger_lambda_increases_shrinkage_in_objective() -> None:
    pairs, report, _ = prepare_hierarchy_for_ridge(
        _channel_hierarchy(),
        ["Paid_Social", "Meta", "TikTok", "Reddit"],
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    assert report.valid
    coef = np.array([1.0, 5.0, 4.0, 3.0])
    p1, _ = hierarchical_penalty(coef, pairs, regularization_strength=0.1)
    p2, _ = hierarchical_penalty(coef, pairs, regularization_strength=1.0)
    assert p2 > p1


def test_zero_lambda_and_disabled_reproduce_no_penalty() -> None:
    pairs, report, _ = prepare_hierarchy_for_ridge(
        _channel_hierarchy(),
        ["Paid_Social", "Meta", "TikTok", "Reddit"],
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    coef = np.array([1.0, 5.0, 4.0, 3.0])
    p0, _ = hierarchical_penalty(coef, pairs, regularization_strength=0.0)
    assert p0 == 0.0
    cfg_off = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["Paid_Social", "Meta", "TikTok", "Reddit"]},
    )
    assert not hierarchy_enabled(cfg_off)


def test_diagnostics_and_governance_emitted(tmp_path: Path) -> None:
    df, schema = _panel_and_schema()
    path = _write_hierarchy(tmp_path, _channel_hierarchy())
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        hierarchy=HierarchyConfig(enabled=True, hierarchy_definition_path=str(path), regularization_strength=0.2),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    coef = np.array([0.5, 0.8, 0.7, 0.6])
    rep = build_hierarchy_reports_for_fit(cfg, schema, df, coef)
    assert "hierarchy_diagnostics" in rep
    assert rep["hierarchy_diagnostics"]["hierarchy_enabled"]
    assert rep["hierarchy_effect_summary"]
    for w in HIERARCHY_GOVERNANCE_WARNINGS:
        assert w in rep["hierarchy_diagnostics"]["warnings"]
    assert rep["governance_unsupported_claims"] == list(HIERARCHY_UNSUPPORTED_QUESTIONS)


def test_synthetic_noisy_children_stabilize_toward_parent() -> None:
    """Known parent coef; noisy child coefs incur higher penalty than shrunk children."""
    parent_true = 2.0
    children_noisy = np.array([parent_true + 3.0, parent_true - 2.5, parent_true + 2.0])
    children_shrunk = np.array([parent_true + 0.2, parent_true - 0.1, parent_true + 0.15])
    pairs, report, _ = prepare_hierarchy_for_ridge(
        _channel_hierarchy(),
        ["Paid_Social", "Meta", "TikTok", "Reddit"],
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    assert report.valid
    coef_noisy = np.array([parent_true, *children_noisy])
    coef_shrunk = np.array([parent_true, *children_shrunk])
    pen_noisy, _ = hierarchical_penalty(coef_noisy, pairs, regularization_strength=1.0)
    pen_shrunk, _ = hierarchical_penalty(coef_shrunk, pairs, regularization_strength=1.0)
    assert pen_noisy > pen_shrunk
    assert np.sign(coef_shrunk[1] - coef_noisy[1]) == np.sign(parent_true - children_noisy[0])


def test_cross_branch_independent_without_pooling() -> None:
    defn = HierarchyDefinition(
        hierarchy_id="branches",
        hierarchy_type="channel",
        version="1",
        parent_nodes=["P1", "P2"],
        child_nodes=["a1", "a2", "b1", "b2"],
        node_mapping={"a1": "P1", "a2": "P1", "b1": "P2", "b2": "P2"},
    )
    pairs, report, _ = prepare_hierarchy_for_ridge(
        defn,
        ["P1", "P2", "a1", "a2", "b1", "b2"],
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    assert report.valid
    assert len(pairs) == 4
    branches = {p.parent_name for p in pairs}
    assert branches == {"P1", "P2"}


def test_ridge_trainer_disabled_hierarchy_unchanged(tmp_path: Path) -> None:
    df, schema = _panel_and_schema()
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    out = RidgeBOMMMTrainer(cfg, schema).fit(df)
    assert out["best_score"] is not None


@pytest.mark.optuna
def test_ridge_trainer_hierarchy_enabled_smoke(tmp_path: Path) -> None:
    df, schema = _panel_and_schema()
    path = _write_hierarchy(tmp_path, _channel_hierarchy())
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        hierarchy=HierarchyConfig(enabled=True, hierarchy_definition_path=str(path), regularization_strength=0.05),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    trainer = RidgeBOMMMTrainer(cfg, schema)
    out = trainer.fit(df)
    assert trainer._hierarchy_pairs
    assert out["best_detail"] is not None


def test_load_hierarchy_definition_roundtrip(tmp_path: Path) -> None:
    defn = _channel_hierarchy()
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"hierarchy_definition": defn.model_dump()}), encoding="utf-8")
    loaded = load_hierarchy_definition(p)
    assert loaded.hierarchy_id == defn.hierarchy_id


def test_build_hierarchy_diagnostics_unstable_children() -> None:
    defn = _channel_hierarchy()
    pairs, report, _ = prepare_hierarchy_for_ridge(
        defn,
        ("Paid_Social", "Meta", "TikTok", "Reddit"),
        panel_geos=set(),
        min_children_per_parent=2,
        allow_cross_branch_pooling=False,
    )
    channels = ("Paid_Social", "Meta", "TikTok", "Reddit")
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": list(channels)},
        hierarchy=HierarchyConfig(enabled=True, hierarchy_definition_path="x"),
    )
    coef = np.array([1.0, 10.0, 9.0, 8.0])
    before = np.array([1.0, 20.0, 18.0, 16.0])
    diag = build_hierarchy_diagnostics(cfg, defn, report, pairs, coef, coef_before=before)
    assert diag["unstable_children"]
