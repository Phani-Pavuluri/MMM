"""PR 4B: Bayesian hierarchy (research-only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.config.schema import (
    BayesianBackend,
    Framework,
    HierarchyConfig,
    MMMConfig,
    ModelForm,
    PoolingMode,
)
from mmm.data.schema import PanelSchema
from mmm.governance.model_form_policy import LOG_LOG_HIERARCHY_POLICY_MESSAGE
from mmm.governance.policy import PolicyError, RuntimePolicy, require_bayesian_block
from mmm.hierarchy.bayesian_hierarchy import (
    build_bayesian_hierarchy_report,
    prepare_bayesian_hierarchy,
    register_bayesian_hierarchical_media_coefs,
    uses_bayesian_hierarchy,
)
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition
from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer


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


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    channels = ("Paid_Social", "Meta", "TikTok", "Reddit")
    rows = []
    for w in range(20):
        row = {"g": "A", "w": w, "y": 100.0 + w * 0.1}
        for c in channels:
            row[c] = 5.0 + w * 0.01
        rows.append(row)
    return pd.DataFrame(rows), PanelSchema("g", "w", "y", channels)


def _write_hierarchy(tmp_path: Path, definition: HierarchyDefinition) -> str:
    p = tmp_path / "hierarchy.json"
    p.write_text(definition.model_dump_json(), encoding="utf-8")
    return str(p)


def test_bayesian_hierarchy_disabled_flag() -> None:
    cfg = MMMConfig(framework=Framework.BAYESIAN, data={"channel_columns": ["a"]})
    assert not uses_bayesian_hierarchy(cfg)


def test_bayesian_hierarchy_enabled_requires_definition_path() -> None:
    with pytest.raises(ValueError, match="hierarchy_definition_path"):
        MMMConfig(
            framework=Framework.BAYESIAN,
            model_form=ModelForm.SEMI_LOG,
            data={"channel_columns": ["Paid_Social", "Meta", "TikTok", "Reddit"]},
            bayesian={"use_hierarchy": True},
        )


def test_log_log_plus_bayesian_hierarchy_fails() -> None:
    with pytest.raises(PolicyError, match=LOG_LOG_HIERARCHY_POLICY_MESSAGE):
        MMMConfig(
            framework=Framework.BAYESIAN,
            model_form=ModelForm.LOG_LOG,
            data={"channel_columns": ["Paid_Social", "Meta", "TikTok", "Reddit"]},
            hierarchy=HierarchyConfig(enabled=False, hierarchy_definition_path="/tmp/h.json"),
            bayesian={"use_hierarchy": True},
        )


def test_cyclic_hierarchy_rejected(tmp_path: Path) -> None:
    panel, schema = _panel()
    defn = _channel_hierarchy()
    defn = defn.model_copy(update={"node_mapping": {"Meta": "TikTok", "TikTok": "Meta", "Reddit": "Paid_Social"}})
    path = _write_hierarchy(tmp_path, defn)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": list(schema.channel_columns),
        },
        hierarchy=HierarchyConfig(enabled=False, hierarchy_definition_path=path),
        bayesian={"use_hierarchy": True},
    )
    with pytest.raises(ValueError, match="validation failed"):
        prepare_bayesian_hierarchy(cfg, panel, schema)


def test_valid_hierarchy_prepares_pairs(tmp_path: Path) -> None:
    panel, schema = _panel()
    path = _write_hierarchy(tmp_path, _channel_hierarchy())
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": list(schema.channel_columns),
        },
        hierarchy=HierarchyConfig(enabled=False, hierarchy_definition_path=path),
        bayesian={"use_hierarchy": True},
    )
    prep = prepare_bayesian_hierarchy(cfg, panel, schema)
    assert len(prep.pairs) == 3


@pytest.mark.pymc
def test_hierarchy_disabled_model_has_no_hier_sigma_group() -> None:
    pymc = pytest.importorskip("pymc")
    with pymc.Model() as model:
        register_bayesian_hierarchical_media_coefs(
            pymc,
            n_media=3,
            media_prior="half_normal_nonneg",
            media_sigma=0.8,
            pairs=[],
            group_sigma_prior=0.5,
        )
    assert "hier_sigma_group" not in model.named_vars


@pytest.mark.pymc
def test_valid_hierarchy_adds_hier_sigma_group(tmp_path: Path) -> None:
    pymc = pytest.importorskip("pymc")
    panel, schema = _panel()
    path = _write_hierarchy(tmp_path, _channel_hierarchy())
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        data={"channel_columns": list(schema.channel_columns)},
        hierarchy=HierarchyConfig(enabled=False, hierarchy_definition_path=path),
        bayesian={"use_hierarchy": True},
    )
    prep = prepare_bayesian_hierarchy(cfg, panel, schema)
    with pymc.Model() as model:
        register_bayesian_hierarchical_media_coefs(
            pymc,
            n_media=4,
            media_prior="half_normal_nonneg",
            media_sigma=0.8,
            pairs=prep.pairs,
            group_sigma_prior=0.5,
        )
    assert "hier_sigma_group" in model.named_vars
    assert any("beta_mu_media_child" in k for k in model.named_vars)


def test_build_report_disabled() -> None:
    from mmm.hierarchy.bayesian_hierarchy import BayesianHierarchyPrepareResult

    cfg = MMMConfig(framework=Framework.BAYESIAN, data={"channel_columns": ["a"]})
    rep = build_bayesian_hierarchy_report(cfg, BayesianHierarchyPrepareResult(), None)
    assert rep.get("enabled") is False


def test_prod_bayesian_decisioning_still_blocked() -> None:
    policy = RuntimePolicy(
        prod=True,
        require_planning_allowed=True,
        require_panel_qa_pass=True,
        require_replay_calibration=True,
        allow_bayesian_decisioning=False,
        allowed_cv_modes=["calendar"],
        allow_unsafe_decision_apis=False,
    )
    with pytest.raises(PolicyError, match="Bayesian"):
        require_bayesian_block(Framework.BAYESIAN, policy)


@pytest.mark.pymc
@pytest.mark.slow
def test_bayesian_fit_emits_hierarchy_report(tmp_path: Path) -> None:
    pytest.importorskip("arviz")
    panel, schema = _panel()
    path = _write_hierarchy(tmp_path, _channel_hierarchy())
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": list(schema.channel_columns),
        },
        hierarchy=HierarchyConfig(enabled=False, hierarchy_definition_path=path),
        bayesian={
            "backend": BayesianBackend.PYMC,
            "use_hierarchy": True,
            "draws": 30,
            "tune": 30,
            "chains": 2,
            "nuts_seed": 0,
        },
    )
    out = BayesianMMMTrainer(cfg, schema).fit(panel)
    rep = out.get("bayesian_hierarchy_report")
    assert isinstance(rep, dict)
    assert rep.get("enabled") is True
    assert rep.get("prod_decisioning_allowed") is False
    assert rep.get("parent_child_mapping")
    assert rep.get("posterior_shrinkage_summary")
    assert rep.get("group_variance_summary")
    assert rep.get("prior_posterior_overlap")
    assert any("causal" in w.lower() for w in rep.get("governance_warnings") or [])


@pytest.mark.pymc
@pytest.mark.slow
def test_bayesian_disabled_unchanged_smoke() -> None:
    """Without use_hierarchy, fit completes and report is disabled."""
    pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "g": ["A"] * n,
            "w": np.arange(n),
            "y": np.exp(0.2 + 0.1 * rng.uniform(1, 5, n) + rng.normal(0, 0.05, n)),
            "m1": rng.uniform(1, 5, n),
            "m2": rng.uniform(1, 5, n),
        }
    )
    schema = PanelSchema("g", "w", "y", ("m1", "m2"))
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["m1", "m2"],
        },
        bayesian={"draws": 20, "tune": 20, "chains": 2, "nuts_seed": 1},
    )
    out = BayesianMMMTrainer(cfg, schema).fit(df)
    rep = out.get("bayesian_hierarchy_report") or {}
    assert rep.get("enabled") is False
