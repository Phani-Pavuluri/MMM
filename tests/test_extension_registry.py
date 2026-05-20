"""Extension registry: ordering, failures, and post-refactor report parity."""

from __future__ import annotations

import json

import pytest

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import (
    ExtensionDependencyError,
    ExtensionRegistrationError,
    ExtensionRegistry,
    ExtensionSpec,
    get_extension_registry,
    reset_extension_registry_for_tests,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _ridge_extension_report() -> dict:
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=3, n_weeks=50))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    return run_post_fit_extensions(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out=fit,
        yhat=yhat,
        store=None,
    )


def test_optimization_gate_always_runs_when_gates_disabled_non_prod() -> None:
    """Registry must not skip optimization_gate; disabled is handled inside OptimizationSafetyGate."""
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=3, n_weeks=50))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        run_environment=RunEnvironment.RESEARCH,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        extensions={"optimization_gates": {"enabled": False}},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    rep = run_post_fit_extensions(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out=fit,
        yhat=tr.predict(df),
        store=None,
    )
    og = rep["decision_bundle"]["optimization_gate"]
    assert og["allowed"] is True
    assert "gates_disabled" in og["reasons"]
    assert rep["decision_bundle"]["decision_safety_flags"]["optimization_gate_allowed"] is True
    assert rep["operational_health"]["inputs"]["optimization_gate_allowed"] is True


def test_extension_report_keys_stable_across_two_runs() -> None:
    a = _ridge_extension_report()
    b = _ridge_extension_report()
    assert sorted(a.keys()) == sorted(b.keys())


def test_extension_report_top_level_keys_regression() -> None:
    rep = _ridge_extension_report()
    keys = frozenset(rep.keys())
    expected = frozenset(
        {
                "artifact_backend",
                "artifact_tier_disclosure",
                "control_governance",
            "baselines",
            "calibration_summary",
            "replay_calibration_sensitivity",
            "replay_holdout_validation",
            "curve_bundle",
            "curve_bundles",
                "curve_decision_alignment",
                "curve_decision_alignment_matrix",
            "curve_safe_for_optimization",
            "curve_stress",
            "data_fingerprint",
            "drift_report",
            "decision_bundle",
            "decision_policy",
            "decision_uncertainty",
            "economics_contract",
            "economics_output_metadata",
            "experiment_scheduler_report",
            "feature_separability_report",
            "falsification",
            "geo_spillover",
            "governance",
            "guidance",
            "identifiability",
            "lag_diagnostics",
            "model_card_md",
            "model_release",
            "operational_health",
            "performance_report",
            "panel_qa",
            "post_fit_validation",
            "response_diagnostics",
            "ridge_fit_summary",
            "roi_summary",
            "run_manifest",
            "seed_resolution",
            "transform_policy",
            "uncertainty_decomposition",
            "data_quality",
        }
    )
    assert keys == expected


def test_duplicate_registration_fails() -> None:
    reg = ExtensionRegistry()

    def _noop(_ctx: ExtensionContext) -> None:
        return None

    reg.register(
        ExtensionSpec(name="dup", artifact_key="dup_a", priority=1, run=_noop, report_keys=()),
    )
    with pytest.raises(ExtensionRegistrationError, match="already registered"):
        reg.register(
            ExtensionSpec(name="dup", artifact_key="dup_b", priority=2, run=_noop, report_keys=()),
        )


def test_duplicate_artifact_key_raises() -> None:
    reg = ExtensionRegistry()

    def _noop(_ctx: ExtensionContext) -> None:
        return None

    reg.register(
        ExtensionSpec(name="one", artifact_key="shared_key", priority=1, run=_noop, report_keys=()),
    )
    with pytest.raises(ExtensionRegistrationError, match="artifact_key"):
        reg.register(
            ExtensionSpec(name="two", artifact_key="shared_key", priority=2, run=_noop, report_keys=()),
        )


def test_missing_dependency_fails_at_run() -> None:
    reg = ExtensionRegistry()

    def _noop(_ctx: ExtensionContext) -> None:
        return None

    reg.register(
        ExtensionSpec(
            name="orphan",
            artifact_key="orphan",
            priority=5,
            dependencies=("missing_parent",),
            run=_noop,
            report_keys=(),
        )
    )
    ctx = ExtensionContext(
        panel=__import__("pandas").DataFrame(),
        panel_s=__import__("pandas").DataFrame(),
        schema=__import__("mmm.data.schema", fromlist=["PanelSchema"]).PanelSchema(
            "g", "w", "y", ("c1",)
        ),
        config=MMMConfig(data={"channel_columns": ["c1"]}),
        fit_out={},
        yhat=__import__("numpy").zeros(0),
        store=None,
        out={},
        rng=__import__("numpy").random.default_rng(0),
        ext=MMMConfig(data={"channel_columns": ["c1"]}).extensions,
        seed_resolution={},
    )
    with pytest.raises(ExtensionDependencyError, match="missing_parent"):
        reg.run_all(ctx)


def test_disabled_extension_skipped_via_config_flag() -> None:
    reg = ExtensionRegistry()
    calls: list[str] = []

    def _mark(ctx: ExtensionContext) -> None:
        ctx.out["flag"] = True

    reg.register(
        ExtensionSpec(
            name="parent",
            artifact_key="parent",
            priority=1,
            run=lambda ctx: calls.append("parent"),
            report_keys=(),
        )
    )
    reg.register(
        ExtensionSpec(
            name="optional_child",
            artifact_key="flag",
            priority=2,
            dependencies=("parent",),
            config_key="extensions.optimization_gates.enabled",
            run=_mark,
            report_keys=("flag",),
        ),
    )
    cfg = MMMConfig(
        data={"channel_columns": ["c1"]},
        extensions={"optimization_gates": {"enabled": False}},
    )
    ctx = ExtensionContext(
        panel=__import__("pandas").DataFrame(),
        panel_s=__import__("pandas").DataFrame(),
        schema=__import__("mmm.data.schema", fromlist=["PanelSchema"]).PanelSchema(
            "g", "w", "y", ("c1",)
        ),
        config=cfg,
        fit_out={},
        yhat=__import__("numpy").zeros(0),
        store=None,
        out={},
        rng=__import__("numpy").random.default_rng(0),
        ext=cfg.extensions,
        seed_resolution={},
    )
    reg.run_all(ctx)
    assert "flag" not in ctx.out


def test_execution_order_by_priority() -> None:
    reg = ExtensionRegistry()
    order: list[str] = []

    def make(name: str, priority: int, deps: tuple[str, ...] = ()) -> ExtensionSpec:
        return ExtensionSpec(
            name=name,
            artifact_key=name,
            priority=priority,
            dependencies=deps,
            run=lambda _ctx, n=name: order.append(n),
            report_keys=(),
        )

    reg.register(make("late", 30, ("early",)))
    reg.register(make("early", 10))
    ctx = ExtensionContext(
        panel=__import__("pandas").DataFrame(),
        panel_s=__import__("pandas").DataFrame(),
        schema=__import__("mmm.data.schema", fromlist=["PanelSchema"]).PanelSchema(
            "g", "w", "y", ("c1",)
        ),
        config=MMMConfig(data={"channel_columns": ["c1"]}),
        fit_out={},
        yhat=__import__("numpy").zeros(0),
        store=None,
        out={},
        rng=__import__("numpy").random.default_rng(0),
        ext=MMMConfig(data={"channel_columns": ["c1"]}).extensions,
        seed_resolution={},
    )
    reg.run_all(ctx)
    assert order == ["early", "late"]


def test_registry_specs_deterministic() -> None:
    a = [s.name for s in get_extension_registry().specs_sorted()]
    reset_extension_registry_for_tests()
    b = [s.name for s in get_extension_registry().specs_sorted()]
    assert a == b


def test_experiment_scheduler_disabled_still_emits_skipped_report() -> None:
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=3, n_weeks=50))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        extensions={"experiment_scheduler": {"enabled": False}},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    rep = run_post_fit_extensions(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out=fit,
        yhat=tr.predict(df),
        store=None,
    )
    sched = rep["experiment_scheduler_report"]
    assert sched.get("skipped") is True


def test_builtin_registry_validate_passes() -> None:
    reg = reset_extension_registry_for_tests()
    reg.validate()
    names = reg.names()
    assert "governance" in names
    assert "curve_decision_alignment" in names
    assert names == tuple(sorted(names))


def test_extension_report_json_roundtrip_stable_keys() -> None:
    rep = _ridge_extension_report()
    k1 = sorted(rep.keys())
    blob = json.dumps(rep, sort_keys=True, default=str)
    k2 = sorted(json.loads(blob).keys())
    assert k1 == k2
