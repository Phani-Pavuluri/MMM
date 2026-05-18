"""Planning scenario contract, assumptions metadata, and optimize fixed non-media world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.cli import main as cli_main
from mmm.config.extensions import ExtensionSuiteConfig, PlanningPolicyConfig
from mmm.config.schema import BudgetConfig, CVConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.decision.gates import allow_decision_pipeline
from mmm.decision.service import _scenario_lineage_for_optimize_without_scenario, simulate_decision
from mmm.governance.policy import PolicyError
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.optimization.budget import simulation_optimizer as sim_opt
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.assumptions import build_planning_assumptions, infer_controls_assumption
from mmm.planning.context import ridge_context_from_fit
from mmm.planning.control_overlay import ControlOverlaySpec, overlay_spec_sha256
from mmm.planning.optimize_context import OptimizeNonMediaContext
from mmm.planning.policy import evaluate_control_scenario_policy
from mmm.planning.scenario import planning_scenario_from_dict
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _ridge_ctx_with_promo():
    df0, _ = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=16, channels=("a",), betas=(0.4,))
    )
    df = df0.copy()
    df["promo"] = 0.0
    schema = PanelSchema(
        geo_column="geo",
        week_column="week",
        target_column="y",
        channel_columns=("a",),
        control_columns=("promo",),
    )
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="geo",
            week_column="week",
            target_column="y",
            channel_columns=["a"],
            control_columns=["promo"],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
        extensions=ExtensionSuiteConfig(
            planning_policy=PlanningPolicyConfig(
                promo_columns=["promo"],
                name_heuristic_warnings=False,
            )
        ),
    )
    fit = RidgeBOMMMTrainer(cfg, schema).fit(df)
    ctx = ridge_context_from_fit(df, schema, cfg, fit)
    return ctx, df, schema, cfg


def test_simulate_no_overlay_controls_observed() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    bau = bau_baseline_from_panel(df, schema)
    sim = simulate(dict(bau.spend_by_channel), ctx, baseline_plan=bau, uncertainty_mode="point")
    pa = sim.extra["planning_assumptions"]
    assert pa["controls_assumption"] == "observed"
    assert pa["media_assumption"] == "constant"
    assert pa["world_assumption"] == "historical_panel"


def test_simulate_with_overlay_controls_overlay() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    bau = bau_baseline_from_panel(df, schema)
    wk = df[schema.week_column].iloc[0]
    geo = str(df[schema.geo_column].iloc[0])
    ov = ControlOverlaySpec.from_dict(
        {"overrides": [{"geo": geo, "week": wk, "column": "promo", "value": 1.0}]}
    )
    sim = simulate(
        dict(bau.spend_by_channel),
        ctx,
        baseline_plan=bau,
        control_overlay_plan=ov,
        scenario_lineage={"scenario_id": "t1"},
    )
    assert sim.extra["planning_assumptions"]["controls_assumption"] == "overlay"


def test_overlay_missing_column_fails() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    bau = bau_baseline_from_panel(df, schema)
    ov = ControlOverlaySpec.from_dict(
        {
            "overrides": [
                {"geo": "missing", "week": 1, "column": "promo", "value": 1.0},
            ]
        }
    )
    with pytest.raises(ValueError, match="matched no rows"):
        simulate(dict(bau.spend_by_channel), ctx, baseline_plan=bau, control_overlay_plan=ov)


def test_planning_scenario_hash_deterministic() -> None:
    raw = {
        "scenario_id": "promo_q1",
        "media": {"candidate_spend": {"a": 10.0}},
        "controls": {
            "control_overlay_plan": {
                "overrides": [{"geo": "G1", "week": 1, "column": "promo", "value": 1.0}],
            }
        },
    }
    s1 = planning_scenario_from_dict(raw)
    s2 = planning_scenario_from_dict(raw)
    assert s1.scenario_hash() == s2.scenario_hash()


def test_strict_prod_blocks_observed_sensitive_controls(tmp_path: Path) -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    cfg = cfg.model_copy(
        update={
            "run_environment": RunEnvironment.PROD,
            "extensions": cfg.extensions.model_copy(
                update={
                    "planning_policy": cfg.extensions.planning_policy.model_copy(
                        update={"strict_prod_requires_explicit_control_scenario": True}
                    )
                }
            ),
        }
    )
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"path": str(csv)})})
    ext = tmp_path / "ext.json"
    art = RidgeBOMMMTrainer(cfg, schema).fit(df)["artifacts"]
    ext.write_text(
        json.dumps(
            {
                "ridge_fit_summary": {
                    "best_params": dict(art.best_params),
                    "coef": art.coef.tolist(),
                    "intercept": art.intercept.tolist(),
                },
                "governance": {"approved_for_optimization": True},
                "response_diagnostics": {"safe_for_optimization": True},
                "identifiability": {"identifiability_score": 0.1},
                "panel_qa": {"max_severity": "info", "issues": []},
                "model_release": {"state": "planning_allowed", "reasons": [], "triggers": {}},
                "experiment_matching": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    scen = tmp_path / "scen.yaml"
    scen.write_text("candidate_spend:\n  a: 12.0\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="explicit"):
        simulate_decision(
            cfg=cfg,
            scenario={"candidate_spend": {"a": 12.0}},
            extension_report=json.loads(ext.read_text()),
            out=tmp_path / "out.json",
        )


def test_optimize_fixed_scenario_same_overlay_each_eval() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    wk = df[schema.week_column].iloc[0]
    geo = str(df[schema.geo_column].iloc[0])
    ov = ControlOverlaySpec.from_dict(
        {"overrides": [{"geo": geo, "week": wk, "column": "promo", "value": 1.0}]}
    )
    cfg = cfg.model_copy(
        update={
            "budget": BudgetConfig(
                total_budget=20.0,
                channel_min={"a": 0.0},
                channel_max={"a": 20.0},
            )
        }
    )
    ctx = type(ctx)(
        panel=ctx.panel,
        schema=ctx.schema,
        config=cfg,
        best_params=ctx.best_params,
        coef=ctx.coef,
        intercept=ctx.intercept,
    )
    nm = OptimizeNonMediaContext(
        control_overlay_plan=ov,
        frozen_non_media=True,
        scenario_lineage={"scenario_id": "opt_promo"},
    )
    with allow_decision_pipeline():
        res = optimize_budget_via_simulation(
            ctx,
            current_spend=np.array([10.0]),
            total_budget=20.0,
            channel_min=np.array([0.0]),
            channel_max=np.array([20.0]),
            non_media=nm,
        )
    sim_at = res.get("simulation_at_recommendation") or {}
    assert sim_at.get("planning_assumptions", {}).get("controls_assumption") == "frozen_scenario"
    assert sim_at.get("planning_assumptions", {}).get("media_assumption") == "optimized"


def test_infer_controls_assumption_frozen() -> None:
    got = infer_controls_assumption(
        has_baseline_overlay=False,
        has_plan_overlay=True,
        frozen_non_media=True,
    )
    assert got == "frozen_scenario"


def test_prod_bundle_requires_planning_assumptions() -> None:
    cfg = MMMConfig(run_environment=RunEnvironment.PROD, data={"channel_columns": ["a"], "control_columns": []})
    schema = PanelSchema("g", "w", "y", ("a",))
    fp = {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}
    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        artifact_tier="decision",
    )
    miss = validate_prod_decision_bundle(bundle, run_environment=RunEnvironment.PROD, decision_cli_surface=True)
    assert any("planning_assumptions" in m for m in miss)


def test_explicit_scenario_requires_lineage_hash() -> None:
    cfg = MMMConfig(run_environment=RunEnvironment.PROD, data={"channel_columns": ["a"], "control_columns": []})
    schema = PanelSchema("g", "w", "y", ("a",))
    fp = {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}
    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        artifact_tier="decision",
        planning_assumptions=build_planning_assumptions(
            controls_assumption="overlay",
            media_assumption="constant",
            world_assumption="explicit_scenario",
        ),
        scenario_lineage={},
    )
    miss = validate_prod_decision_bundle(bundle, run_environment=RunEnvironment.PROD, decision_cli_surface=True)
    assert any("scenario_lineage" in m for m in miss)


def test_overlay_spec_sha256_stable() -> None:
    ov = ControlOverlaySpec.from_dict(
        {"overrides": [{"geo": "G1", "week": 1, "column": "promo", "value": 1.0}]}
    )
    h1 = overlay_spec_sha256(ov)
    h2 = overlay_spec_sha256(ov)
    assert h1 == h2
    assert h1 is not None and len(h1) == 64


def test_scenario_lineage_includes_overlay_hashes() -> None:
    raw = {
        "scenario_id": "promo_q1",
        "controls": {
            "control_overlay_plan": {
                "overrides": [{"geo": "G1", "week": 1, "column": "promo", "value": 1.0}],
            }
        },
    }
    ps = planning_scenario_from_dict(raw)
    lin = ps.lineage_payload()
    assert lin.get("plan_overlay_spec_sha256")
    assert lin.get("scenario_hash")


def test_policy_warning_on_observed_sensitive_controls() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    pol = evaluate_control_scenario_policy(cfg, controls_assumption="observed")
    assert pol.severity == "warning"
    assert pol.messages
    assert "promo" in pol.sensitive_columns_matched


def test_optimizer_passes_overlay_on_each_simulate_call() -> None:
    ctx, df, schema, cfg = _ridge_ctx_with_promo()
    wk = df[schema.week_column].iloc[0]
    geo = str(df[schema.geo_column].iloc[0])
    ov = ControlOverlaySpec.from_dict(
        {"overrides": [{"geo": geo, "week": wk, "column": "promo", "value": 1.0}]}
    )
    nm = OptimizeNonMediaContext(control_overlay_plan=ov, frozen_non_media=True)
    calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return sim_opt.simulate(*args, **kwargs)

    cfg = cfg.model_copy(
        update={
            "budget": BudgetConfig(
                total_budget=20.0,
                channel_min={"a": 0.0},
                channel_max={"a": 20.0},
            ),
            "extensions": cfg.extensions.model_copy(
                update={
                    "product": cfg.extensions.product.model_copy(
                        update={"simulation_optimizer_n_starts": 1}
                    )
                }
            ),
        }
    )
    ctx = type(ctx)(
        panel=ctx.panel,
        schema=ctx.schema,
        config=cfg,
        best_params=ctx.best_params,
        coef=ctx.coef,
        intercept=ctx.intercept,
    )
    with patch.object(sim_opt, "simulate", side_effect=_capture):
        sim_opt.optimize_budget_via_simulation(
            ctx,
            current_spend=np.array([10.0]),
            total_budget=20.0,
            channel_min=np.array([0.0]),
            channel_max=np.array([20.0]),
            non_media=nm,
        )
    assert len(calls) >= 2
    for kw in calls:
        assert kw.get("frozen_non_media_controls") is True
        assert kw.get("control_overlay_plan") is not None


def test_optimize_without_scenario_lineage_explicit() -> None:
    lin = _scenario_lineage_for_optimize_without_scenario()
    assert lin["non_media_overlay_supplied"] is False
    assert lin["non_media_overlay_applied"] is False


def test_cli_help_mentions_planning_assumptions() -> None:
    r = CliRunner().invoke(cli_main.app, ["decide", "optimize-budget", "--help"])
    assert r.exit_code == 0
    assert "media" in (r.stdout or "").lower()
    assert "promo" in (r.stdout or "").lower() or "non-media" in (r.stdout or "").lower()
