"""PR 5B: robust optimization research (not prod optimize-budget)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.optimization.robust.robust_optimization_research import (
    _rank_by_objective,
    _uncertainty_inputs,
    build_robust_optimization_research,
)


def _propagation_proxy_only() -> dict:
    return {
        "uncertainty_sources": {
            "parameter_uncertainty": {"magnitude_proxy": 0.4, "present": False},
            "experiment_uncertainty": {"magnitude_proxy": 0.2, "present": False},
            "hierarchy_uncertainty": {"magnitude_proxy": 0.1, "present": False},
            "allocation_uncertainty": {"magnitude_proxy": 0.15, "present": False},
        },
        "ridge_bootstrap_summary": {"summarized": False},
        "bayesian_posterior_summary": {"summarized": False},
        "prod_monetary_ci_allowed": False,
    }


def _propagation_with_bootstrap() -> dict:
    p = _propagation_proxy_only()
    p["ridge_bootstrap_summary"] = {"summarized": True}
    return p


def test_disabled_by_default() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a", "b"]})
    rep = build_robust_optimization_research(
        cfg,
        panel=pd.DataFrame(),
        schema=PanelSchema("g", "w", "y", ("a", "b")),
        fit_out={},
        extension_report={},
    )
    assert rep["enabled"] is False
    assert rep.get("skipped") is True


def test_no_calibrated_uncertainty_warning() -> None:
    _, calibrated, warnings, _ = _uncertainty_inputs(_propagation_proxy_only())
    assert not calibrated
    assert any("calibrated" in w.lower() or "proxy" in w.lower() for w in warnings)


def test_risk_aversion_changes_ranking() -> None:
    rows = [
        {
            "candidate_id": "high_mu_high_unc",
            "expected_delta_mu": 10.0,
            "lower_confidence_bound_proxy": 4.0,
            "risk_adjusted_score": 3.0,
        },
        {
            "candidate_id": "moderate_mu_low_unc",
            "expected_delta_mu": 8.0,
            "lower_confidence_bound_proxy": 7.2,
            "risk_adjusted_score": 7.5,
        },
    ]
    assert _rank_by_objective(rows, "maximize_expected_delta_mu")[0] == "high_mu_high_unc"
    assert _rank_by_objective(rows, "maximize_risk_adjusted_score")[0] == "moderate_mu_low_unc"
    assert _rank_by_objective(rows, "maximize_lower_confidence_bound_proxy")[0] == "moderate_mu_low_unc"


def test_prod_decisioning_false() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a", "b"]},
        extensions={"robust_optimization_research": {"enabled": True}},
    )
    rep = build_robust_optimization_research(
        cfg,
        panel=pd.DataFrame(),
        schema=PanelSchema("g", "w", "y", ("a", "b")),
        fit_out={},
        extension_report={"uncertainty_propagation_report": _propagation_proxy_only()},
    )
    assert rep["prod_decisioning_allowed"] is False
    assert rep["decision_safe"] is False
    assert rep.get("recommended_prod_allocation") is None


def test_robust_report_emits_when_enabled_with_mock_simulate() -> None:
    rows = []
    for g in range(2):
        for w in range(12):
            rows.append({"geo_id": f"g{g}", "week_start_date": w, "revenue": 100.0, "c1": 5.0, "c2": 3.0})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("c1", "c2"))

    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["c1", "c2"],
        },
        budget={"total_budget": 16.0},
        extensions={
            "robust_optimization_research": {"enabled": True, "n_candidates": 4, "n_stability_scenarios": 3},
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=6, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )

    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

    fit_out = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    call_n = {"n": 0}
    mu_sequence = [0.0, 4.0, 1.5, 1.0, 0.8]

    def fake_eval(spend: dict, **kwargs: object) -> float:
        i = min(call_n["n"], len(mu_sequence) - 1)
        call_n["n"] += 1
        return float(mu_sequence[i])

    with patch(
        "mmm.optimization.robust.robust_optimization_research._evaluate_candidate",
        side_effect=fake_eval,
    ), patch(
        "mmm.optimization.robust.robust_optimization_research._stability_metrics",
        return_value={"available": True, "delta_mu_std": 0.1, "downside_gap_vs_mean": 0.2},
    ):
        rep = build_robust_optimization_research(
            cfg,
            panel=panel,
            schema=schema,
            fit_out=fit_out,
            extension_report={
                "uncertainty_propagation_report": _propagation_with_bootstrap(),
            },
        )

    assert rep.get("skipped") is not True
    assert rep["enabled"] is True
    assert "baseline_allocation" in rep
    assert "candidate_allocations" in rep
    assert "risk_return_frontier" in rep
    assert len(rep["risk_return_frontier"]) >= 2
    assert "allocation_stability" in rep
    assert "downside_risk_proxy" in rep
    assert rep["prod_optimize_budget_path_used"] is False
    assert rep["research_only"] is True


def test_decision_service_does_not_import_robust_module() -> None:
    import inspect

    import mmm.decision.service as svc

    src = inspect.getsource(svc)
    assert "robust_optimization_research" not in src
    assert "optimize_budget_via_simulation" in src


def test_optimize_budget_decision_unchanged_import() -> None:
    from mmm.decision.service import optimize_budget_decision

    assert callable(optimize_budget_decision)
