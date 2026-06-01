"""Bayes-H3 research sandbox guardrails (audit P0)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmm.config.schema import Framework, RunEnvironment
from mmm.governance.policy import PolicyError, RuntimePolicy, require_bayesian_block
from mmm.research.bayes_h3_sandbox import (
    RESEARCH_ONLY_LABEL,
    apply_research_only_envelope,
    assert_not_production_decision_surface,
    assert_optimizer_input_not_bayes_sandbox,
    build_diagnostic_trust_stub,
    reject_if_prod_decisioning_flags,
    run_sandbox_fit,
    validate_research_only_artifact,
    wrap_sandbox_artifact,
)
from mmm.research.bayes_h3_sandbox.fencing import BayesSandboxGuardError
from mmm.research.bayes_h3_sandbox.labels import ResearchOnlyLabelError, assert_diagnostic_only_outputs


def _minimal_fit_out() -> dict:
    return {
        "idata": None,
        "ppc": {},
        "linear_coef_draws": None,
        "bayesian_hierarchy_report": {"enabled": False},
    }


def test_research_only_envelope_required_fields() -> None:
    art = apply_research_only_envelope({})
    validate_research_only_artifact(art)
    assert art["label"] == RESEARCH_ONLY_LABEL
    assert art["research_only"] is True
    assert art["decision_grade"] is False
    assert art["approved_for_prod"] is False
    assert art["prod_decisioning_allowed"] is False


def test_missing_label_fails() -> None:
    with pytest.raises(ResearchOnlyLabelError, match="missing required"):
        validate_research_only_artifact({"research_only": True})


def test_approved_for_prod_true_fails() -> None:
    bad = apply_research_only_envelope({})
    bad["approved_for_prod"] = True
    with pytest.raises(ResearchOnlyLabelError, match="approved_for_prod"):
        validate_research_only_artifact(bad)


def test_prod_decisioning_allowed_true_fails() -> None:
    bad = apply_research_only_envelope({})
    bad["prod_decisioning_allowed"] = True
    with pytest.raises(ResearchOnlyLabelError, match="prod_decisioning_allowed"):
        validate_research_only_artifact(bad)


def test_wrong_label_fails() -> None:
    bad = apply_research_only_envelope({})
    bad["label"] = "PRODUCTION READY"
    with pytest.raises(ResearchOnlyLabelError, match="label must"):
        validate_research_only_artifact(bad)


def test_wrap_sandbox_artifact_includes_diagnostic_trust() -> None:
    art = wrap_sandbox_artifact(_minimal_fit_out())
    assert "diagnostic_trust_report" in art
    assert art["diagnostic_trust_report"]["trust_report_kind"] == "bayes_h3_diagnostic_stub"
    assert "posterior_summary" in art["diagnostic_trust_report"]
    assert "convergence_diagnostics" in art["diagnostic_trust_report"]
    assert "hierarchy_evidence" in art["diagnostic_trust_report"]
    assert "pooling_diagnostics" in art["diagnostic_trust_report"]


def test_cannot_emit_production_decision_surface() -> None:
    art = wrap_sandbox_artifact(_minimal_fit_out())
    art["production_decision_surface"] = True
    with pytest.raises(ResearchOnlyLabelError, match="production_decision_surface"):
        assert_not_production_decision_surface(art)


def test_cannot_produce_production_recommendation() -> None:
    art = wrap_sandbox_artifact(_minimal_fit_out())
    art["production_recommendation"] = True
    with pytest.raises(BayesSandboxGuardError, match="recommendation"):
        from mmm.research.bayes_h3_sandbox.fencing import assert_no_production_recommendation

        assert_no_production_recommendation(art)


def test_optimizer_rejects_sandbox_posterior() -> None:
    art = wrap_sandbox_artifact(_minimal_fit_out())
    with pytest.raises(BayesSandboxGuardError, match="optimizer"):
        assert_optimizer_input_not_bayes_sandbox(art, context="optimize_budget_via_simulation")


def test_optimizer_rejects_prod_bayes_sandbox_config() -> None:
    cfg = SimpleNamespace(
        framework=Framework.BAYESIAN,
        run_environment=RunEnvironment.PROD,
    )
    art = wrap_sandbox_artifact(_minimal_fit_out())
    with pytest.raises(BayesSandboxGuardError, match="Bayes sandbox"):
        assert_optimizer_input_not_bayes_sandbox(art, config=cfg, context="optimize_budget_via_simulation")


def test_posterior_outputs_marked_diagnostic_only() -> None:
    art = wrap_sandbox_artifact(_minimal_fit_out())
    assert art["outputs_are_diagnostic_only"] is True
    reject_if_prod_decisioning_flags(art)
    assert_diagnostic_only_outputs(art)


def test_prod_decide_blocks_bayesian_framework() -> None:
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


def test_diagnostic_trust_stub_is_research_only() -> None:
    stub = build_diagnostic_trust_stub(posterior_summary={"mean": 0.1})
    validate_research_only_artifact(stub)


@pytest.mark.pymc
@pytest.mark.slow
def test_pymc_trainer_fit_applies_research_envelope() -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    import numpy as np
    import pandas as pd

    from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
    from mmm.data.schema import PanelSchema
    from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer

    rng = np.random.default_rng(0)
    n = 40
    x1 = rng.uniform(1, 5, size=n)
    x2 = rng.uniform(1, 5, size=n)
    eps = rng.normal(0, 0.05, size=n)
    y = np.exp(0.3 + 0.15 * x1 + 0.1 * x2 + eps)
    df = pd.DataFrame({"g": ["A"] * n, "w": np.arange(n), "y": y, "m1": x1, "m2": x2})
    schema = PanelSchema("g", "w", "y", ("m1", "m2"))
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["m1", "m2"],
            "control_columns": [],
        },
        bayesian={
            "backend": BayesianBackend.PYMC,
            "draws": 40,
            "tune": 40,
            "chains": 2,
            "target_accept": 0.85,
            "nuts_seed": 1,
            "prior_predictive_draws": 20,
            "posterior_predictive_draws": 20,
        },
    )
    out = BayesianMMMTrainer(cfg, schema).fit(df)
    validate_research_only_artifact(out)
    assert out.get("bayes_h3_sandbox") is True
    assert out["approved_for_prod"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_run_sandbox_fit_entrypoint_wraps_output() -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    import numpy as np
    import pandas as pd

    from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
    from mmm.data.schema import PanelSchema

    rng = np.random.default_rng(0)
    n = 40
    x1 = rng.uniform(1, 5, size=n)
    x2 = rng.uniform(1, 5, size=n)
    eps = rng.normal(0, 0.05, size=n)
    y = np.exp(0.3 + 0.15 * x1 + 0.1 * x2 + eps)
    df = pd.DataFrame({"g": ["A"] * n, "w": np.arange(n), "y": y, "m1": x1, "m2": x2})
    schema = PanelSchema("g", "w", "y", ("m1", "m2"))
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["m1", "m2"],
            "control_columns": [],
        },
        bayesian={
            "backend": BayesianBackend.PYMC,
            "draws": 40,
            "tune": 40,
            "chains": 2,
            "target_accept": 0.85,
            "nuts_seed": 1,
            "prior_predictive_draws": 20,
            "posterior_predictive_draws": 20,
        },
    )
    art = run_sandbox_fit(cfg, schema, df)
    assert art["sandbox_entrypoint"] == "mmm.research.bayes_h3_sandbox.run_sandbox_fit"
    validate_research_only_artifact(art)
