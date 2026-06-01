"""Bayes-H3 research sandbox MVP fit (diagnostic hierarchical prototype)."""

from __future__ import annotations

import pytest

from mmm.research.bayes_h3_sandbox import (
    RESEARCH_ONLY_LABEL,
    run_sandbox_fit,
    validate_research_only_artifact,
    wrap_sandbox_artifact,
)
from mmm.research.bayes_h3_sandbox.fencing import BayesSandboxGuardError, assert_optimizer_input_not_bayes_sandbox
from mmm.research.bayes_h3_sandbox.fixtures import (
    TOY_CALIBRATION_SIGNAL_STUB,
    TOY_GEO_HIERARCHY,
    toy_sandbox_bundle,
    toy_sandbox_panel,
)
from mmm.research.bayes_h3_sandbox.model import MODEL_KIND, fit_h3_sandbox_hierarchical


def test_toy_fixture_is_deterministic() -> None:
    a = toy_sandbox_panel(seed=7)
    b = toy_sandbox_panel(seed=7)
    c = toy_sandbox_panel(seed=8)
    assert a.equals(b)
    assert not a.equals(c)
    assert set(a["geo_id"]) == {"dma_a", "dma_b"}
    assert list(a.columns) == ["geo_id", "week", "y", "tv", "search"]


def test_mvp_wrapped_artifact_has_diagnostic_fields() -> None:
    raw = {
        "model_kind": MODEL_KIND,
        "posterior_summary": {"model_kind": MODEL_KIND, "n_obs": 24},
        "convergence_diagnostics": {"rhat_max": 1.01},
        "hierarchy_evidence_diagnostics": {"h2d_alignment": "partial_pooling_beta_gc"},
        "pooling_diagnostics": {"pooling_mode": "partial"},
        "calibration_signal_slots": {"reserved": True, "signals": []},
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "decision_surface": None,
        "optimizer_ready_curves": None,
        "budget_recommendation": None,
    }
    art = wrap_sandbox_artifact(raw)
    validate_research_only_artifact(art)
    assert art["label"] == RESEARCH_ONLY_LABEL
    assert art["posterior_summary"]["n_obs"] == 24
    assert art["diagnostic_trust_report"]["trust_report_kind"] == "bayes_h3_diagnostic_stub"
    assert "convergence_diagnostics" in art["diagnostic_trust_report"]
    assert art.get("decision_surface") is None
    assert art.get("production_decision_surface") is False
    assert art.get("budget_recommendation") is None
    assert art.get("optimizer_ready_curves") is None
    with pytest.raises(BayesSandboxGuardError, match="optimizer"):
        assert_optimizer_input_not_bayes_sandbox(art)


@pytest.mark.pymc
@pytest.mark.slow
def test_run_sandbox_fit_mvp_hierarchical_sample() -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    cfg, schema, df = toy_sandbox_bundle(fast_mcmc=True)
    art = run_sandbox_fit(
        cfg,
        schema,
        df,
        geo_hierarchy_mapping=TOY_GEO_HIERARCHY,
        calibration_signals_stub=TOY_CALIBRATION_SIGNAL_STUB,
    )
    validate_research_only_artifact(art)
    assert art["model_kind"] == MODEL_KIND
    assert art["sandbox_entrypoint"] == "mmm.research.bayes_h3_sandbox.run_sandbox_fit"
    assert art["approved_for_prod"] is False
    assert art["prod_decisioning_allowed"] is False
    ps = art["posterior_summary"]
    assert ps["n_geo"] == 2
    assert "tv" in ps["mu_channel_mean"]
    assert art["convergence_diagnostics"]["rhat_max"] == art["convergence_diagnostics"]["rhat_max"]
    assert art["hierarchy_evidence_diagnostics"]["geo_hierarchy_mapping"] == TOY_GEO_HIERARCHY
    assert art["calibration_signal_slots"]["reserved"] is True
    assert art["diagnostic_trust_report"]["posterior_summary"]["model_kind"] == MODEL_KIND
    rh = float(art["convergence_diagnostics"].get("rhat_max", float("nan")))
    assert rh == rh, "convergence diagnostics should report rhat_max"


@pytest.mark.pymc
@pytest.mark.slow
def test_fit_h3_sandbox_hierarchical_direct() -> None:
    pytest.importorskip("pymc")
    cfg, schema, df = toy_sandbox_bundle(fast_mcmc=True)
    raw = fit_h3_sandbox_hierarchical(
        cfg,
        schema,
        df,
        geo_hierarchy_mapping=TOY_GEO_HIERARCHY,
        calibration_signals_stub=TOY_CALIBRATION_SIGNAL_STUB,
    )
    assert raw["outputs_are_diagnostic_only"] is True
    assert raw["production_decision_surface"] is False
    assert "mu_channel_mean" in raw["pooling_diagnostics"]
