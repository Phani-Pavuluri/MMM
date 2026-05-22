"""PR 3: Bayesian experiment likelihood (research-only)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mmm.calibration.bayesian_experiment_likelihood import (
    LiftScaleMismatchError,
    compute_adjusted_standard_error,
    prepare_bayesian_experiment_likelihood_terms,
    uses_bayesian_experiment_likelihood,
    validate_experiment_lift_scale,
)
from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
from mmm.data.schema import PanelSchema
from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    ExperimentType,
    GeoGranularity,
    TimeWindow,
)
from mmm.governance.policy import PolicyError, RuntimePolicy, require_bayesian_block


def _evidence(**kwargs: object) -> ExperimentEvidence:
    d = dict(
        experiment_id="e1",
        experiment_type=ExperimentType.GEOX,
        channel="m1",
        kpi="y",
        estimand="ATT",
        lift_estimate=0.5,
        standard_error=0.1,
        spend_delta=10.0,
        metadata={"spend_multiplier": 0.9, "market": "US", "lift_scale": "log_mean_kpi_delta"},
        time_window=TimeWindow(start="0", end="20"),
        geo_scope=["A"],
        geo_granularity=GeoGranularity.GEO,
        source_system="test",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )
    d.update(kwargs)
    return ExperimentEvidence(**d)  # type: ignore[arg-type]


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    for w in range(25):
        rows.append({"g": "A", "w": w, "y": 100.0 + w * 0.1, "m1": 5.0 + w * 0.01})
    return pd.DataFrame(rows), PanelSchema("g", "w", "y", ("m1",))


def _write_registry(path: Path, *evidence: ExperimentEvidence) -> None:
    data = {
        "registry_version": "mmm_experiment_evidence_registry_v1",
        "experiments": {e.experiment_id: e.to_registry_dict() for e in evidence},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_config_disabled_no_likelihood_flag() -> None:
    cfg = MMMConfig(framework=Framework.BAYESIAN, data={"channel_columns": ["m1"]})
    assert not uses_bayesian_experiment_likelihood(cfg)


def test_enabled_without_registry_fails() -> None:
    with pytest.raises(ValueError, match="experiment_registry_path"):
        MMMConfig(
            framework=Framework.BAYESIAN,
            data={"channel_columns": ["m1"]},
            bayesian={"use_experiment_likelihood": True},
        )


def test_incompatible_evidence_excluded(tmp_path: Path) -> None:
    panel, schema = _panel()
    bad = _evidence(channel="missing", kpi="other")
    reg = tmp_path / "ev.json"
    _write_registry(reg, bad)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        bayesian={
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
            "min_experiment_quality_tier": "medium",
        },
    )
    prep = prepare_bayesian_experiment_likelihood_terms(cfg, panel, schema)
    assert len(prep.used) == 0
    assert prep.rejected


def test_missing_se_rejected_unless_conservative(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(standard_error=None)
    reg = tmp_path / "ev.json"
    _write_registry(reg, ev)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        bayesian={
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
            "allow_conservative_missing_se": False,
        },
    )
    prep = prepare_bayesian_experiment_likelihood_terms(cfg, panel, schema)
    assert len(prep.used) == 0


def test_aggregate_only_blocked_when_disallowed(tmp_path: Path) -> None:
    rows = []
    for g in ("dma_0", "dma_1"):
        for w in range(15):
            rows.append({"g": g, "w": w, "y": 100.0, "m1": 5.0})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("g", "w", "y", ("m1",))
    ev = _evidence(
        geo_granularity=GeoGranularity.USER,
        geo_scope=["US"],
        metadata={"market": "US", "spend_multiplier": 0.9, "lift_scale": "log_mean_kpi_delta"},
    )
    reg = tmp_path / "ev.json"
    _write_registry(reg, ev)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        calibration={"model_geo_granularity": "dma"},
        bayesian={
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
            "allow_aggregate_only_evidence": False,
        },
    )
    prep = prepare_bayesian_experiment_likelihood_terms(cfg, panel, schema)
    assert any(r.get("reason") == "aggregate_only_not_allowed" for r in prep.rejected)


def test_level_log_scale_mismatch_fails() -> None:
    with pytest.raises(LiftScaleMismatchError, match="semi_log"):
        validate_experiment_lift_scale("mean_kpi_level_delta", ModelForm.SEMI_LOG)


def test_adjusted_se_higher_when_quality_lower() -> None:
    se_high, _ = compute_adjusted_standard_error(
        reported_se=0.1,
        evidence_weight=0.9,
        compatibility_status="compatible",
        allocation_required=False,
        allocation_quality="high",
        expiration_status="active",
        allow_missing_se=False,
    )
    se_low, _ = compute_adjusted_standard_error(
        reported_se=0.1,
        evidence_weight=0.2,
        compatibility_status="compatible",
        allocation_required=False,
        allocation_quality="high",
        expiration_status="active",
        allow_missing_se=False,
    )
    assert se_low > se_high


def test_valid_evidence_prepares_term(tmp_path: Path) -> None:
    rows = []
    for g in ("A", "B"):
        for w in range(25):
            rows.append({"g": g, "w": w, "y": 100.0 + w * 0.1, "m1": 5.0 + w * 0.01})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("g", "w", "y", ("m1",))
    ev = _evidence(geo_scope=["A", "B"], geo_granularity=GeoGranularity.GEO)
    reg = tmp_path / "ev.json"
    _write_registry(reg, ev)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        bayesian={
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
        },
    )
    prep = prepare_bayesian_experiment_likelihood_terms(cfg, panel, schema)
    assert len(prep.used) == 1
    term = prep.used[0]
    assert term.X_obs.shape == term.X_cf.shape
    assert term.adjusted_se > 0
    assert term.supports_subgeo_claims in {True, False}


def test_allocated_shock_blocked_when_disallowed(tmp_path: Path) -> None:
    rows = []
    for g in ("dma_0", "dma_1"):
        for w in range(15):
            rows.append({"g": g, "w": w, "y": 100.0, "m1": 5.0})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("g", "w", "y", ("m1",))
    ev = _evidence(geo_scope=["dma_0", "dma_1"], geo_granularity=GeoGranularity.REGION)
    reg = tmp_path / "ev.json"
    _write_registry(reg, ev)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        calibration={"model_geo_granularity": "national"},
        bayesian={
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
            "allow_allocated_shocks": False,
        },
    )
    prep = prepare_bayesian_experiment_likelihood_terms(cfg, panel, schema)
    assert any(r.get("reason") == "allocated_shock_not_allowed" for r in prep.rejected)


def test_prod_bayesian_decisioning_still_blocked() -> None:
    policy = RuntimePolicy(prod=True, allow_bayesian_decisioning=False)
    with pytest.raises(PolicyError, match="blocks Bayesian"):
        require_bayesian_block(Framework.BAYESIAN, policy)


@pytest.mark.pymc
def test_pymc_fit_emits_experiment_likelihood_report(tmp_path: Path) -> None:
    pytest.importorskip("pymc")
    panel, schema = _panel()
    ev = _evidence()
    reg = tmp_path / "ev.json"
    _write_registry(reg, ev)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "channel_columns": ["m1"],
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
        },
        bayesian={
            "backend": BayesianBackend.PYMC,
            "draws": 30,
            "tune": 30,
            "chains": 2,
            "use_experiment_likelihood": True,
            "experiment_registry_path": str(reg),
            "experiment_likelihood_weight": 1.0,
        },
    )
    from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer

    out = BayesianMMMTrainer(cfg, schema).fit(panel)
    rep = out.get("bayesian_experiment_likelihood_report") or {}
    assert rep.get("enabled") is True
    assert rep.get("research_only") is True
    assert rep.get("prod_decisioning_allowed") is False
    assert rep.get("n_evidence_units_used", 0) >= 1
    assert "governance_warnings" in rep
