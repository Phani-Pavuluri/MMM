"""Ridge uncertainty research path — coverage diagnostics, production blocked."""

from __future__ import annotations

import pytest

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.governance.decision_uncertainty import build_decision_uncertainty
from mmm.governance.ridge_uncertainty_research import (
    PRODUCTION_INTERVALS_ALLOWED,
    build_ridge_uncertainty_research_report,
)
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


@pytest.mark.slow
@pytest.mark.optuna
def test_research_report_blocks_production_intervals() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=50, channels=("ch0",), betas=(0.5,), decay=0.3, noise=0.05)
    df, schema = generate_geo_panel(spec, seed=1)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=15, horizon_weeks=4),
        ridge_bo={"n_trials": 3},
        extensions={"ridge_uncertainty_research": {"enabled": True, "bootstrap_rounds": 6}},
    )
    resolve_seed_contract(cfg)
    rep = build_ridge_uncertainty_research_report(df, schema, cfg, seed=2)
    assert rep["production_intervals_allowed"] is False
    assert rep["uncertainty_available_for_decisioning"] is False
    assert PRODUCTION_INTERVALS_ALLOWED is False
    prod = build_decision_uncertainty(cfg)
    assert prod["uncertainty_available"] is False
