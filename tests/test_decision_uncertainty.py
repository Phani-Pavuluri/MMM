"""Ridge uncertainty disclosure — no fabricated CIs."""

from __future__ import annotations

from mmm.artifacts.decision_bundle import build_decision_bundle
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.governance.decision_uncertainty import RIDGE_DISCLOSURE, build_decision_uncertainty


def test_ridge_default_uncertainty_unavailable() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["c1"]})
    du = build_decision_uncertainty(cfg)
    assert du["uncertainty_available"] is False
    assert du["confidence_supported"] is False
    assert RIDGE_DISCLOSURE in du["disclosure_text"]


def test_decision_bundle_includes_uncertainty() -> None:
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    schema = PanelSchema("g", "w", "y", ("c1",))
    fp = {"sha256_panel_keycols_sorted_csv": "a" * 64, "sha256_schema_json": "b" * 64, "n_rows": 1}
    b = build_decision_bundle(
        config=cfg,
        schema=schema,
        simulation_contract={"source": "t"},
        data_fingerprint=fp,
    )
    assert b["decision_uncertainty"]["uncertainty_available"] is False


def test_prod_ridge_disclosure_flags() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        framework=Framework.RIDGE_BO,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "data_version_id": "v1"},
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
        extensions={"optimization_gates": {"enabled": True}},
    )
    du = build_decision_uncertainty(cfg)
    assert du["ridge_production_forbids_precise_monetary_ci"] is True
