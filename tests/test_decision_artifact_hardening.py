"""Fail-closed train↔decide artifact contracts (fingerprint, ridge summary, replay policy, prod transforms)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.decision.service import simulate_decision
from mmm.governance.decision_fingerprint import (
    compare_training_and_decision_fingerprints,
    require_decision_fingerprint_match,
)
from mmm.governance.decision_ridge_summary import validate_ridge_fit_summary_for_prod_decide
from mmm.governance.policy import PolicyError
from mmm.governance.replay_refit_prod_policy import validate_prod_replay_refit_mode
from tests.prod_extension_fixtures import (
    enrich_prod_ridge_decide_extension,
    merge_prod_extension,
    prod_replay_evidence_block,
)


def _panel_csv(tmp_path: Path) -> Path:
    p = tmp_path / "panel.csv"
    p.write_text(
        "geo,week,c1,revenue\nG1,1,10,100\nG1,2,11,101\nG2,1,9,99\nG2,2,10,100\n",
        encoding="utf-8",
    )
    return p


def _schema() -> PanelSchema:
    return PanelSchema("geo", "week", "revenue", ("c1",))


def _prod_cfg(tmp_path: Path, panel_csv: Path) -> MMMConfig:
    return MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={
            "path": str(panel_csv),
            "geo_column": "geo",
            "week_column": "week",
            "target_column": "revenue",
            "channel_columns": ["c1"],
            "data_version_id": "test-v1",
        },
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions={"optimization_gates": {"enabled": True}},
        budget={"total_budget": 100.0},
    )


def _ridge_summary(coef: list[float] | None = None) -> dict:
    return {
        "coef": coef if coef is not None else [0.1],
        "intercept": [4.5],
        "model_form": "semi_log",
        "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
        "transform_policy": {
            "policy_version": "mmm_transform_policy_v1",
            "framework": "ridge_bo",
            "model_form": "semi_log",
            "adstock": "geometric",
            "saturation": "hill",
            "mode_family": "ridge_bo_joint_hyperparams",
        },
    }


def _extension_report(tmp_path: Path, cfg: MMMConfig, panel: pd.DataFrame, schema: PanelSchema) -> dict:
    from mmm.contracts.seed_resolution import resolve_seed_contract

    fp = fingerprint_panel(panel, schema, config=cfg, seed_resolution=resolve_seed_contract(cfg))
    base = merge_prod_extension(
        {
            "ridge_fit_summary": _ridge_summary(),
            "transform_policy": _ridge_summary()["transform_policy"],
            "data_fingerprint": fp,
            "governance": {"approved_for_optimization": True},
            "response_diagnostics": {"safe_for_optimization": True},
            "identifiability": {"identifiability_score": 0.4},
            "panel_qa": {"max_severity": "info"},
            "model_release": {"state": "planning_allowed"},
        }
    )
    base.update(prod_replay_evidence_block())
    return base


def test_fingerprint_match_passes(tmp_path: Path) -> None:
    from mmm.data.loader import DatasetBuilder
    from mmm.data.panel_order import sort_panel_for_modeling
    from mmm.data.schema import validate_panel

    panel_csv = _panel_csv(tmp_path)
    cfg = _prod_cfg(tmp_path, panel_csv)
    builder = DatasetBuilder(cfg.data)
    schema = builder.schema()
    panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    er = enrich_prod_ridge_decide_extension(
        merge_prod_extension({"ridge_fit_summary": _ridge_summary()}),
        cfg=cfg,
        panel=panel,
        schema=schema,
    )
    builder2 = DatasetBuilder(cfg.data)
    panel2 = sort_panel_for_modeling(validate_panel(builder2.build(), schema), schema)
    from mmm.contracts.seed_resolution import resolve_seed_contract

    out = require_decision_fingerprint_match(
        cfg,
        er,
        panel=panel2,
        schema=schema,
        seed_resolution=resolve_seed_contract(cfg),
    )
    assert out.get("matched") is True


def test_fingerprint_mismatch_fails_prod(tmp_path: Path) -> None:
    panel_csv = _panel_csv(tmp_path)
    cfg = _prod_cfg(tmp_path, panel_csv)
    panel = pd.read_csv(panel_csv)
    schema = _schema()
    er = _extension_report(tmp_path, cfg, panel, schema)
    er["data_fingerprint"] = dict(er["data_fingerprint"])
    er["data_fingerprint"]["sha256_combined"] = "0" * 64
    with pytest.raises(PolicyError, match="fingerprint mismatch"):
        require_decision_fingerprint_match(cfg, er, panel=panel, schema=schema)


def test_fingerprint_override_requires_waiver_path() -> None:
    with pytest.raises(ValueError, match="waiver_path"):
        MMMConfig(
            governance={"allow_decision_fingerprint_mismatch": True},
        )


def test_fingerprint_override_loud(tmp_path: Path) -> None:
    panel_csv = _panel_csv(tmp_path)
    waiver = tmp_path / "fp_waiver.json"
    waiver.write_text(
        json.dumps(
            {
                "waiver_id": "fp-1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "reason": "controlled rollback test",
            }
        ),
        encoding="utf-8",
    )
    cfg = _prod_cfg(tmp_path, panel_csv)
    cfg = cfg.model_copy(
        update={
            "governance": cfg.governance.model_copy(
                update={
                    "allow_decision_fingerprint_mismatch": True,
                    "decision_fingerprint_mismatch_waiver_path": str(waiver),
                }
            )
        }
    )
    panel = pd.read_csv(panel_csv)
    schema = _schema()
    er = _extension_report(tmp_path, cfg, panel, schema)
    er["data_fingerprint"]["sha256_combined"] = "f" * 64
    out = require_decision_fingerprint_match(cfg, er, panel=panel, schema=schema)
    assert out.get("severe_warning")
    assert "OVERRIDE" in out["severe_warning"]


def test_legacy_fingerprint_comparison_warns() -> None:
    train = {"sha256_panel_keycols_sorted_csv": "a" * 64}
    decide = {"sha256_panel_keycols_sorted_csv": "a" * 64}
    out = compare_training_and_decision_fingerprints(train, decide)
    assert out["matched"] is True
    assert out["warnings"]


def test_ridge_summary_missing_best_params_fails() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "data_version_id": "x"},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions={"optimization_gates": {"enabled": True}},
    )
    er = {
        "ridge_fit_summary": {"coef": [0.1], "intercept": [0.0], "model_form": "semi_log"},
        "transform_policy": _ridge_summary()["transform_policy"],
        "data_fingerprint": {"sha256_combined": "a" * 64},
    }
    with pytest.raises(PolicyError, match="best_params"):
        validate_ridge_fit_summary_for_prod_decide(cfg, er)


def test_prod_weibull_config_fails_parse() -> None:
    with pytest.raises((PolicyError, ValueError), match="geometric|Unsupported"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "data_version_id": "x"},
            cv={"mode": "rolling"},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
            transforms={"adstock": "weibull"},
            extensions={"optimization_gates": {"enabled": True}},
        )


def test_prod_full_panel_replay_fails_without_waiver(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="full_panel_replay_refit_prod_waiver"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "data_version_id": "x"},
            cv={"mode": "rolling"},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
            calibration={"use_replay_calibration": True, "replay_refit_mode": "full_panel_refit"},
            extensions={"optimization_gates": {"enabled": True}},
        )


def test_research_full_panel_replay_still_parses() -> None:
    cfg = MMMConfig(
        data={"channel_columns": ["c1"]},
        calibration={"use_replay_calibration": True, "replay_refit_mode": "full_panel_refit"},
    )
    validate_prod_replay_refit_mode(cfg)


def test_prod_replay_waiver_parses(tmp_path: Path) -> None:
    w = tmp_path / "replay_waiver.json"
    w.write_text(
        json.dumps(
            {
                "waiver_id": "replay-1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "reason": "legacy replay migration window",
            }
        ),
        encoding="utf-8",
    )
    MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "data_version_id": "x"},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        calibration={
            "use_replay_calibration": True,
            "replay_refit_mode": "full_panel_refit",
            "full_panel_replay_refit_prod_waiver_path": str(w),
        },
        extensions={"optimization_gates": {"enabled": True}},
    )


def test_simulate_decision_aborts_on_fingerprint_mismatch(tmp_path: Path) -> None:
    panel_csv = _panel_csv(tmp_path)
    cfg = _prod_cfg(tmp_path, panel_csv)
    panel = pd.read_csv(panel_csv)
    schema = _schema()
    er = _extension_report(tmp_path, cfg, panel, schema)
    er["data_fingerprint"]["sha256_combined"] = "deadbeef" * 8
    with pytest.raises(PolicyError, match="fingerprint mismatch"):
        simulate_decision(
            cfg=cfg,
            scenario={"candidate_spend": {"c1": 12.0}},
            extension_report=er,
            out=tmp_path / "out.json",
        )


def test_transform_policy_mismatch_fails() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        transforms={"adstock": "geometric", "saturation": "hill"},
        data={"channel_columns": ["c1"], "data_version_id": "x"},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions={"optimization_gates": {"enabled": True}},
    )
    tp = dict(_ridge_summary()["transform_policy"])
    tp["adstock"] = "weibull"
    er = {
        "ridge_fit_summary": _ridge_summary(),
        "transform_policy": tp,
        "data_fingerprint": {"sha256_combined": "a" * 64},
    }
    with pytest.raises(PolicyError, match="transform_policy.adstock"):
        validate_ridge_fit_summary_for_prod_decide(cfg, er)
