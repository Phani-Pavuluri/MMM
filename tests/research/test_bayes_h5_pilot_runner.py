"""Bayes-H5 sandbox pilot artifact (research only)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_pilot_runner import (
    DEFAULT_ARTIFACT_PATH,
    EXTENDED_PILOT_ID,
    build_h5_pilot_summary,
    build_h5_repeated_pilot_summary,
    validate_h5_extended_repeated_pilot_schema,
    validate_h5_pilot_schema,
    validate_h5_repeated_pilot_schema,
)
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import H5_WORLD_IDS, h5_world_production_flags
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED


def _mock_world_row(world_id: str, *, mismatch: bool) -> dict[str, Any]:
    mode = "intentional_mismatch" if mismatch else "aligned"
    warnings = [f"h5:transform_mismatch:{world_id}"] if mismatch else []
    return {
        "world_id": world_id,
        "transform_mismatch_mode": mode,
        "h5_diagnostic_warnings": warnings,
        "beta_gc_mae": 0.12 if not mismatch else 0.55,
        "policy_outcomes": {"outcome": "warn" if mismatch else "pass", "hard_gate": False},
        **h5_world_production_flags(),
        "decision_surface": None,
        "optimizer_ready_curves": None,
        "budget_recommendation": None,
        "recommendation": None,
    }


def test_pilot_summary_schema_valid() -> None:
    rows = [
        _mock_world_row("WORLD-BAYES-H5-ADSTOCK-ALIGNED", mismatch=False),
        _mock_world_row("WORLD-BAYES-H5-ADSTOCK-MISMATCH", mismatch=True),
    ]
    summary = build_h5_pilot_summary(rows, fast_mcmc=True, sampler_metadata={"draws": 50})
    validate_h5_pilot_schema(summary)
    assert summary["model_spec_version"] == H5_MODEL_SPEC_VERSION
    assert summary["hard_gate"] is False
    assert "WORLD-BAYES-H5-ADSTOCK-MISMATCH" in summary["mismatch_worlds"]
    assert "WORLD-BAYES-H5-ADSTOCK-ALIGNED" in summary["aligned_worlds"]
    assert any("transform_mismatch" in w for w in summary["diagnostic_warnings"])


@patch("mmm.research.bayes_h3_sandbox.h5_pilot_runner.run_h5_recovery_world")
def test_run_h5_pilot_writes_artifact(mock_run, tmp_path) -> None:
    def _fake(wid: str, **kwargs: object) -> dict[str, object]:
        mismatch = "MISMATCH" in wid
        return {
            "model_spec_version": H5_MODEL_SPEC_VERSION,
            "h4_recovery": {
                "beta_gc_mae": 0.2,
                "h5_diagnostic_warnings": [f"h5:transform_mismatch:{wid}"] if mismatch else [],
            },
            "h5_transform_diagnostics": {"transform_mismatch_detected": mismatch},
            "decision_surface": None,
            "budget_recommendation": None,
            "optimizer_ready_curves": None,
        }

    mock_run.side_effect = _fake
    from mmm.research.bayes_h3_sandbox.h5_pilot_runner import run_h5_pilot

    out_path = tmp_path / "pilot.json"
    summary = run_h5_pilot(fast_mcmc=True, world_ids=H5_WORLD_IDS[:2], artifact_path=out_path)
    validate_h5_pilot_schema(summary)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["approved_for_prod"] is False
    assert len(loaded["per_world_metrics"]) == 2


def test_default_artifact_path_constant() -> None:
    assert DEFAULT_ARTIFACT_PATH.name == "BAYES_H5_SANDBOX_PILOT_20260601.json"


def test_repeated_pilot_schema_valid() -> None:
    per_run = [
        {
            "world_id": "WORLD-BAYES-H5-ADSTOCK-ALIGNED",
            "nuts_seed": 4400,
            "beta_gc_mae": 0.26,
            "h5_diagnostic_warnings": [],
            **h5_world_production_flags(),
            "decision_surface": None,
            "optimizer_ready_curves": None,
            "budget_recommendation": None,
            "recommendation": None,
        },
        {
            "world_id": "WORLD-BAYES-H5-ADSTOCK-MISMATCH",
            "nuts_seed": 4400,
            "beta_gc_mae": 0.28,
            "h5_diagnostic_warnings": ["h5:transform_mismatch:WORLD-BAYES-H5-ADSTOCK-MISMATCH: x"],
            **h5_world_production_flags(),
            "decision_surface": None,
            "optimizer_ready_curves": None,
            "budget_recommendation": None,
            "recommendation": None,
        },
    ]
    summary = build_h5_repeated_pilot_summary(
        per_run,
        seeds=(4400,),
        h4c_baselines={"WORLD-BAYES-H4C-ADSTOCKED-MEDIA": 0.279},
    )
    validate_h5_repeated_pilot_schema(summary)
    assert summary["hard_gate"] is False
    assert summary["aggregate_by_world"]["WORLD-BAYES-H5-ADSTOCK-ALIGNED"]["beta_gc_mae"]["mean"] == 0.26


@patch("mmm.research.bayes_h3_sandbox.h5_pilot_runner.run_h5_recovery_world")
def test_run_h5_repeated_pilot(mock_run, tmp_path) -> None:
    def _fake(wid: str, **kwargs: object) -> dict[str, object]:
        seed = int(kwargs.get("nuts_seed", 4400))
        return {
            "h4_recovery": {
                "beta_gc_mae": 0.2 + seed * 1e-6,
                "h5_diagnostic_warnings": [],
            },
            "h5_transform_diagnostics": {},
        }

    mock_run.side_effect = _fake
    from mmm.research.bayes_h3_sandbox.h5_pilot_runner import run_h5_repeated_pilot

    out = tmp_path / "repeated.json"
    summary = run_h5_repeated_pilot(
        seeds=(4400, 4401),
        world_ids=H5_WORLD_IDS[:1],
        fast_mcmc=True,
        artifact_path=out,
    )
    validate_h5_repeated_pilot_schema(summary)
    assert summary["aggregate_by_world"]["WORLD-BAYES-H5-ADSTOCK-ALIGNED"]["n_runs"] == 2


def test_extended_repeated_pilot_schema_valid() -> None:
    h5b_agg = {
        "WORLD-BAYES-H5-ADSTOCK-ALIGNED": {
            "beta_gc_mae": {"mean": 0.264},
            "transform_mismatch_warning_rate": 0.0,
            "unexpected_mismatch_warning_rate": 0.0,
        },
        "WORLD-BAYES-H5-ADSTOCK-MISMATCH": {
            "beta_gc_mae": {"mean": 0.278},
            "transform_mismatch_warning_rate": 1.0,
            "unexpected_mismatch_warning_rate": 0.0,
        },
    }
    per_run = [
        {
            "world_id": "WORLD-BAYES-H5-ADSTOCK-ALIGNED",
            "nuts_seed": 4400,
            "beta_gc_mae": 0.265,
            "h5_diagnostic_warnings": [],
            "transform_mismatch_warning_rate": 0.0,
            **h5_world_production_flags(),
            "decision_surface": None,
            "optimizer_ready_curves": None,
            "budget_recommendation": None,
            "recommendation": None,
        },
        {
            "world_id": "WORLD-BAYES-H5-ADSTOCK-MISMATCH",
            "nuts_seed": 4400,
            "beta_gc_mae": 0.28,
            "h5_diagnostic_warnings": ["h5:transform_mismatch:WORLD-BAYES-H5-ADSTOCK-MISMATCH: x"],
            **h5_world_production_flags(),
            "decision_surface": None,
            "optimizer_ready_curves": None,
            "budget_recommendation": None,
            "recommendation": None,
        },
    ]
    summary = build_h5_repeated_pilot_summary(
        per_run,
        seeds=(4400,),
        extended_mcmc=True,
        sampler_metadata={**SAMPLER_EXTENDED, "extended_mcmc_profile": True},
        h5b_data={"aggregate_by_world": h5b_agg},
    )
    validate_h5_extended_repeated_pilot_schema(summary)
    assert summary["pilot_id"] == EXTENDED_PILOT_ID
    assert summary["hard_gate"] is False
    assert "comparison_to_h5b_fast_pilot" in summary
    assert summary["comparison_to_h5b_fast_pilot"]["WORLD-BAYES-H5-ADSTOCK-ALIGNED"]["material_change_vs_h5b"] is False
    agg = summary["aggregate_by_world"]["WORLD-BAYES-H5-ADSTOCK-MISMATCH"]
    assert agg["transform_mismatch_warning_rate"] == 1.0


@pytest.mark.slow
@pytest.mark.pymc
def test_repeated_pilot_smoke_two_worlds_two_seeds() -> None:
    from mmm.research.bayes_h3_sandbox.h5_pilot_runner import run_h5_repeated_pilot

    summary = run_h5_repeated_pilot(
        seeds=(4400, 4401),
        world_ids=(
            "WORLD-BAYES-H5-ADSTOCK-ALIGNED",
            "WORLD-BAYES-H5-ADSTOCK-MISMATCH",
        ),
        fast_mcmc=True,
    )
    validate_h5_repeated_pilot_schema(summary)
    assert summary["aggregate_by_world"]["WORLD-BAYES-H5-ADSTOCK-MISMATCH"]["transform_mismatch_warning_rate"] == 1.0


@pytest.mark.slow
@pytest.mark.pymc
def test_extended_repeated_pilot_smoke_two_worlds_two_seeds() -> None:
    from mmm.research.bayes_h3_sandbox.h5_pilot_runner import run_h5_repeated_pilot

    summary = run_h5_repeated_pilot(
        seeds=(4400, 4401),
        world_ids=(
            "WORLD-BAYES-H5-SATURATION-ALIGNED",
            "WORLD-BAYES-H5-SATURATION-MISMATCH",
        ),
        extended_mcmc=True,
    )
    validate_h5_extended_repeated_pilot_schema(summary)
    assert summary["sampler_settings"]["draws"] == 600
    assert summary["h5c_conclusions"]["unexpected_mismatch_clean"] is True
