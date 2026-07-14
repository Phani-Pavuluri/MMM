"""H7 — Ridge production diagnostic hardening tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.config.schema import Framework
from mmm.diagnostics.ridge_diagnostics import (
    FORBIDDEN_OUTPUT_FIELDS,
    compose_ridge_diagnostic_report,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.research.h6_synthetic.benchmark_harness import run_ridge_h6_benchmark
from mmm.research.h6_synthetic.production_shapes import (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_MEDIA_CORR,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)

ARCHIVE_PATH = Path(
    "docs/05_validation/archives/RIDGE_DIAGNOSTICS_H6_RETAIL_OMITTED_CONTROLS_20260601.json"
)


def _fit_and_report(world_id: str, *, vertical_id: str, media_correlated: bool = False):
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    trainer = RidgeBOMMMTrainer(config, schema)
    fit = trainer.fit(panel)
    report = compose_ridge_diagnostic_report(
        panel,
        schema,
        config,
        fit,
        trainer=trainer,
        vertical_id=vertical_id,
        media_correlated_controls=media_correlated,
        known_truth=spec.known_truth,
        world_metadata={"world_id": spec.world_id, "stress_variant": spec.stress_variant},
    )
    return spec, panel, fit, report


def test_retail_full_controls_no_missing_required_warning() -> None:
    _, _, _, report = _fit_and_report(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    cc = report["control_completeness"]
    assert not cc["omitted_control_risk"]
    assert cc["missing_required_controls"] == []
    assert "no_clean_media_attribution_claim" not in report["forbidden_claims"]


def test_omitted_controls_emit_forbidden_claims() -> None:
    _, _, _, report = _fit_and_report(WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail")
    cc = report["control_completeness"]
    assert cc["omitted_control_risk"]
    assert cc["missing_required_controls"]
    fc = report["forbidden_claims"]
    assert "no_clean_media_attribution_claim" in fc
    assert "no_budget_reallocation_claim_based_only_on_this_run" in fc
    assert report["production_flags"]["approved_for_prod"] is False


def test_media_correlated_controls_confounding_risk() -> None:
    _, _, _, report = _fit_and_report(
        WORLD_H6_PILOT_RETAIL_MEDIA_CORR,
        vertical_id="retail",
        media_correlated=True,
    )
    assert report["control_completeness"]["media_correlated_controls"] is True
    assert report["diagnostic_severity"] in ("low", "medium", "high")


def test_sparse_channel_forbidden_claim_on_pilot() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    # Force extreme sparsity on radio for diagnostic probe
    panel = panel.copy()
    panel["radio"] = 0.0
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(
        panel, schema, config, None, vertical_id="retail"
    )
    sparse = report["sparse_channels"]
    assert sparse["silent_drop_occurred"] is False
    if "radio" in (sparse.get("sparse_channel_extreme") or []):
        assert any("no_separate_channel_effect_claim_for_radio" in c for c in report["forbidden_claims"])


def test_high_collinearity_weak_id_warning() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    # Inflate correlation between display and ctv
    panel = panel.copy()
    panel["ctv"] = panel["display"] * 0.98 + 0.01
    report = compose_ridge_diagnostic_report(panel, schema, config, None, vertical_id="retail")
    col = report["collinearity"]
    if col["max_abs_correlation"] >= 0.85:
        assert col["weak_identification_risk"]
        assert any("weak_identification" in w for w in col["warnings"])


def test_transform_metadata_warning_when_no_fit() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(panel, schema, config, fit_result=None, vertical_id="retail")
    td = report["transform_diagnostics"]
    assert not td["metadata_complete"]
    assert any("missing_best_params" in w for w in td["warnings"])


def test_no_optimizer_decision_surface_recommendation_fields() -> None:
    _, _, _, report = _fit_and_report(WORLD_H6_PILOT_RETAIL_FULL, vertical_id="retail")
    for field in FORBIDDEN_OUTPUT_FIELDS:
        assert field not in report or report.get(field) is None
    assert report["production_flags"]["optimizer_enabled"] is False
    assert report["production_flags"]["recommendations_enabled"] is False
    assert report["lift_simulation_stability"]["lift_simulation_run"] is False


def test_compose_requires_ridge_framework() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_FULL)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    config.framework = Framework.BAYESIAN
    schema = h6_panel_schema(spec)
    with pytest.raises(ValueError, match="ridge_bo"):
        compose_ridge_diagnostic_report(panel, schema, config, None)


def test_write_representative_archive() -> None:
    spec, panel, fit, report = _fit_and_report(
        WORLD_H6_PILOT_RETAIL_OMITTED, vertical_id="retail"
    )
    artifact = {
        "artifact_kind": "RIDGE_PRODUCTION_DIAGNOSTICS",
        "milestone": "H7",
        "world_metadata": {
            "world_id": spec.world_id,
            "vertical": spec.vertical_id,
            "control_variant": spec.stress_variant,
            "scale": spec.scale,
        },
        "ridge_diagnostics": report,
        "ridge_benchmark_excerpt": run_ridge_h6_benchmark(spec, panel),
        "production_boundary": {
            "ridge_remains_production_baseline": True,
            "bayes_h5_research_only": True,
            "diagnostics_are_not_hard_gates": True,
        },
    }
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_PATH.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    assert ARCHIVE_PATH.is_file()
    assert report["forbidden_claims"]
