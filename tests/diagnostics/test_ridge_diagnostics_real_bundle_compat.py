"""H11 — Ridge diagnostics real/realistic bundle compatibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.config.schema import Framework
from mmm.diagnostics.ridge_diagnostics import (
    FORBIDDEN_OUTPUT_FIELDS,
    build_control_completeness_diagnostics,
    build_ridge_transform_diagnostics,
    compose_ridge_diagnostic_report,
)
from mmm.diagnostics.ridge_real_bundle_hardening import (
    BENCHMARK_BUNDLE_SPEC,
    BUNDLE_BENCHMARK_GEO_PANEL_V1,
    run_real_bundle_ridge_diagnostics,
    validate_h11_artifact_completeness,
)
from mmm.data.schema import PanelSchema
from mmm.research.h6_synthetic.production_shapes import (
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)

ARCHIVE_JSON = Path(
    "docs/05_validation/archives/"
    "H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_20260601.json"
)
def test_missing_transform_metadata_warns_not_crash() -> None:
    schema = PanelSchema(
        geo_column="g",
        week_column="w",
        target_column="y",
        channel_columns=("tv",),
        control_columns=(),
    )
    from mmm.config.schema import MMMConfig, ModelForm, PoolingMode, RunEnvironment

    config = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        run_environment=RunEnvironment.RESEARCH,
        data={
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["tv"],
        },
        transforms={"adstock": "geometric", "saturation": "hill"},
    )
    diag = build_ridge_transform_diagnostics(config, schema, fit_result={})
    assert diag["metadata_complete"] is False
    assert "ridge_transform:missing_best_params_metadata" in diag["warnings"]


def test_unknown_vertical_handled_explicitly() -> None:
    schema = PanelSchema(
        geo_column="geo_id",
        week_column="week_start_date",
        target_column="revenue",
        channel_columns=("search", "social", "tv"),
        control_columns=(),
    )
    cc = build_control_completeness_diagnostics(schema, vertical_id="unknown_vertical_xyz")
    assert cc["vertical_profile_known"] is False
    assert cc["vertical_id"] == "unknown_vertical_xyz"
    assert any("unknown_vertical" in w for w in cc["warnings"])


def test_missing_control_list_recorded_for_retail_on_no_controls_panel() -> None:
    if not Path("examples/benchmark_geo_panel_v1.csv").is_file():
        pytest.skip("benchmark panel missing")
    result = run_real_bundle_ridge_diagnostics(
        {**BENCHMARK_BUNDLE_SPEC, "n_trials": 2},
    )
    cc = result["report"]["control_completeness"]
    assert cc["missing_required_controls"]
    assert cc["omitted_control_risk"] is True


def test_calibration_context_absent_is_explicit() -> None:
    if not Path("examples/benchmark_geo_panel_v1.csv").is_file():
        pytest.skip("benchmark panel missing")
    result = run_real_bundle_ridge_diagnostics({**BENCHMARK_BUNDLE_SPEC, "n_trials": 2})
    lineage = result["report"]["evidence_attachment_lineage"]
    assert lineage["attempted"] is False
    assert lineage["calibration_evidence_context_present"] is False
    assert lineage["mip_c1_attachment_wired"] is False
    assert "CalibrationSignal" in lineage["note"]


def test_redacted_archive_validates_completeness() -> None:
    if not ARCHIVE_JSON.is_file():
        pytest.skip("H11 archive not materialized")
    payload = json.loads(ARCHIVE_JSON.read_text(encoding="utf-8"))
    report = payload["ridge_production_diagnostics_report"]
    completeness = validate_h11_artifact_completeness(report)
    assert completeness["all_passed"]
    assert payload["bundle_id"] == BUNDLE_BENCHMARK_GEO_PANEL_V1
    coef = report["coefficient_stability"]["media_coef_by_channel"]
    assert all(v == "[redacted]" for v in coef.values())


def test_no_optimizer_decision_surface_recommendation_fields_on_bundle() -> None:
    if not Path("examples/benchmark_geo_panel_v1.csv").is_file():
        pytest.skip("benchmark panel missing")
    result = run_real_bundle_ridge_diagnostics({**BENCHMARK_BUNDLE_SPEC, "n_trials": 2})
    report = result["report"]
    flags = report["production_flags"]
    assert flags["optimizer_enabled"] is False
    assert flags["decision_surface_enabled"] is False
    assert flags["recommendations_enabled"] is False
    for key in FORBIDDEN_OUTPUT_FIELDS:
        assert key not in report


def test_h11_archive_markdown_exists() -> None:
    md = Path(
        "docs/05_validation/archives/"
        "H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_MMM_BENCHMARK_GEO_PANEL_V1_SUMMARY_20260601.md"
    )
    if not md.is_file():
        pytest.skip("H11 summary archive missing")
    text = md.read_text(encoding="utf-8")
    assert "Calibration evidence" in text
    assert "not attached" in text.lower() or "MIP-C1" in text


def test_compose_evidence_lineage_on_h6_stub() -> None:
    spec = get_h6_world("WORLD-H6-PILOT-RETAIL-FULL-CONTROLS")
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    report = compose_ridge_diagnostic_report(panel, schema, config, fit_result={})
    assert "evidence_attachment_lineage" in report
    assert report["evidence_attachment_lineage"]["calibration_evidence_context_present"] is False
