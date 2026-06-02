"""Bayes-H5d research TrustReport diagnostic mapping (not production TrustReport)."""

from __future__ import annotations

import json

import pytest

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    WARNING_TAXONOMY,
    H5TrustDiagnosticMappingError,
    build_trust_diagnostic_mapping,
    build_world_trust_diagnostic_payload,
    classify_convergence_status,
    derive_real_panel_transform_warning_codes,
    derive_warning_codes,
    evidence_promotion_allowed,
    resolve_world_role,
    validate_trust_diagnostic_mapping_artifact,
    write_trust_diagnostic_mapping_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    H5_WORLD_IDS,
    WORLD_BAYES_H5_ADSTOCK_ALIGNED,
    WORLD_BAYES_H5_ADSTOCK_MISMATCH,
    WORLD_BAYES_H5_CORRELATED_CHANNELS,
    WORLD_BAYES_H5_SATURATION_ALIGNED,
    WORLD_BAYES_H5_SATURATION_MISMATCH,
    WORLD_BAYES_H5_SPARSE_RECOVERY,
    WORLD_BAYES_H5_WEAK_SIGNAL,
)


def _agg(
    *,
    tm_rate: float = 0.0,
    unexp: float = 0.0,
    col: float = 0.0,
    weak: float = 0.0,
    beta_mean: float = 0.2,
) -> dict:
    return {
        "n_runs": 3,
        "beta_gc_mae": {"mean": beta_mean, "std": 0.001},
        "mu_c_mae": {"mean": 0.2},
        "transform_mismatch_warning_rate": tm_rate,
        "unexpected_mismatch_warning_rate": unexp,
        "collinearity_warning_rate": col,
        "weak_identification_warning_rate": weak,
    }


def _minimal_pilot() -> dict:
    return {
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "aggregate_by_world": {wid: _agg() for wid in H5_WORLD_IDS},
        "h4c_baseline_comparison": {},
        "per_run": [{"world_id": wid, "role": "recovery_candidate"} for wid in H5_WORLD_IDS],
    }


def test_mapping_artifact_from_h5c_file_valid() -> None:
    mapping = write_trust_diagnostic_mapping_artifact()
    validate_trust_diagnostic_mapping_artifact(mapping)
    assert mapping["hard_gate"] is False
    assert mapping["approved_for_prod"] is False
    assert "comparison_to_h5b" not in json.dumps(mapping)
    assert len(mapping["per_world_diagnostic_payloads"]) == len(H5_WORLD_IDS)


def test_production_flags_false_on_all_payloads() -> None:
    mapping = build_trust_diagnostic_mapping(
        _minimal_pilot(),
        source_artifact="test.json",
    )
    for payload in mapping["per_world_diagnostic_payloads"]:
        pf = payload["production_flags"]
        assert pf["hard_gate"] is False
        assert pf["approved_for_prod"] is False
        assert pf["prod_decisioning_allowed"] is False
        assert pf["production_promotion"] is False


def test_no_optimizer_decision_surface_recommendation_fields() -> None:
    mapping = build_trust_diagnostic_mapping(_minimal_pilot(), source_artifact="t.json")
    blob = json.dumps(mapping)
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert forbidden not in blob or '"decision_surface": null' not in blob
    validate_trust_diagnostic_mapping_artifact(mapping)


def test_mismatch_worlds_map_transform_mismatch_warnings() -> None:
    pilot = _minimal_pilot()
    pilot["aggregate_by_world"][WORLD_BAYES_H5_ADSTOCK_MISMATCH] = _agg(tm_rate=1.0, beta_mean=0.28)
    pilot["aggregate_by_world"][WORLD_BAYES_H5_SATURATION_MISMATCH] = _agg(tm_rate=1.0, beta_mean=0.10)
    mapping = build_trust_diagnostic_mapping(pilot, source_artifact="t.json")
    by_id = {p["world_id"]: p for p in mapping["per_world_diagnostic_payloads"]}
    assert "h5:transform_mismatch:adstock" in by_id[WORLD_BAYES_H5_ADSTOCK_MISMATCH]["warning_codes"]
    assert "h5:transform_mismatch:saturation" in by_id[WORLD_BAYES_H5_SATURATION_MISMATCH]["warning_codes"]
    assert by_id[WORLD_BAYES_H5_ADSTOCK_MISMATCH]["transform_alignment_status"] == "intentional_mismatch"


def test_weak_id_worlds_map_weak_id_warnings() -> None:
    pilot = _minimal_pilot()
    pilot["aggregate_by_world"][WORLD_BAYES_H5_CORRELATED_CHANNELS] = _agg(col=1.0)
    pilot["aggregate_by_world"][WORLD_BAYES_H5_WEAK_SIGNAL] = _agg(weak=1.0)
    mapping = build_trust_diagnostic_mapping(pilot, source_artifact="t.json")
    by_id = {p["world_id"]: p for p in mapping["per_world_diagnostic_payloads"]}
    assert "h5:weak_identification:collinearity" in by_id[WORLD_BAYES_H5_CORRELATED_CHANNELS]["warning_codes"]
    assert "h5:weak_identification:weak_signal_generative" in by_id[WORLD_BAYES_H5_WEAK_SIGNAL]["warning_codes"]
    assert "h5:transform_mismatch:adstock" not in by_id[WORLD_BAYES_H5_CORRELATED_CHANNELS]["warning_codes"]


def test_sparse_maps_report_only() -> None:
    pilot = _minimal_pilot()
    mapping = build_trust_diagnostic_mapping(pilot, source_artifact="t.json")
    by_id = {p["world_id"]: p for p in mapping["per_world_diagnostic_payloads"]}
    sparse = by_id[WORLD_BAYES_H5_SPARSE_RECOVERY]
    assert sparse["sparse_recovery_status"] == "report_only"
    assert "h5:sparse_recovery:report_only" in sparse["warning_codes"]
    assert "h5:recovery_candidate:stable_research_only" not in sparse["warning_codes"]


def test_aligned_recovery_maps_stable_research_only_not_prod_pass() -> None:
    pilot = _minimal_pilot()
    pilot["aggregate_by_world"][WORLD_BAYES_H5_SATURATION_ALIGNED] = _agg(beta_mean=0.11)
    pilot["h4c_baseline_comparison"] = {
        WORLD_BAYES_H5_SATURATION_ALIGNED: {"improved_vs_h4c": True},
    }
    mapping = build_trust_diagnostic_mapping(pilot, source_artifact="t.json")
    aligned = next(
        p for p in mapping["per_world_diagnostic_payloads"] if p["world_id"] == WORLD_BAYES_H5_SATURATION_ALIGNED
    )
    assert "h5:recovery_candidate:stable_research_only" in aligned["warning_codes"]
    assert "h5:production:block" in aligned["warning_codes"]
    assert aligned["production_flags"]["approved_for_prod"] is False
    assert "production pass" not in aligned["recommended_interpretation"].lower()


def test_warning_taxonomy_deterministic() -> None:
    agg = _agg(tm_rate=1.0)
    codes_a = derive_warning_codes(WORLD_BAYES_H5_ADSTOCK_MISMATCH, aggregate=agg, role="transform_mismatch")
    codes_b = derive_warning_codes(WORLD_BAYES_H5_ADSTOCK_MISMATCH, aggregate=agg, role="transform_mismatch")
    assert codes_a == codes_b
    assert all(c in WARNING_TAXONOMY for c in codes_a)


def test_unknown_world_role_fails_closed() -> None:
    with pytest.raises(H5TrustDiagnosticMappingError, match="unknown H5 world"):
        resolve_world_role("WORLD-BAYES-H5-UNKNOWN")


def test_unknown_role_string_fails_closed() -> None:
    with pytest.raises(H5TrustDiagnosticMappingError, match="unknown world role"):
        resolve_world_role(WORLD_BAYES_H5_ADSTOCK_ALIGNED, pilot_row={"role": "mystery_role"})


def test_build_world_payload_aligned_no_transform_mismatch_code() -> None:
    payload = build_world_trust_diagnostic_payload(
        WORLD_BAYES_H5_ADSTOCK_ALIGNED,
        aggregate=_agg(beta_mean=0.26),
        pilot_source={"model_spec_version": H5_MODEL_SPEC_VERSION, "h4c_baseline_comparison": {}},
        sample_run={"h5_transform_diagnostics": {"transform_mismatch_detected": False}},
    )
    assert "h5:transform_mismatch:adstock" not in payload["warning_codes"]
    assert payload["transform_alignment_status"] == "aligned"


def test_real_panel_assumption_codes_in_taxonomy() -> None:
    codes = derive_real_panel_transform_warning_codes(
        {
            "transform_registry_id": "bayes_h5_media_transform_registry_v1",
            "media_transforms_by_channel": {"tv": "identity"},
            "transform_mismatch_mode": "aligned",
        }
    )
    assert all(c in WARNING_TAXONOMY for c in codes)
    assert "h5:transform_unknown:real_panel" in codes


def test_convergence_taxonomy_and_evidence_gate() -> None:
    assert classify_convergence_status(rhat_max=2.09, divergence_count=9) == "failed_convergence"
    assert evidence_promotion_allowed("failed_convergence") is False
    assert "h5:convergence:failed" in WARNING_TAXONOMY
    assert "h5:evidence:blocked" in WARNING_TAXONOMY


def test_aggregate_warning_rates_recorded() -> None:
    mapping = build_trust_diagnostic_mapping(_minimal_pilot(), source_artifact="t.json")
    rates = mapping["aggregate_warning_rates"]
    assert "h5:production:block" not in rates
    assert isinstance(rates, dict)
