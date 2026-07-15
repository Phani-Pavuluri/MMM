import json
from pathlib import Path

from mmm.contracts.calibration_treatment import MMMCalibrationTreatmentLineage
from mmm.contracts.diagnostics_limitations import MMMDiagnosticsLimitations
from mmm.contracts.mip_failure import MMMFailurePacket
from mmm.contracts.run_manifest import MMMRunManifest


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export" / "golden_v1"


def test_golden_fixture_index_is_deterministic_and_references_public_contracts():
    index = json.loads((ROOT / "index.json").read_text())
    assert index["schema_version"] == "mmm_producer_golden_fixture_set_v1"
    scenarios = index["scenarios"]
    assert len({item["scenario_id"] for item in scenarios}) == len(scenarios) == 8
    assert {item["scenario_id"] for item in scenarios} >= {"ridge_success", "warning_restriction", "blocked_insufficient_history", "failed_input_validation", "calibration_scope_rejection", "unsupported_extrapolation", "bayesian_research_only", "mixed_claims"}
    for scenario in scenarios:
        for kind, relative in scenario["components"].items():
            payload = json.loads((ROOT / relative).resolve().read_text())
            if kind == "manifest":
                MMMRunManifest.model_validate(payload)
            elif kind == "failure":
                MMMFailurePacket.model_validate(payload)
            elif kind == "calibration":
                MMMCalibrationTreatmentLineage.model_validate(payload)
            else:
                MMMDiagnosticsLimitations.model_validate(payload)
        assert "/Users/" not in json.dumps(scenario)


def test_golden_fixture_scenarios_preserve_terminal_and_safety_boundaries():
    index = json.loads((ROOT / "index.json").read_text())
    forbidden = ("/Users/", "Traceback", "stack trace", "DataFrame", "Exception(", "secret", "password")
    for scenario in index["scenarios"]:
        text = json.dumps(scenario, sort_keys=True)
        assert not any(value in text for value in forbidden)
        components = scenario["components"]
        manifest = None
        if "manifest" in components:
            manifest = MMMRunManifest.model_validate(json.loads((ROOT / components["manifest"]).resolve().read_text()))
            assert manifest.status.value == scenario["run_status"]
        if scenario["terminal_outcome"] == "success":
            assert manifest is None or manifest.failure_packet is None
        else:
            assert "failure" in components
            failure = MMMFailurePacket.model_validate(json.loads((ROOT / components["failure"]).resolve().read_text()))
            assert failure.code.value == scenario["failure_code"]
            assert manifest is None or manifest.failure_packet is not None
        if scenario["scenario_id"] == "bayesian_research_only":
            diagnostics = MMMDiagnosticsLimitations.model_validate(json.loads((ROOT / components["diagnostics"]).resolve().read_text()))
            assert diagnostics.limitations[0].research_only is True
