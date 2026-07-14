"""Regression checks for MMM's producer-only MIP handoff boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mmm.contracts.mip_export import (
    MMMExportBundle,
    parse_export_artifact,
    parse_export_bundle,
    validate_mmm_export_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
MMM_SOURCE = ROOT / "mmm"
PRODUCER_FIXTURE = ROOT / "tests/fixtures/mip_export/readiness_only_bundle.json"
CONSUMER_FIXTURE_DIRECTORY = ROOT / "tests/fixtures" / ("mmm" + "_export")
REMOVED_MODULES = (
    "mmm.contracts." + "mmm_export_bundle",
    "mmm.llm." + "mmm_export_answerability",
)
REMOVED_SYMBOLS = (
    "Parsed" + "MMMExportBundle",
    "parse_" + "mmm_export_bundle",
    "load_" + "mmm_export_bundle",
    "MMM" + "Intent",
    "Answerability" + "Result",
    "evaluate_" + "mmm_export_answerability",
)


def test_mip_consumer_modules_and_symbols_do_not_ship_in_mmm() -> None:
    for module in REMOVED_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        assert spec is None
    source = "\n".join(path.read_text(encoding="utf-8") for path in MMM_SOURCE.rglob("*.py"))
    assert all(symbol not in source for symbol in REMOVED_SYMBOLS)
    assert "Cannot" + " say:" not in source


def test_mmm_has_no_orphaned_consumer_answerability_fixtures() -> None:
    assert not CONSUMER_FIXTURE_DIRECTORY.exists()
    assert not any(
        "fixtures/" + "mmm_export" in path.as_posix()
        for path in ROOT.rglob("*.py")
        if path != Path(__file__)
    )


def test_producer_schema_validation_and_technical_claim_evidence_remain() -> None:
    raw = json.loads(PRODUCER_FIXTURE.read_text(encoding="utf-8"))
    bundle = parse_export_bundle(raw)

    assert isinstance(bundle, MMMExportBundle)
    assert validate_mmm_export_bundle(bundle) == []
    artifact = parse_export_artifact(bundle.artifacts[0])
    assert "readiness_explanation_allowed" in artifact.allowed_claims
    assert "budget_shift_recommendation" in artifact.forbidden_claims
