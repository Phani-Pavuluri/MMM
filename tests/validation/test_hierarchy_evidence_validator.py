"""Tests for Bayes-H2b hierarchy_evidence_validator (no-fit fixture contract)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.validation.synthetic.hierarchy_evidence_validator import (
    REQUIRED_BUNDLE_FILES,
    REQUIRED_FIXTURE_SECTIONS,
    SMOKE_ID,
    load_hierarchy_evidence_world,
    validate_hierarchy_evidence_world,
    validate_world_catalog,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "validation/worlds/world_catalog.index.json"
BAYES_WORLD_IDS = (
    "WORLD-BAYES-GEOX-LOCAL",
    "WORLD-BAYES-CLS-NATIONAL",
    "WORLD-BAYES-CONFLICT",
    "WORLD-BAYES-STALE",
    "WORLD-BAYES-MISSING-SE",
    "WORLD-BAYES-SPARSE-GEO",
    "WORLD-BAYES-ESTIMAND-EXCLUDE",
)


def _bayes_bundle(world_id: str) -> Path:
    return REPO_ROOT / "validation/worlds" / world_id


@pytest.fixture
def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_lists_seven_bayes_worlds(catalog: dict) -> None:
    bayes = [e for e in catalog["entries"] if e.get("validation_family") == "bayes-hierarchy-evidence"]
    assert len(bayes) == 7
    assert {e["world_id"] for e in bayes} == set(BAYES_WORLD_IDS)


@pytest.mark.parametrize("world_id", BAYES_WORLD_IDS)
def test_world_has_required_files(world_id: str) -> None:
    bundle = _bayes_bundle(world_id)
    for name in REQUIRED_BUNDLE_FILES:
        assert (bundle / name).is_file(), f"missing {name} in {world_id}"


@pytest.mark.parametrize("world_id", BAYES_WORLD_IDS)
def test_world_json_files_parse(world_id: str) -> None:
    bundle = _bayes_bundle(world_id)
    json.loads((bundle / "hierarchy_evidence_fixture.json").read_text(encoding="utf-8"))
    json.loads((bundle / "hierarchy_spec.json").read_text(encoding="utf-8"))
    json.loads((bundle / "calibration_signals.json").read_text(encoding="utf-8"))
    json.loads((bundle / "estimand_allowlist.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("world_id", BAYES_WORLD_IDS)
def test_validate_world_passes(world_id: str) -> None:
    report = validate_hierarchy_evidence_world(_bayes_bundle(world_id))
    assert report["status"] == "pass", report.get("failure_reasons")
    assert report["world_id"] == world_id
    val_rows = [r for r in report["assertion_results"] if r["validation_id"].startswith("VAL-BAYES-0")]
    assert len(val_rows) >= 12
    assert all(r["outcome"] == "pass" for r in val_rows if r["validation_id"] in {
        f"VAL-BAYES-{i:03d}" for i in range(1, 13)
    })


def test_validate_world_deterministic() -> None:
    path = _bayes_bundle("WORLD-BAYES-GEOX-LOCAL")
    assert validate_hierarchy_evidence_world(path) == validate_hierarchy_evidence_world(path)


def test_validate_world_catalog_passes() -> None:
    summary = validate_world_catalog(CATALOG_PATH)
    assert summary["status"] == "pass", summary.get("failure_reasons")
    assert summary["world_count"] == 7
    smoke_rows = [r for r in summary["assertion_results"] if r["validation_id"] == SMOKE_ID]
    assert any(r["outcome"] == "pass" for r in smoke_rows)


def test_val_bayes_h2b_smoke() -> None:
    """VAL-BAYES-H2B-SMOKE: all seven worlds pass contract validation."""
    summary = validate_world_catalog(CATALOG_PATH)
    assert summary["status"] == "pass"
    for report in summary["world_reports"]:
        assert report["status"] == "pass", (report["world_id"], report.get("failure_reasons"))


def test_missing_file_blocked(tmp_path: Path) -> None:
    src = _bayes_bundle("WORLD-BAYES-GEOX-LOCAL")
    dest = tmp_path / "WORLD-BAYES-GEOX-LOCAL"
    dest.mkdir()
    for name in REQUIRED_BUNDLE_FILES:
        if name != "calibration_signals.json":
            (dest / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    report = validate_hierarchy_evidence_world(dest)
    assert report["status"] in ("blocked", "fail")
    assert report["failure_reasons"]


def test_malformed_json_fails(tmp_path: Path) -> None:
    src = _bayes_bundle("WORLD-BAYES-GEOX-LOCAL")
    dest = tmp_path / "WORLD-BAYES-GEOX-LOCAL"
    dest.mkdir()
    for name in REQUIRED_BUNDLE_FILES:
        content = (src / name).read_text(encoding="utf-8")
        if name == "hierarchy_spec.json":
            content = "{ not valid json"
        (dest / name).write_text(content, encoding="utf-8")
    report = validate_hierarchy_evidence_world(dest)
    assert report["status"] == "fail"
    assert any("malformed" in r.lower() or "JSON" in r for r in report["failure_reasons"])


def test_missing_expected_section_fails(tmp_path: Path) -> None:
    src = _bayes_bundle("WORLD-BAYES-GEOX-LOCAL")
    dest = tmp_path / "WORLD-BAYES-GEOX-LOCAL"
    dest.mkdir()
    for name in REQUIRED_BUNDLE_FILES:
        (dest / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    fixture = json.loads((dest / "hierarchy_evidence_fixture.json").read_text(encoding="utf-8"))
    for key in REQUIRED_FIXTURE_SECTIONS:
        fixture.pop(key, None)
    (dest / "hierarchy_evidence_fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    report = validate_hierarchy_evidence_world(dest)
    assert report["status"] == "fail"
    assert any("missing fixture sections" in r for r in report["failure_reasons"])


def test_load_hierarchy_evidence_world() -> None:
    loaded = load_hierarchy_evidence_world(_bayes_bundle("WORLD-BAYES-CONFLICT"))
    assert loaded["world_id"] == "WORLD-BAYES-CONFLICT"
    assert loaded["calibration_signals"]
    assert loaded["fixture"]["expected_conflicts"]
