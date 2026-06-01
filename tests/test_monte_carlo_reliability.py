"""Phase 5F — Monte Carlo pilot characterization."""

from __future__ import annotations

from pathlib import Path

from mmm.validation.synthetic.monte_carlo_reliability import (
    CHARACTERIZATION_ARTIFACT,
    PROGRAM_VERSION,
    build_pilot_characterization,
    write_pilot_characterization,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_pilot_characterization() -> None:
    payload = build_pilot_characterization(REPO_ROOT)
    assert payload["program_version"] == PROGRAM_VERSION
    assert payload["tier"] == "tier_0_pilot"
    assert payload["capability_distributions"]
    assert payload["threshold_recommendations"]
    assert payload["trust_report_calibration"]


def test_write_artifact(tmp_path: Path) -> None:
    out = write_pilot_characterization(REPO_ROOT, output_dir=tmp_path)
    assert out.name == CHARACTERIZATION_ARTIFACT
    assert out.is_file()
