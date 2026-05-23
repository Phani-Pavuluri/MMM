"""Nightly certification aggregation."""

from __future__ import annotations

from mmm.evaluation.nightly_certification import run_nightly_certification_suite, write_nightly_certification_artifacts


def test_nightly_certification_suite_labels_categories() -> None:
    summary = run_nightly_certification_suite()
    assert summary["overall_status"] == "pass"
    cats = {c["category"] for c in summary["categories"]}
    assert "synthetic_certification" in cats
    assert "optimizer_certification" in cats
    assert "production_readiness" in cats
    assert summary["failures"] == []


def test_write_nightly_artifacts(tmp_path) -> None:
    summary = run_nightly_certification_suite()
    path = write_nightly_certification_artifacts(summary, tmp_path)
    assert path.is_file()
    assert (tmp_path / "optimizer_certification_report.json").is_file()
