"""Runtime synthetic certification parity with CI."""

from __future__ import annotations

from mmm.governance.synthetic_certification import EXACT_CHECK_NAMES, run_synthetic_certification_suite


def test_runtime_exact_suite_passes() -> None:
    rep = run_synthetic_certification_suite(mode="exact")
    assert rep["certification_level"] == "exact"
    assert rep["certification_status"] == "pass"
    assert rep["n_pass"] == len(EXACT_CHECK_NAMES)


def test_runtime_smoke_subset() -> None:
    rep = run_synthetic_certification_suite(mode="smoke")
    assert rep["certification_level"] in ("smoke", "exact")
    assert rep["n_pass"] >= 3
