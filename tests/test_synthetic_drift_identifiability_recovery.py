"""Phase 4B-5 — drift and identifiability recovery worlds."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.data.schema import PanelSchema
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import compute_dgp_series, materialize_dgp_world
from mmm.validation.synthetic.recovery_certification import (
    DRIFT_RECOVERY_WORLD_IDS,
    IDENTIFIABILITY_RECOVERY_WORLD_IDS,
    is_drift_recovery_eligible,
    is_identifiability_recovery_eligible,
    run_recovery_certification,
)
from mmm.validation.synthetic.reliability_truth import (
    build_world_011_truth,
    build_world_012_truth,
    panel_channel_correlation,
)
from mmm.validation.synthetic.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_011 = REPO_ROOT / "validation" / "worlds" / "WORLD-011-drift-recovery"
WORLD_012 = REPO_ROOT / "validation" / "worlds" / "WORLD-012-identifiability-recovery"


@pytest.fixture(scope="module")
def world_011_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w011") / "WORLD-011-drift-recovery"
    shutil.copytree(WORLD_011, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


@pytest.fixture(scope="module")
def world_012_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w012") / "WORLD-012-identifiability-recovery"
    shutil.copytree(WORLD_012, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_world_011_eligible_and_materializes(world_011_bundle: Path) -> None:
    truth = json.loads((WORLD_011 / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_id"] in DRIFT_RECOVERY_WORLD_IDS
    assert is_drift_recovery_eligible(truth)
    assert (world_011_bundle / "panel.parquet").is_file()


def test_drift_changepoint_in_diagnostics(world_011_bundle: Path) -> None:
    diag = pd.read_parquet(world_011_bundle / "dgp_diagnostics.parquet")
    assert "effective_beta" in diag.columns
    pre = diag.loc[diag["week_index"] < 8, "effective_beta"].unique()
    post = diag.loc[diag["week_index"] >= 8, "effective_beta"].unique()
    assert float(np.max(pre)) > float(np.min(post))


def test_drift_coef_recovery_skipped_not_false_pass(world_011_bundle: Path) -> None:
    result = run_recovery_certification(world_011_bundle)
    coef = next(c for c in result.checks if c.check_id == "REC-4B5-DRIFT-COEF")
    assert coef.status == "skipped"
    drift = next(c for c in result.checks if c.check_id == "REC-4B5-DRIFT")
    assert drift.status == "pass"
    assert drift.details.get("fit_degradation") is True
    assert drift.details.get("val_012_outcome") in ("pass", "warning")


def test_val_012_executes_for_world_011(world_011_bundle: Path) -> None:
    result = run_recovery_certification(world_011_bundle)
    drift = next(c for c in result.checks if c.check_id == "REC-4B5-DRIFT")
    assert drift.details.get("registry_validation_id") == "VAL-012"
    cert = run_world_certification(world_011_bundle, write_report=False)
    skipped = {r["check_id"] for r in cert.report["skipped_validations"]}
    assert "VAL-012" not in skipped


def test_corrupted_drift_truth_fails(world_011_bundle: Path) -> None:
    truth = json.loads((world_011_bundle / "world_truth.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(truth)
    bad["drift_truth"]["coefficient_drift"] = []
    result = run_recovery_certification(world_011_bundle, truth_override=bad)
    assert result.passed is False


def test_world_012_eligible_and_high_correlation(world_012_bundle: Path) -> None:
    truth = json.loads((WORLD_012 / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_id"] in IDENTIFIABILITY_RECOVERY_WORLD_IDS
    assert is_identifiability_recovery_eligible(truth)
    panel = pd.read_parquet(world_012_bundle / "panel.parquet")
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search", "social"), ())
    assert panel_channel_correlation(panel, schema, ["search", "social"]) >= 0.95


def test_identifiability_warning_and_unstable_coef(world_012_bundle: Path) -> None:
    result = run_recovery_certification(world_012_bundle)
    coef = next(c for c in result.checks if c.check_id == "REC-4B5-ID-COEF")
    assert coef.status == "skipped"
    assert coef.skip_reason == "recovery_marked_unstable"
    ident = next(c for c in result.checks if c.check_id == "REC-4B5-ID")
    assert ident.status == "pass"
    assert ident.details["identifiability_warning_emitted"] is True


def test_val_013_014_not_skipped_for_world_012(world_012_bundle: Path) -> None:
    cert = run_world_certification(world_012_bundle, write_report=False)
    skipped = {r["check_id"] for r in cert.report["skipped_validations"]}
    assert "VAL-013" not in skipped
    assert "VAL-014" not in skipped


def test_corrupted_collinearity_expectation_fails(world_012_bundle: Path) -> None:
    truth = json.loads((world_012_bundle / "world_truth.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(truth)
    bad["artifact_truth"]["expected_warnings"] = []
    result = run_recovery_certification(world_012_bundle, truth_override=bad)
    ident = next(c for c in result.checks if c.check_id == "REC-4B5-ID")
    assert ident.status == "fail"


def test_validator_passes_both_worlds(world_011_bundle: Path, world_012_bundle: Path) -> None:
    assert validate_bundle(world_011_bundle, max_level=3).passed
    assert validate_bundle(world_012_bundle, max_level=3).passed


def test_build_worlds_smoke() -> None:
    t11 = build_world_011_truth()
    t12 = build_world_012_truth()
    panel11, _ = compute_dgp_series(t11)
    panel12, _ = compute_dgp_series(t12)
    assert len(panel11) > 0
    assert len(panel12) > 0
