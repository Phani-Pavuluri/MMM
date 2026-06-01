"""Phase 4B-2 — train/decide recovery certification on WORLD-008."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.recovery_certification import (
    RECOVERY_ELIGIBLE_WORLD_IDS,
    compute_analytic_delta_mu,
    is_recovery_eligible,
    run_recovery_certification,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_008 = REPO_ROOT / "validation" / "worlds" / "WORLD-008-exact-recovery"


@pytest.fixture(scope="module")
def world_008_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w008") / "WORLD-008-exact-recovery"
    shutil.copytree(WORLD_008, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_world_008_is_recovery_eligible() -> None:
    truth = json.loads((WORLD_008 / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_id"] in RECOVERY_ELIGIBLE_WORLD_IDS
    assert is_recovery_eligible(truth)


def test_train_path_completes(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    train = next(c for c in result.checks if c.check_id == "REC-4B2-TRAIN")
    assert train.status == "pass"
    assert result.train_decide_summary.get("train_completed") is True


def test_coefficient_recovery_executes(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    coef = next(c for c in result.checks if c.check_id == "REC-4B2-001")
    assert coef.status in ("pass", "fail")
    assert coef.metric_kind == "provisional_statistical"
    assert "fitted_beta" in coef.details


def test_transform_consistency_executes(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    formula = next(c for c in result.checks if c.check_id == "REC-4B2-004")
    assert formula.status == "pass"
    assert formula.metric_kind == "exact_formula"
    ad = next(c for c in result.checks if c.check_id == "REC-4B2-002")
    assert ad.status in ("pass", "fail")


def test_analytic_delta_mu_available(world_008_bundle: Path) -> None:
    truth = json.loads((world_008_bundle / "world_truth.json").read_text(encoding="utf-8"))
    import pandas as pd

    from mmm.data.schema import PanelSchema
    from mmm.validation.synthetic.recovery_certification import build_recovery_mmm_config

    panel = pd.read_parquet(world_008_bundle / "panel.parquet")
    config = build_recovery_mmm_config(truth, panel_path=world_008_bundle / "panel.parquet")
    schema = PanelSchema(
        geo_column="geo_id",
        week_column="week_start_date",
        target_column="revenue",
        channel_columns=("search", "social", "display"),
        control_columns=(),
    )
    sc = truth["decision_truth"]["scenarios"][0]
    analytic = compute_analytic_delta_mu(truth, panel, schema, config, sc)
    assert analytic != 0.0


def test_delta_mu_recovery_runs_with_analytic_truth(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    delta = next(c for c in result.checks if c.check_id == "REC-4B2-005")
    assert delta.status in ("pass", "fail")
    assert delta.skip_reason is None
    assert "analytic_true_delta_mu" in delta.details
    assert delta.details.get("placeholder_replaced") is True


def test_train_decide_fingerprint_passes(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    fp = next(c for c in result.checks if c.check_id == "REC-4B2-007")
    assert fp.status == "pass"


def test_decision_artifact_compatibility(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    dec = next(c for c in result.checks if c.check_id == "REC-4B2-008")
    assert dec.status == "pass"
    assert "delta_mu" in dec.details


def test_optimizer_recovery_skipped_not_passed(world_008_bundle: Path) -> None:
    result = run_recovery_certification(world_008_bundle)
    opt = next(c for c in result.checks if c.check_id == "REC-4B2-006")
    assert opt.status == "skipped"
    assert opt.skip_reason == "requires_optimizer_truth_thresholds"


def test_corrupted_coef_truth_fails_recovery(world_008_bundle: Path) -> None:
    truth = json.loads((world_008_bundle / "world_truth.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(truth)
    bad["coefficient_truth"]["true_beta_by_channel"]["search"] = 99.0
    result = run_recovery_certification(world_008_bundle, truth_override=bad)
    coef = next(c for c in result.checks if c.check_id == "REC-4B2-001")
    assert coef.status == "fail"


def test_certification_runner_integrates_recovery(world_008_bundle: Path) -> None:
    cert = run_world_certification(world_008_bundle, write_report=False, include_recovery=True)
    assert "recovery_certification_version" in cert.report
    assert "coefficient_recovery" in cert.report
    skipped_ids = {r["check_id"] for r in cert.report["skipped_validations"]}
    assert "VAL-001" not in skipped_ids
    assert "VAL-004" not in skipped_ids
    rec_ids = {c["check_id"] for c in cert.report["recovery_validation_results"]}
    assert "REC-4B2-001" in rec_ids
    assert "REC-4B2-006" in rec_ids


def test_non_recovery_world_skips_recovery(tmp_path: Path) -> None:
    from mmm.validation.synthetic.materializer import materialize_world

    src = REPO_ROOT / "validation" / "worlds" / "WORLD-001-baseline"
    bundle = tmp_path / "WORLD-001-baseline"
    shutil.copytree(src, bundle, dirs_exist_ok=True)
    materialize_world(bundle, overwrite=True)
    result = run_recovery_certification(bundle)
    assert result.recovery_results.get("skipped") is True
