"""Phase 4B-3 — optimizer recovery world WORLD-009."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from mmm.data.schema import PanelSchema
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.optimizer_truth import (
    build_world_009_truth,
    grid_search_true_optimum,
    validate_optimizer_surface,
)
from mmm.validation.synthetic.recovery_certification import (
    OPTIMIZER_RECOVERY_WORLD_IDS,
    build_recovery_mmm_config,
    is_optimizer_recovery_eligible,
    run_recovery_certification,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_009 = REPO_ROOT / "validation" / "worlds" / "WORLD-009-optimizer-recovery"


@pytest.fixture(scope="module")
def world_009_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w009") / "WORLD-009-optimizer-recovery"
    shutil.copytree(WORLD_009, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_world_009_present_and_eligible() -> None:
    assert (WORLD_009 / "world_truth.json").is_file()
    truth = json.loads((WORLD_009 / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_id"] in OPTIMIZER_RECOVERY_WORLD_IDS
    assert is_optimizer_recovery_eligible(truth)
    opt_budget = truth["decision_truth"]["true_optimal_budget"]
    assert opt_budget["high_return"] > opt_budget["low_return"]


def test_world_009_materializes(world_009_bundle: Path) -> None:
    assert (world_009_bundle / "panel.parquet").is_file()
    assert (world_009_bundle / "dgp_diagnostics.parquet").is_file()


def test_grid_optimum_not_flat(tmp_path: Path) -> None:
    from mmm.validation.synthetic.dgp_materializer import compute_dgp_series

    truth = build_world_009_truth()
    panel, _ = compute_dgp_series(truth)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    schema = PanelSchema(
        "geo_id",
        "week_start_date",
        "revenue",
        ("high_return", "low_return"),
        (),
    )
    opt = grid_search_true_optimum(truth, panel, schema, config, total_budget=40.0, channel_max=40.0)
    validate_optimizer_surface(opt)
    assert opt["true_optimal_budget"]["high_return"] > opt["true_optimal_budget"]["low_return"]


def test_flat_optimum_rejected() -> None:
    flat = {
        "true_optimal_budget": {"high_return": 20.0, "low_return": 20.0},
        "optimum_interior": True,
        "optimum_lift_over_bau": 0.001,
    }
    with pytest.raises(ValueError, match="flat|equal"):
        validate_optimizer_surface(flat, min_lift=0.05)


def test_train_and_optimizer_path_complete(world_009_bundle: Path) -> None:
    result = run_recovery_certification(world_009_bundle)
    train = next(c for c in result.checks if c.check_id == "REC-4B3-TRAIN")
    path = next(c for c in result.checks if c.check_id == "REC-4B3-OPT-PATH")
    assert train.status == "pass"
    assert path.status == "pass"


def test_val_005_executes_not_skipped(world_009_bundle: Path) -> None:
    result = run_recovery_certification(world_009_bundle)
    opt = next(c for c in result.checks if c.check_id == "REC-4B3-OPT")
    assert opt.status in ("pass", "fail")
    assert opt.skip_reason is None
    assert opt.details.get("registry_validation_id") == "VAL-005"
    cert = run_world_certification(world_009_bundle, write_report=False)
    skipped = {r["check_id"] for r in cert.report["skipped_validations"]}
    assert "VAL-005" not in skipped


def test_optimizer_allocation_band(world_009_bundle: Path) -> None:
    result = run_recovery_certification(world_009_bundle)
    opt = next(c for c in result.checks if c.check_id == "REC-4B3-OPT")
    assert opt.status == "pass"
    details = opt.details
    band = details["expected_allocation_band"]
    assert details["high_return_budget_share"] >= band["high_return_min_budget_share"]
    assert float(details["budget_conservation_error"]) <= 0.5


def test_budget_conserved(world_009_bundle: Path) -> None:
    result = run_recovery_certification(world_009_bundle)
    opt = next(c for c in result.checks if c.check_id == "REC-4B3-OPT")
    assert float(opt.details["budget_conservation_error"]) <= 0.5


def test_corrupted_decision_truth_fails(world_009_bundle: Path) -> None:
    truth = json.loads((world_009_bundle / "world_truth.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(truth)
    bad["decision_truth"]["true_optimal_budget"] = {"high_return": 5.0, "low_return": 35.0}
    result = run_recovery_certification(world_009_bundle, truth_override=bad)
    opt = next(c for c in result.checks if c.check_id == "REC-4B3-OPT")
    assert opt.status == "fail"


def test_certification_report_includes_optimizer_recovery(world_009_bundle: Path) -> None:
    cert = run_world_certification(world_009_bundle, write_report=False)
    assert "optimizer_recovery" in cert.report
    assert cert.report.get("optimizer_recovery_status") in ("pass", "fail")
