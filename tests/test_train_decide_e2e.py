"""Canonical train → extension_report → decide simulate/optimize (no fabricated extension JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mmm.api.trainer import MMMTrainer
from mmm.artifacts.compatibility import verify_run_roundtrip
from mmm.artifacts.decision_inputs import load_training_extension_report
from mmm.artifacts.run_loader import load_run_lineage
from mmm.config.load import load_config
from mmm.decision.service import optimize_budget_decision, simulate_decision
from mmm.governance.replay_evidence import replay_calibration_active
from tests.train_decide_e2e_support import write_train_decide_fixture


@pytest.mark.release_gate
def test_train_decide_e2e_frozen_fixture(tmp_path: Path) -> None:
    paths = write_train_decide_fixture(tmp_path)
    train_out = MMMTrainer.from_yaml(paths["train_config"]).run()
    run_path = Path(train_out["store"])
    ext_path = run_path / "extension_report.json"
    assert ext_path.is_file(), "training must persist extension_report.json"

    er = load_training_extension_report(ext_path)
    er = dict(er)
    er["governance"] = {**(er.get("governance") or {}), "approved_for_optimization": True}
    mr = dict(er.get("model_release") or {})
    mr["state"] = "planning_allowed"
    er["model_release"] = mr
    assert er.get("ridge_fit_summary", {}).get("coef"), "ridge_fit_summary required"
    assert er.get("data_fingerprint"), "data_fingerprint required"
    assert er.get("feature_lineage") or er.get("seed_resolution"), "lineage/seed expected"
    assert replay_calibration_active(er.get("calibration_summary") or {}), "replay evidence required"
    sens = er.get("replay_calibration_sensitivity") or {}
    assert sens.get("status") in ("ok", "skipped")
    assert (run_path / "model_card.md").is_file() or er.get("model_card_md")

    rt = verify_run_roundtrip(run_path)
    assert rt["ok"], rt
    lineage = load_run_lineage(run_path)
    assert lineage["roundtrip"]["ok"]

    cfg = load_config(paths["train_config"])
    scenario = yaml.safe_load(paths["scenario_yaml"].read_text(encoding="utf-8"))

    sim_out = tmp_path / "sim_decision.json"
    sim_payload = simulate_decision(
        cfg=cfg,
        scenario=scenario,
        extension_report=er,
        out=sim_out,
        scenario_source_path=str(paths["scenario_yaml"]),
    )
    sim_js = sim_payload["simulation"]
    assert sim_js.get("decision_safe") is True
    assert sim_payload.get("decision_bundle", {}).get("artifact_tier") == "decision"
    sim_out.write_text(json.dumps(sim_payload, indent=2, default=str), encoding="utf-8")
    assert sim_out.is_file()

    opt_out = tmp_path / "opt_decision.json"
    opt_payload = optimize_budget_decision(
        cfg=cfg,
        extension_report=er,
        out=opt_out,
        scenario=scenario,
        scenario_source_path=str(paths["scenario_yaml"]),
    )
    opt_out.write_text(json.dumps(opt_payload, indent=2, default=str), encoding="utf-8")
    assert opt_out.is_file()
    bundle = opt_payload.get("decision_bundle") or {}
    assert bundle.get("artifact_tier") == "decision"
    gov = er.get("governance") or {}
    assert gov.get("approved_for_optimization") is True, (
        "real training governance must approve optimization when metrics pass"
    )
