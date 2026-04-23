"""Canonical ``mmm decide …`` CLI vs deprecated top-level shims (parity + warnings)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner


def _strip_volatile_decision_payload(d: dict) -> dict:
    """Drop wall-clock bundle fields so back-to-back decide vs shim comparisons are stable."""
    out = json.loads(json.dumps(d))
    db = out.get("decision_bundle")
    if isinstance(db, dict):
        db.pop("created_at", None)
    return out


def _prod_panel_extension_and_yaml(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Small prod Ridge panel + extension_report + config YAML (full-panel path)."""
    csv_path = tmp_path / "panel.csv"
    rows = []
    for g in ("G1", "G2"):
        for w in range(1, 16):
            rows.append(
                {
                    "geo": g,
                    "week": w,
                    "c1": float(10 + w * 0.1),
                    "c2": float(8 + w * 0.05),
                    "revenue": float(100 + w + (1 if g == "G1" else 0)),
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ext = tmp_path / "ext.json"
    ext.write_text(
        json.dumps(
            {
                "ridge_fit_summary": {
                    "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
                    "coef": [0.05, 0.04],
                    "intercept": [3.0],
                },
                "governance": {"approved_for_optimization": True},
                "response_diagnostics": {"safe_for_optimization": True},
                "identifiability": {"identifiability_score": 0.5},
                "panel_qa": {"max_severity": "info", "issues": []},
                "model_release": {"state": "planning_allowed", "reasons": [], "triggers": {}},
                "experiment_matching": {"replay_ok": True},
            }
        ),
        encoding="utf-8",
    )
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        f"""
run_environment: prod
allow_unsafe_decision_apis: false
cv:
  mode: rolling
objective:
  normalization_profile: strict_prod
  named_profile: ridge_bo_standard_v1
prod_canonical_modeling_contract_id: ridge_bo_semi_log_calendar_cv_v1
data:
  path: {csv_path}
  geo_column: geo
  week_column: week
  target_column: revenue
  channel_columns: [c1, c2]
  control_columns: []
  data_version_id: shim-test-snapshot
budget:
  total_budget: 100
extensions:
  optimization_gates:
    enabled: true
""",
        encoding="utf-8",
    )
    return yaml, ext, csv_path


def test_decide_optimize_budget_prod_json_matches_shim(tmp_path: Path) -> None:
    from mmm.cli import main as cli_main

    yaml, ext, _ = _prod_panel_extension_and_yaml(tmp_path)
    out_decide = tmp_path / "opt_decide.json"
    out_shim = tmp_path / "opt_shim.json"
    runner = CliRunner()
    argv_decide = [
        "decide",
        "optimize-budget",
        str(yaml),
        "--extension-report",
        str(ext),
        "--out",
        str(out_decide),
    ]
    r1 = runner.invoke(cli_main.app, argv_decide)
    assert r1.exit_code == 0, r1.stdout + (r1.stderr or "")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r2 = runner.invoke(
            cli_main.app,
            [
                "optimize-budget",
                str(yaml),
                "--extension-report",
                str(ext),
                "--out",
                str(out_shim),
            ],
        )
    assert r2.exit_code == 0, r2.stdout + (r2.stderr or "")
    dep = [
        w
        for w in rec
        if issubclass(w.category, DeprecationWarning) and "mmm decide optimize-budget" in str(w.message)
    ]
    assert dep, "expected DeprecationWarning pointing at mmm decide optimize-budget"

    a = json.loads(out_decide.read_text(encoding="utf-8"))
    b = json.loads(out_shim.read_text(encoding="utf-8"))
    assert _strip_volatile_decision_payload(a) == _strip_volatile_decision_payload(b)


def test_decide_simulate_prod_json_matches_shim(tmp_path: Path) -> None:
    from mmm.cli import main as cli_main

    yaml, ext, _ = _prod_panel_extension_and_yaml(tmp_path)
    scen = tmp_path / "scen.yaml"
    scen.write_text(
        """
candidate_spend:
  c1: 15.0
  c2: 12.0
""",
        encoding="utf-8",
    )
    out_decide = tmp_path / "sim_decide.json"
    out_shim = tmp_path / "sim_shim.json"
    runner = CliRunner()
    r1 = runner.invoke(
        cli_main.app,
        [
            "decide",
            "simulate",
            str(yaml),
            "--scenario",
            str(scen),
            "--extension-report",
            str(ext),
            "--out",
            str(out_decide),
        ],
    )
    assert r1.exit_code == 0, r1.stdout + (r1.stderr or "")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r2 = runner.invoke(
            cli_main.app,
            [
                "simulate",
                str(yaml),
                "--scenario",
                str(scen),
                "--extension-report",
                str(ext),
                "--out",
                str(out_shim),
            ],
        )
    assert r2.exit_code == 0, r2.stdout + (r2.stderr or "")
    dep = [
        w
        for w in rec
        if issubclass(w.category, DeprecationWarning) and "mmm decide simulate" in str(w.message)
    ]
    assert dep, "expected DeprecationWarning pointing at mmm decide simulate"

    a = json.loads(out_decide.read_text(encoding="utf-8"))
    b = json.loads(out_shim.read_text(encoding="utf-8"))
    assert _strip_volatile_decision_payload(a) == _strip_volatile_decision_payload(b)


@pytest.mark.parametrize("cmd", [["decide", "simulate"], ["simulate"]])
def test_prod_simulate_requires_out_parity(tmp_path: Path, cmd: list[str]) -> None:
    from mmm.cli import main as cli_main

    yaml, ext, _ = _prod_panel_extension_and_yaml(tmp_path)
    scen = tmp_path / "scen.yaml"
    scen.write_text("candidate_spend:\n  c1: 15.0\n  c2: 12.0\n", encoding="utf-8")
    runner = CliRunner()
    base = [str(yaml), "--scenario", str(scen), "--extension-report", str(ext)]
    if cmd == ["simulate"]:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = runner.invoke(cli_main.app, cmd + base)
    else:
        result = runner.invoke(cli_main.app, cmd + base)
    assert result.exit_code == 2
    err = ((result.stderr or "") + (result.stdout or "")).lower()
    assert "requires" in err and "--out" in err


@pytest.mark.parametrize("cmd", [["decide", "optimize-budget"], ["optimize-budget"]])
def test_prod_optimize_requires_out_parity(tmp_path: Path, cmd: list[str]) -> None:
    from mmm.cli import main as cli_main

    yaml, ext, _ = _prod_panel_extension_and_yaml(tmp_path)
    runner = CliRunner()
    argv = [*cmd, str(yaml), "--extension-report", str(ext)]
    if cmd == ["optimize-budget"]:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = runner.invoke(cli_main.app, argv)
    else:
        result = runner.invoke(cli_main.app, argv)
    assert result.exit_code == 2
    err = ((result.stderr or "") + (result.stdout or "")).lower()
    assert "requires" in err and "--out" in err


def test_root_help_separates_train_diagnose_decide() -> None:
    from mmm.cli import main as cli_main

    runner = CliRunner()
    r = runner.invoke(cli_main.app, ["--help"])
    assert r.exit_code == 0
    h = (r.stdout or "").lower()
    assert "train" in h and "decide" in h
    assert "diagnose" in h or "evaluate" in h
    assert "shim" in h or "decide" in h


def test_decide_group_help_mentions_governance_and_curves() -> None:
    from mmm.cli import main as cli_main

    runner = CliRunner()
    r = runner.invoke(cli_main.app, ["decide", "--help"])
    assert r.exit_code == 0
    h = r.stdout or ""
    assert "simulate" in h.lower() and "optimize" in h.lower()
    combined = h.lower()
    assert "governance" in combined or "prod" in combined
    assert "diagnostic" in combined or "curve" in combined
