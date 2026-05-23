"""Nightly certification aggregation (category-labeled failures for CI)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mmm.config.schema import MMMConfig
from mmm.governance.production_readiness import build_production_readiness_report
from mmm.governance.synthetic_certification import run_synthetic_certification_suite
from mmm.optimization.optimizer_certification import build_optimizer_certification_report


def _minimal_extension_report_for_nightly(synth: dict[str, Any], opt: dict[str, Any]) -> dict[str, Any]:
    """Stub extension report so nightly can exercise readiness rollup without a full train run."""
    return {
        "ridge_fit_summary": {
            "coef": [0.1, 0.2],
            "intercept": [4.6],
            "model_form": "semi_log",
            "best_params": {"decay": 0.1, "hill_half": 1e6, "hill_slope": 1.0},
        },
        "transform_policy": {"adstock": "geometric", "saturation": "hill"},
        "data_fingerprint": {"sha256_combined": "n" * 64},
        "reproducibility_certification_report": {
            "self_certification": False,
            "certification_status": "pass",
            "reproducibility_evidence": True,
            "identical_output": True,
        },
        "governance": {"approved_for_optimization": True},
        "model_release": {"state": "planning_allowed"},
        "calibration_readiness_report": {"stale_calibration_warning": None},
        "calibration_summary": {"replay_generalization_gap_severity": "low"},
        "synthetic_certification_report": synth,
        "optimizer_certification_report": opt,
    }


def run_nightly_certification_suite(
    *,
    extension_report: dict[str, Any] | None = None,
    config: MMMConfig | None = None,
) -> dict[str, Any]:
    """
    Run fast certification modules suitable for nightly CI.

    Returns a summary with per-category status so workflow logs identify failure type.
    """
    t0 = time.perf_counter()
    categories: list[dict[str, Any]] = []
    failures: list[str] = []

    def _record(category: str, status: str, detail: dict[str, Any] | None = None) -> None:
        categories.append({"category": category, "status": status, **(detail or {})})
        if status != "pass":
            failures.append(category)

    synth = run_synthetic_certification_suite(mode="exact")
    _record(
        "synthetic_certification",
        str(synth.get("certification_status", "fail")),
        {"n_pass": synth.get("n_pass"), "n_checks": synth.get("n_checks")},
    )

    opt = build_optimizer_certification_report()
    _record(
        "optimizer_certification",
        str(opt.get("certification_status", "fail")),
        {"n_pass": opt.get("n_pass"), "n_scenarios": opt.get("n_scenarios")},
    )

    er = dict(extension_report or _minimal_extension_report_for_nightly(synth, opt))
    er.setdefault("synthetic_certification_report", synth)
    er.setdefault("optimizer_certification_report", opt)
    from mmm.config.schema import Framework

    cfg = config or MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv", "search"]},
        budget={"enabled": True, "total_budget": 200.0},
    )
    readiness = build_production_readiness_report(cfg, er, synthetic_certification=synth, optimizer_certification=opt)
    er["production_readiness_report"] = readiness
    _record(
        "production_readiness",
        "pass" if readiness.get("approved_for_prod") else "fail",
        {
            "readiness_score": readiness.get("readiness_score"),
            "blocked_reasons": readiness.get("blocked_reasons"),
        },
    )

    runtime_s = time.perf_counter() - t0
    overall = "pass" if not failures else "fail"
    return {
        "report_version": "mmm_nightly_certification_v1",
        "overall_status": overall,
        "categories": categories,
        "failures": failures,
        "runtime_seconds": round(runtime_s, 3),
        "synthetic_certification_report": synth,
        "optimizer_certification_report": opt,
        "production_readiness_report": readiness,
        "governance_warnings": [
            "Nightly certification validates numerical and optimizer behavior — not causal incrementality.",
        ],
    }


def write_nightly_certification_artifacts(summary: dict[str, Any], out_dir: str | Path) -> Path:
    """Persist summary + per-category reports for CI artifact upload."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "nightly_certification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for key in (
        "synthetic_certification_report",
        "optimizer_certification_report",
        "production_readiness_report",
    ):
        blob = summary.get(key)
        if isinstance(blob, dict):
            (root / f"{key}.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    return summary_path
