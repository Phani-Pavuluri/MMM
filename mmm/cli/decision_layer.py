"""Shared decision CLI layer: prod gates, model_release, panel_qa gate, decision bundles.

Used by ``mmm decide …`` and by compatibility shims for top-level ``simulate`` / ``optimize-budget``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import typer

from mmm.config.load import load_config
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.decision.extension_gate import optimization_gate_result
from mmm.decision.legacy_budget_optimization import run_legacy_diagnostic_optimize_budget
from mmm.decision.service import load_scenario_yaml, optimize_budget_decision, simulate_decision
from mmm.governance.decision_safety import MSG_OPTIMIZATION_BLOCKED
from mmm.governance.policy import PolicyError

CANONICAL_SIMULATE = "mmm decide simulate"
CANONICAL_OPTIMIZE = "mmm decide optimize-budget"


def emit_shim_deprecation(*, legacy_subcommand: str, canonical: str) -> None:
    msg = (
        f"CLI compatibility shim `mmm {legacy_subcommand}` is deprecated; use `{canonical}` "
        "(canonical governance-checked decision path). This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    typer.secho(f"DeprecationWarning: {msg}", fg=typer.colors.YELLOW, err=True)


def _optimization_gate_and_extension(
    cfg: MMMConfig, extension_report: Path | None
) -> tuple[dict | None, dict, bool, Any]:
    """Load extension JSON if present; run optimization safety gate."""
    er_data: dict | None = None
    ext_present = extension_report is not None and extension_report.exists()
    if ext_present and extension_report is not None:
        er_data = json.loads(extension_report.read_text(encoding="utf-8"))
    gov = er_data.get("governance", {}) if isinstance(er_data, dict) else {}
    gr = optimization_gate_result(cfg, er_data, extension_report_present=ext_present)
    return er_data, gov, ext_present, gr


def run_decision_simulate(
    *,
    config: Path,
    scenario: Path,
    extension_report: Path,
    out: Path | None,
) -> int:
    """Full-panel Δμ simulate; returns shell exit code (0 success). Delegates to ``decision.service``."""
    cfg = load_config(config)
    if not extension_report.exists():
        typer.secho("--extension-report must exist and include ridge_fit_summary.", fg=typer.colors.RED, err=True)
        return 1
    er = json.loads(extension_report.read_text(encoding="utf-8"))
    rs = er.get("ridge_fit_summary")
    if not isinstance(rs, dict) or not rs.get("coef"):
        typer.secho("extension_report must contain ridge_fit_summary.coef", fg=typer.colors.RED, err=True)
        return 1
    try:
        raw = load_scenario_yaml(scenario)
        payload = simulate_decision(cfg=cfg, scenario=raw, extension_report=er, out=out)
    except PolicyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        return 2
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        return 1
    text = json.dumps(payload, indent=2, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)
    return 0


def run_decision_optimize_budget(
    *,
    config: Path,
    extension_report: Path | None,
    curve_bundle: Path | None,
    allow_unsafe_decision_apis: bool,
    legacy_diagnostic_curve_optimizer: bool,
    out: Path | None,
) -> int:
    """Budget optimization CLI; returns shell exit code."""
    cfg = load_config(config)
    if cfg.run_environment == RunEnvironment.PROD and (extension_report is None or not extension_report.exists()):
        typer.secho(
            "Production optimize-budget requires an existing --extension-report JSON from training.",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    er_data, _, ext_present, gr = _optimization_gate_and_extension(cfg, extension_report)
    if not gr.allowed:
        typer.echo(json.dumps({"allowed": False, "reasons": gr.reasons, "audit": gr.audit}, indent=2))
        return 2

    ridge_summary = er_data.get("ridge_fit_summary") if isinstance(er_data, dict) else None
    full_model_ready = bool(cfg.data.path and isinstance(ridge_summary, dict) and ridge_summary.get("coef"))

    if cfg.run_environment == RunEnvironment.PROD and not full_model_ready:
        typer.secho(
            "Production optimize-budget requires config.data.path to the training panel and "
            "extension_report.json containing ridge_fit_summary (full-panel Δμ objective via simulate).",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    if cfg.run_environment == RunEnvironment.PROD and full_model_ready and out is None:
        typer.secho(
            "Production optimize-budget requires --out PATH to persist the decision bundle and optimization JSON.",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    if full_model_ready:
        assert er_data is not None  # full_model_ready implies path + ridge in extension
        try:
            payload = optimize_budget_decision(cfg=cfg, extension_report=er_data, out=out)
        except PolicyError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            return 2
        except (ValueError, KeyError) as e:
            typer.secho(f"Invalid ridge_fit_summary or inputs: {e}", fg=typer.colors.RED, err=True)
            return 2
        text = json.dumps(payload, indent=2, default=str)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            typer.echo(f"wrote {out}")
        if cfg.run_environment != RunEnvironment.PROD or out is None:
            typer.echo(text)
        return 0

    if not cfg.allow_unsafe_decision_apis or not allow_unsafe_decision_apis:
        typer.secho(MSG_OPTIMIZATION_BLOCKED, fg=typer.colors.YELLOW, err=True)
        return 2

    if cfg.run_environment == RunEnvironment.PROD:
        typer.secho(
            "Production forbids curve-only optimization; use data.path + extension_report with ridge_fit_summary.",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    if not legacy_diagnostic_curve_optimizer:
        typer.secho(
            "Full-panel inputs unavailable. Pass --legacy-diagnostic-curve-optimizer (non-prod) to use "
            "deprecated curve interpolation, or provide data.path + ridge_fit_summary.",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    if os.environ.get("MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET", "").strip() != "1":
        typer.secho(
            "Legacy curve-only budget optimization is quarantined: set MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET=1 "
            "in the environment to acknowledge non-canonical, non-decision-safe behavior.",
            fg=typer.colors.RED,
            err=True,
        )
        return 2

    er_payload: dict[str, Any] = er_data if isinstance(er_data, dict) else {}
    try:
        payload = run_legacy_diagnostic_optimize_budget(
            cfg=cfg,
            er_data=er_payload,
            curve_bundle=curve_bundle,
            allow_unsafe_decision_apis=allow_unsafe_decision_apis,
        )
    except PolicyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        return 2
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        return 2
    typer.secho(
        "Using legacy diagnostic budget path (non-canonical); prefer full-panel optimize_budget_decision.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    typer.echo(payload.get("result", ""))
    return 0
