"""Command-line interface for MMM runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
import yaml

from mmm.api.trainer import MMMTrainer
from mmm.calibration.units_io import write_calibration_units_to_json
from mmm.config.load import load_config
from mmm.config.schema import RunEnvironment
from mmm.economics.canonical import economics_contract_for_curve_bundles
from mmm.optimization.budget.curve_bundles_io import gather_curve_bundles_from_dict, gather_curve_bundles_from_path
from mmm.reporting.builder import ReportBuilder
from mmm.reporting.roi_sections import curve_bundles_to_roi_summary
from mmm.services.calibration_service import run_calibration_extensions

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "MMM CLI: ``train`` fits models and writes artifacts (not a Δμ decision). ``diagnose``/``evaluate`` read "
        "extension reports (diagnostic summaries). For allocation-grade Δμ, use ``mmm decide …``; it runs config + "
        "governance + semantic checks in prod (fail-closed). Top-level ``simulate``/``optimize-budget`` shims delegate "
        "to the same decide logic but are deprecated."
    ),
)

decide_app = typer.Typer(
    help=(
        "Full-panel Δμ simulation and budget optimization. PROD fail-closed: extension_report, "
        "``model_release.state=planning_allowed``, optimization gates, panel QA rules, ``--out``, lineage, "
        "``artifact_tier=decision``, and semantic contract validation on ``decision_bundle`` + ``simulation``. "
        "Curves are not used here—use ``simulate-diagnostic-curves`` (diagnostic tier only). "
        "Bayesian / posterior-draw outputs elsewhere in the repo are **research or diagnostic only** under "
        "current prod policy—not approved for prod Δμ budgeting or CLI ``decide`` surfaces."
    ),
    add_completion=False,
    no_args_is_help=True,
)


@decide_app.command("simulate")
def decide_simulate(
    config: Annotated[Path, typer.Argument(help="Resolved YAML (must include data.path for full-panel Δμ).")],
    scenario: Annotated[
        Path,
        typer.Option("--scenario", help="Scenario YAML (candidate spend / paths / overlays)."),
    ],
    extension_report: Annotated[
        Path,
        typer.Option("--extension-report", help="extension_report.json with ridge_fit_summary."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write JSON decision artifact (required in PROD)."),
    ] = None,
) -> None:
    """Canonical governance-checked full-panel Δμ simulate (delegates to ``decision.service``)."""
    from mmm.cli.decision_layer import run_decision_simulate

    code = run_decision_simulate(
        config=config,
        scenario=scenario,
        extension_report=extension_report,
        out=out,
    )
    raise typer.Exit(code)


@decide_app.command("optimize-budget")
def decide_optimize_budget(
    config: Annotated[Path, typer.Argument(help="Resolved YAML.")],
    extension_report: Annotated[
        Path | None,
        typer.Option(
            "--extension-report",
            help="Training extension_report.json (required for PROD full-panel optimize).",
        ),
    ] = None,
    curve_bundle: Annotated[
        Path | None,
        typer.Option("--curve-bundle", help="Optional curve bundles JSON (legacy diagnostic path)."),
    ] = None,
    allow_unsafe_decision_apis: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-decision-apis",
            help="Non-prod: must match YAML for legacy curve paths.",
        ),
    ] = False,
    legacy_diagnostic_curve_optimizer: Annotated[
        bool,
        typer.Option(
            "--legacy-diagnostic-curve-optimizer",
            help="Non-prod: use clamped curve-bundle optimizer when full-panel inputs unavailable.",
        ),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write optimization + decision bundle JSON (required in PROD)."),
    ] = None,
) -> None:
    """Canonical governance-checked full-panel Δμ budget optimize."""
    from mmm.cli.decision_layer import run_decision_optimize_budget

    code = run_decision_optimize_budget(
        config=config,
        extension_report=extension_report,
        curve_bundle=curve_bundle,
        allow_unsafe_decision_apis=allow_unsafe_decision_apis,
        legacy_diagnostic_curve_optimizer=legacy_diagnostic_curve_optimizer,
        out=out,
    )
    raise typer.Exit(code)


app.add_typer(decide_app, name="decide")


@app.command()
def train(config: Path) -> None:
    """Fit model + CV + extensions; writes ``extension_report.json``.

    Does not run ``decide`` gates or emit a decision bundle.
    """
    trainer = MMMTrainer.from_yaml(config)
    out = trainer.run()
    typer.echo(f"Finished run: {out['run_id']} artifacts at {out['store']}")
    for w in out.get("warnings") or []:
        typer.secho(w, fg=typer.colors.YELLOW, err=True)


@app.command()
def diagnose(
    config: Path,
    extension_report: Annotated[
        Path,
        typer.Option("--extension-report", help="extension_report.json from a training run (diagnostic read)."),
    ],
) -> None:
    """Diagnostic-only workflow: summarize extension artifacts (not a Δμ decision command).

    For canonical planning, use ``mmm decide simulate`` or ``mmm decide optimize-budget`` (PROD gates).
    """
    evaluate(config, extension_report=extension_report)


@app.command()
def evaluate(
    config: Path,
    extension_report: Annotated[
        Path | None,
        typer.Option("--extension-report", help="extension_report.json from a training run"),
    ] = None,
) -> None:
    """Summarize key sections from a saved extension report (baselines, governance, response)."""
    _ = load_config(config)
    if extension_report is None or not extension_report.exists():
        typer.secho(
            "Pass --extension-report PATH to summarize baselines, governance, and response_diagnostics.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    data = json.loads(extension_report.read_text(encoding="utf-8"))
    summary = {
        "baselines": data.get("baselines"),
        "governance": data.get("governance"),
        "response_diagnostics": data.get("response_diagnostics"),
        "curve_safe_for_optimization": data.get("curve_safe_for_optimization"),
        "roi_summary": curve_bundles_to_roi_summary(data.get("curve_bundles") or []),
    }
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def compare(
    panel: Annotated[Path, typer.Argument(help="CSV or Parquet panel (columns per first config data.*)")],
    config_a: Annotated[Path, typer.Argument(help="First YAML config")],
    config_b: Annotated[Path, typer.Argument(help="Second YAML config")],
    out: Annotated[Path | None, typer.Option("--out", help="Write leaderboard JSON")] = None,
) -> None:
    """Run two resolved configs on the same panel and emit a compact leaderboard (Tier 2)."""
    from mmm.api.trainer import MMMComparator
    from mmm.config.load import load_config
    from mmm.config.transform_policy import cross_framework_transform_drift
    from mmm.data.loader import DatasetBuilder

    ca, cb = load_config(config_a), load_config(config_b)
    schema = DatasetBuilder(ca.data).schema()
    suf = panel.suffix.lower()
    if suf in {".parquet", ".pq"}:
        raw = pd.read_parquet(panel)
    elif suf == ".csv":
        raw = pd.read_csv(panel)
    else:
        typer.secho(f"Unsupported panel: {suf}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    df = DatasetBuilder(ca.data, schema).build(raw)
    rows = MMMComparator([ca, cb]).run_all(df)
    board = []
    for i, r in enumerate(rows):
        ext = r.get("extensions") or {}
        gov = ext.get("governance") or {}
        fit = r.get("fit") or {}
        mae = float(fit["best_score"]) if isinstance(fit, dict) and "best_score" in fit else None
        board.append(
            {
                "index": i,
                "run_id": r.get("run_id"),
                "cv_objective_best": mae,
                "approved_for_optimization": gov.get("approved_for_optimization"),
                "identifiability": (ext.get("identifiability") or {}).get("identifiability_score"),
            }
        )
    text = json.dumps(
        {
            "leaderboard": board,
            "transform_comparability": cross_framework_transform_drift(ca, cb),
        },
        indent=2,
        default=str,
    )
    if out:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def calibrate(
    config: Path,
    replay_units: Annotated[
        Path | None,
        typer.Option("--replay-units", help="JSON list of CalibrationUnit-compatible dicts"),
    ] = None,
    etl_shifts: Annotated[
        Path | None,
        typer.Option("--etl-shifts", help="YAML with shifts[] for replay ETL (geo, window, channel, multiplier)"),
    ] = None,
    panel: Annotated[
        Path | None,
        typer.Option("--panel", help="CSV or Parquet panel aligned with config data.* columns"),
    ] = None,
    out_units: Annotated[
        Path | None,
        typer.Option("--out-units", help="Write generated replay units JSON (with --etl-shifts)"),
    ] = None,
) -> None:
    """Run calibration hooks, inspect replay JSON, or build replay units from panel + shift YAML."""
    cfg = load_config(config)
    out: dict[str, Any] = run_calibration_extensions(cfg)
    if etl_shifts is not None and etl_shifts.exists() and panel is not None and panel.exists():
        from mmm.calibration.experiment_ingestion import load_spend_shift_specs_csv
        from mmm.calibration.replay_etl import ingest_validate_and_build_replay_units, load_spend_shift_specs
        from mmm.data.loader import DatasetBuilder

        schema = DatasetBuilder(cfg.data).schema()
        suf = panel.suffix.lower()
        if suf in {".parquet", ".pq"}:
            pan = pd.read_parquet(panel)
        elif suf == ".csv":
            pan = pd.read_csv(panel)
        else:
            typer.secho(f"Unsupported panel format: {suf}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        pan = DatasetBuilder(cfg.data, schema).build(pan)
        if str(etl_shifts).lower().endswith(".csv"):
            shifts = load_spend_shift_specs_csv(etl_shifts)
        else:
            shifts = load_spend_shift_specs(etl_shifts)
        exp_kpi = cfg.calibration.experiment_target_kpi or cfg.data.target_column
        units, val_reports = ingest_validate_and_build_replay_units(
            pan,
            schema,
            shifts,
            target_kpi=cfg.data.target_column,
            expected_target_kpi=exp_kpi,
        )
        dest = out_units or (panel.parent / "replay_units_generated.json")
        write_calibration_units_to_json(units, dest)
        vpath = dest.parent / "replay_etl_validation.json"
        vpath.write_text(json.dumps(val_reports, indent=2, default=str), encoding="utf-8")
        out["replay_etl"] = {
            "n_units": len(units),
            "n_shifts_input": len(shifts),
            "written": str(dest),
            "validation_report": str(vpath),
        }
    if replay_units is not None and replay_units.exists():
        from mmm.calibration.units_io import load_calibration_units_from_json

        units = load_calibration_units_from_json(replay_units)
        out["replay_units_path"] = str(replay_units)
        out["replay_units_count"] = len(units)
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command()
def optimize_budget(
    config: Annotated[
        Path,
        typer.Argument(
            help="Resolved YAML. PROD: unsafe APIs forbidden, gates must be on; use data.path + "
            "--extension-report with ridge_fit_summary for full-panel Δμ. Bayesian blocked in PROD. "
            "See docs/decision_runbook.md §2a."
        ),
    ],
    extension_report: Annotated[
        Path | None,
        typer.Option(
            "--extension-report",
            help=(
                "Training extension_report.json (governance, response_diagnostics, "
                "ridge_fit_summary). Required for gated optimize in all envs."
            ),
        ),
    ] = None,
    curve_bundle: Annotated[
        Path | None,
        typer.Option(
            "--curve-bundle",
            help="JSON with curve_bundles list or a full extension_report containing curve_bundles / curve_bundle.",
        ),
    ] = None,
    allow_unsafe_decision_apis: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-decision-apis",
            help=(
                "Non-prod only: must match YAML allow_unsafe_decision_apis for legacy curve / "
                "placeholder paths. Forbidden in PROD (config load fails)."
            ),
        ),
    ] = False,
    legacy_diagnostic_curve_optimizer: Annotated[
        bool,
        typer.Option(
            "--legacy-diagnostic-curve-optimizer",
            help="Non-prod: use clamped curve-bundle optimizer (diagnostic; not full-panel Δμ).",
        ),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write JSON (optimization + decision_bundle). Required in PROD for full-panel optimize.",
        ),
    ] = None,
) -> None:
    """Maximize Δμ vs baseline (Ridge full-panel) or run legacy diagnostic optimizers (non-prod only).

    Default decision path: SLSQP on ``decision_simulate.simulate`` (point Δμ). Posterior/risk APIs
    use precomputed ``delta_mu_draws_*`` (see runbook §7a), not per-iteration full-Bayes resampling.
    """
    import os

    import numpy as np

    from mmm.config.schema import Framework
    from mmm.diagnostics.curve_optimizer import optimize_budget_from_curve_bundles
    from mmm.governance.decision_safety import MSG_OPTIMIZATION_BLOCKED
    from mmm.optimization.budget.optimizer import BudgetOptimizer
    from mmm.optimization.safety_gate import OptimizationSafetyGate

    cfg = load_config(config)
    if cfg.framework == Framework.BAYESIAN and cfg.run_environment == RunEnvironment.PROD:
        typer.secho(
            "Bayesian optimize-budget is experimental_only and is blocked in run_environment=prod.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    gate = OptimizationSafetyGate(cfg.extensions.optimization_gates)
    gov: dict = {}
    resp: dict | None = None
    ident = 0.0
    er_data: dict | None = None
    ext_present = extension_report is not None and extension_report.exists()
    if ext_present:
        er_data = json.loads(extension_report.read_text(encoding="utf-8"))  # type: ignore[arg-type]
        gov = er_data.get("governance", {})
        resp = er_data.get("response_diagnostics")
        ident = float(er_data.get("identifiability", {}).get("identifiability_score", 0.0))
    pq_er = er_data.get("panel_qa") if isinstance(er_data, dict) else None
    gr = gate.check(
        governance=gov,
        response_diag=resp,
        identifiability_score=ident,
        run_environment=cfg.run_environment,
        extension_report_present=ext_present,
        panel_qa=pq_er if isinstance(pq_er, dict) else None,
    )
    if not gr.allowed:
        typer.echo(json.dumps({"allowed": False, "reasons": gr.reasons, "audit": gr.audit}, indent=2))
        raise typer.Exit(code=2)

    names = list(cfg.data.channel_columns)
    n = len(names)
    total_budget = float(cfg.budget.total_budget or n * 1e5)
    channel_min = np.array([float(cfg.budget.channel_min.get(c, 0.0)) for c in names], dtype=float)
    channel_max = np.array([float(cfg.budget.channel_max.get(c, 1e6)) for c in names], dtype=float)
    current = np.ones(n, dtype=float) * 1e5

    ridge_summary = er_data.get("ridge_fit_summary") if isinstance(er_data, dict) else None
    full_model_ready = bool(cfg.data.path and isinstance(ridge_summary, dict) and ridge_summary.get("coef"))

    if cfg.run_environment == RunEnvironment.PROD and not full_model_ready:
        typer.secho(
            "Production optimize-budget requires config.data.path to the training panel and "
            "extension_report.json containing ridge_fit_summary (full-panel Δμ objective via simulate).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if cfg.run_environment == RunEnvironment.PROD and full_model_ready and out is None:
        typer.secho(
            "Production optimize-budget requires --out PATH to persist the decision bundle and optimization JSON.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if cfg.run_environment == RunEnvironment.PROD and ext_present and isinstance(er_data, dict):
        from mmm.governance.model_release import prod_release_allows_decision_cli

        mr0 = er_data.get("model_release")
        ok_mr, mr_notes = prod_release_allows_decision_cli(
            mr0 if isinstance(mr0, dict) else None,
            surface="optimize_budget",
            run_environment=cfg.run_environment,
        )
        if not ok_mr:
            typer.secho(
                "Production optimize-budget blocked by model_release: " + "; ".join(mr_notes),
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    if full_model_ready:
        from mmm.cli.decision_layer import CANONICAL_OPTIMIZE, emit_shim_deprecation, run_decision_optimize_budget

        emit_shim_deprecation(legacy_subcommand="optimize-budget", canonical=CANONICAL_OPTIMIZE)
        code = run_decision_optimize_budget(
            config=config,
            extension_report=extension_report,
            curve_bundle=curve_bundle,
            allow_unsafe_decision_apis=allow_unsafe_decision_apis,
            legacy_diagnostic_curve_optimizer=legacy_diagnostic_curve_optimizer,
            out=out,
        )
        raise typer.Exit(code)

    if not cfg.allow_unsafe_decision_apis or not allow_unsafe_decision_apis:
        typer.secho(MSG_OPTIMIZATION_BLOCKED, fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)

    if cfg.run_environment == RunEnvironment.PROD:
        typer.secho(
            "Production forbids curve-only optimization; use data.path + extension_report with ridge_fit_summary.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if not legacy_diagnostic_curve_optimizer:
        typer.secho(
            "Full-panel inputs unavailable. Pass --legacy-diagnostic-curve-optimizer (non-prod) to use "
            "deprecated curve interpolation, or provide data.path + ridge_fit_summary.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if os.environ.get("MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET", "").strip() != "1":
        typer.secho(
            "Legacy curve-only budget optimization is quarantined: set MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET=1 "
            "in the environment to acknowledge non-canonical, non-decision-safe behavior.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    gathered = None
    if curve_bundle is not None and curve_bundle.exists():
        gathered = gather_curve_bundles_from_path(curve_bundle)
    elif isinstance(er_data, dict):
        gathered = gather_curve_bundles_from_dict(er_data)

    use_curves = gathered is not None
    if use_curves:
        g_names, bundles = gathered
        bmap = {ch: b for ch, b in zip(g_names, bundles, strict=True)}
        missing = [c for c in names if c not in bmap]
        if missing:
            typer.secho(f"Curve data missing channels: {missing}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        ordered_bundles = [bmap[c] for c in names]
        try:
            ec = economics_contract_for_curve_bundles(ordered_bundles, strict=False)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e
        typer.secho(
            "Using legacy curve interpolation (diagnostic only); prefer extension_report + data.path "
            "for full-panel simulate() optimization.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        res = optimize_budget_from_curve_bundles(
            names,
            ordered_bundles,
            config=cfg,
            current_spend=current,
            total_budget=total_budget,
            channel_min=channel_min,
            channel_max=channel_max,
            economics_contract=ec,
        )
        typer.echo(str(res))
        return

    typer.secho(
        "No curve bundles provided; using legacy marginal-ROI placeholder (non-prod only).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    opt = BudgetOptimizer(
        channel_names=names,
        marginal_roi=np.linspace(1.0, 2.0, n),
        channel_min=channel_min,
        channel_max=channel_max,
    )
    res = opt.optimize(current, total_budget=total_budget)
    typer.echo(str(res))


@app.command()
def simulate(
    config: Annotated[
        Path,
        typer.Argument(
            help="YAML with data.path to the training panel. PROD requires data.path. "
            "Uses mmm.planning.decision_simulate.simulate (canonical Δμ); not curve diagnostics."
        ),
    ],
    scenario: Annotated[
        Path,
        typer.Option(
            "--scenario",
            help=(
                "YAML: candidate_spend / candidate_spend_by_geo / candidate_spend_path; "
                "optional baseline_spend, overlays."
            ),
        ),
    ],
    extension_report: Annotated[
        Path,
        typer.Option(
            "--extension-report",
            help="extension_report.json containing ridge_fit_summary (coef, best_params, intercept).",
        ),
    ],
    out: Annotated[Path | None, typer.Option("--out", help="Write JSON result")] = None,
) -> None:
    """Full-panel Δμ decision simulation (Ridge ``decision_simulate.simulate``).

    This is the **canonical decision** path: same μ construction as training. For curve interpolation
    or SpendPlan scenarios, use ``simulate-diagnostic-curves`` (diagnostic only). See runbook §4 and §10.
    """
    from mmm.cli.decision_layer import CANONICAL_SIMULATE, emit_shim_deprecation, run_decision_simulate

    emit_shim_deprecation(legacy_subcommand="simulate", canonical=CANONICAL_SIMULATE)
    code = run_decision_simulate(
        config=config,
        scenario=scenario,
        extension_report=extension_report,
        out=out,
    )
    raise typer.Exit(code)


@app.command()
def simulate_diagnostic_curves(
    config: Annotated[
        Path,
        typer.Argument(
            help="YAML (channels must match curve data). Diagnostic only — not ``decision_simulate``; "
            "PROD curve bundles require economics_contract on bundles (fail-closed)."
        ),
    ],
    scenario: Annotated[
        Path,
        typer.Option(
            "--scenario",
            help="SpendPlan YAML (horizon_weeks, aggregate_steps, …) OR SpendScenario (baseline/proposed_spend).",
        ),
    ],
    curve_bundle: Annotated[
        Path | None,
        typer.Option("--curve-bundle", help="JSON with curve_bundles or extension-style file"),
    ] = None,
    extension_report: Annotated[
        Path | None,
        typer.Option("--extension-report", help="Use curve_bundles from a training extension_report.json"),
    ] = None,
    uncertainty: Annotated[
        str,
        typer.Option("--uncertainty", help="point | bootstrap | posterior (SpendPlan YAML only)"),
    ] = "point",
    bootstrap_bundles: Annotated[
        Path | None,
        typer.Option(
            "--bootstrap-bundles",
            help="JSON list of curve_bundles draws: [[{bundle...},...], ...] for uncertainty=bootstrap",
        ),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Write scenario result JSON")] = None,
) -> None:
    """Curve / response-surface diagnostics (``simulate_curve_diagnostic``) — **not** full-panel Δμ.

    Do not use outputs for production budget decisions. For canonical Δμ, use ``mmm decide simulate`` with
    ``data.path`` + ``ridge_fit_summary``. See runbook §10.
    """
    cfg = load_config(config)
    _ = cfg
    from mmm.simulation.engine import (
        SpendPlan,
        SpendScenario,
        run_curve_bundle_scenario,
        run_stepped_scenario,
        simulate_curve_diagnostic as simulate_engine,
    )

    _typed_curves = cfg.run_environment == RunEnvironment.PROD
    gathered = None
    if curve_bundle is not None and curve_bundle.exists():
        gathered = gather_curve_bundles_from_path(
            curve_bundle, require_typed_curve_quantity=_typed_curves
        )
    elif extension_report is not None and extension_report.exists():
        gathered = gather_curve_bundles_from_dict(
            json.loads(extension_report.read_text(encoding="utf-8")),
            require_typed_curve_quantity=_typed_curves,
        )
    if gathered is None:
        typer.secho("Provide --curve-bundle or --extension-report with curve data.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _, bundles = gathered
    if cfg.run_environment == RunEnvironment.PROD:
        try:
            economics_contract_for_curve_bundles(bundles, strict=True)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e

    raw = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    result: dict[str, Any] | None = None
    if isinstance(raw, dict) and (
        "aggregate_steps" in raw
        or ("horizon_weeks" in raw and "baseline_spend" not in raw)
    ):
        try:
            plan = SpendPlan.from_yaml(scenario)
            unc = str(raw.get("uncertainty", uncertainty))
            bsets = None
            if unc == "bootstrap":
                if bootstrap_bundles is None or not bootstrap_bundles.exists():
                    typer.secho("--bootstrap-bundles required for uncertainty=bootstrap", fg=typer.colors.RED, err=True)
                    raise typer.Exit(code=1)
                bsets = json.loads(bootstrap_bundles.read_text(encoding="utf-8"))
            result = simulate_engine(plan, bundles, uncertainty_mode=unc, bootstrap_bundle_sets=bsets)
        except ValueError:
            result = None
    if result is None:
        scen = SpendScenario.from_yaml(scenario)
        if len(scen.steps) >= 2:
            result = run_stepped_scenario(bundles, steps=scen.steps, y_level_scale=scen.y_level_scale)
        else:
            if not scen.baseline_spend:
                typer.secho(
                    "Use SpendPlan YAML (aggregate_steps) or SpendScenario (baseline_spend / proposed_spend).",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
            try:
                result = run_curve_bundle_scenario(bundles, scen)
            except ValueError as e:
                typer.secho(str(e), fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1) from e
    text = json.dumps(result, indent=2, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def report(
    config: Path,
    extension_report: Annotated[
        Path | None,
        typer.Option("--extension-report", help="Optional extension_report.json for ROI section"),
    ] = None,
) -> None:
    cfg = load_config(config)
    rb = ReportBuilder()
    rb.add("config", cfg.model_dump())
    from mmm.governance.decision_safety import report_decision_safety_section

    rb.add(
        "decision_safety_reporting",
        report_decision_safety_section(allow_unsafe_decision_apis=cfg.allow_unsafe_decision_apis),
    )
    if extension_report is not None and extension_report.exists():
        er = json.loads(extension_report.read_text(encoding="utf-8"))
        cb = er.get("curve_bundles") or []
        if cb:
            rb.add("roi_summary", curve_bundles_to_roi_summary(cb))
    out = Path(cfg.artifacts.run_dir) / "report.json"
    rb.write(out)
    typer.echo(f"wrote {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
