"""Command-line interface for MMM runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
import yaml

from mmm.api.trainer import MMMTrainer
from mmm.calibration.units_io import write_calibration_units_to_json
from mmm.config.load import load_config
from mmm.config.schema import RunEnvironment
from mmm.data.loader import DatasetBuilder
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import validate_panel
from mmm.economics.canonical import economics_contract_for_curve_bundles
from mmm.governance.decision_safety import MSG_OPTIMIZATION_BLOCKED
from mmm.optimization.budget.curve_bundles_io import gather_curve_bundles_from_dict, gather_curve_bundles_from_path
from mmm.optimization.budget.curve_optimizer import optimize_budget_from_curve_bundles
from mmm.optimization.budget.optimizer import BudgetOptimizer
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.optimization.safety_gate import OptimizationSafetyGate
from mmm.planning.context import ridge_context_from_summary
from mmm.reporting.builder import ReportBuilder
from mmm.reporting.roi_sections import curve_bundles_to_roi_summary
from mmm.services.calibration_service import run_calibration_extensions

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def train(config: Path) -> None:
    """Train model from YAML."""
    trainer = MMMTrainer.from_yaml(config)
    out = trainer.run()
    typer.echo(f"Finished run: {out['run_id']} artifacts at {out['store']}")
    for w in out.get("warnings") or []:
        typer.secho(w, fg=typer.colors.YELLOW, err=True)


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
    config: Annotated[Path, typer.Argument(help="YAML config")],
    extension_report: Annotated[
        Path | None,
        typer.Option("--extension-report", help="extension_report.json from a training run"),
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
            help="Required with YAML allow_unsafe_decision_apis to run experimental optimizer (Phase 0 freeze).",
        ),
    ] = False,
    legacy_diagnostic_curve_optimizer: Annotated[
        bool,
        typer.Option(
            "--legacy-diagnostic-curve-optimizer",
            help="Non-prod only: allow deprecated curve-interpolation optimizer (not decision-safe).",
        ),
    ] = False,
) -> None:
    cfg = load_config(config)
    if not cfg.allow_unsafe_decision_apis or not allow_unsafe_decision_apis:
        typer.secho(MSG_OPTIMIZATION_BLOCKED, fg=typer.colors.YELLOW, err=True)
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
    gr = gate.check(
        governance=gov,
        response_diag=resp,
        identifiability_score=ident,
        run_environment=cfg.run_environment,
        extension_report_present=ext_present,
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

    if full_model_ready:
        builder = DatasetBuilder(cfg.data)
        schema = builder.schema()
        panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
        try:
            ctx = ridge_context_from_summary(panel, schema, cfg, ridge_summary)  # type: ignore[arg-type]
        except (ValueError, KeyError) as e:
            typer.secho(f"Invalid ridge_fit_summary: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from e
        res = optimize_budget_via_simulation(
            ctx,
            current_spend=current,
            total_budget=total_budget,
            channel_min=channel_min,
            channel_max=channel_max,
        )
        from mmm.artifacts.decision_bundle import build_decision_bundle
        from mmm.data.fingerprint import fingerprint_panel

        bundle = build_decision_bundle(
            config=cfg,
            schema=schema,
            governance=gov,
            optimization_gate=gr.to_json(),
            simulation_contract={"source": "full_model_simulation_slsqp", "objective": "delta_mu"},
            data_fingerprint=fingerprint_panel(panel, schema),
            uncertainty_mode="point",
            decision_safe=bool(res.get("success")),
            model_summary={
                "objective_delta_mu": res.get("objective_delta_mu"),
                "optimizer_success": res.get("success"),
            },
            economics_surface="full_model_simulation",
        )
        typer.echo(json.dumps({"optimization": res, "decision_bundle": bundle}, indent=2, default=str))
        return

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
    config: Annotated[Path, typer.Argument(help="Resolved YAML config (must include data.path)")],
    scenario: Annotated[
        Path,
        typer.Option(
            "--scenario",
            help="YAML with candidate_spend: {channel: level} and optional baseline_spend (else BAU).",
        ),
    ],
    extension_report: Annotated[
        Path,
        typer.Option("--extension-report", help="extension_report.json with ridge_fit_summary"),
    ],
    out: Annotated[Path | None, typer.Option("--out", help="Write JSON result")] = None,
) -> None:
    """Canonical **full-panel** Δμ simulation (decision-grade). Not curve interpolation."""
    cfg = load_config(config)
    if cfg.run_environment == RunEnvironment.PROD and not cfg.data.path:
        typer.secho("Production simulate requires data.path on config.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if not cfg.data.path:
        typer.secho("simulate requires config.data.path to the training panel.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if not extension_report.exists():
        typer.secho("--extension-report must exist and include ridge_fit_summary.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    er = json.loads(extension_report.read_text(encoding="utf-8"))
    rs = er.get("ridge_fit_summary")
    if not isinstance(rs, dict) or not rs.get("coef"):
        typer.secho("extension_report must contain ridge_fit_summary.coef", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    raw = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        typer.secho("scenario YAML must be a mapping.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    from mmm.planning.baseline import (
        channel_means_from_geo_plan,
        locked_geo_plan_baseline,
        locked_plan_baseline,
    )
    from mmm.planning.control_overlay import ControlOverlaySpec
    from mmm.planning.decision_simulate import simulate as decision_simulate
    from mmm.planning.spend_path import (
        PiecewiseSpendPath,
        counterfactual_piecewise_spend_panel,
        time_mean_spend_by_channel,
    )

    builder = DatasetBuilder(cfg.data)
    schema = builder.schema()
    panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    ctx = ridge_context_from_summary(panel, schema, cfg, rs)
    geos_cli = sorted({str(x) for x in panel[schema.geo_column].unique()})
    spend_path = None
    if isinstance(raw.get("candidate_spend_path"), dict):
        spend_path = PiecewiseSpendPath.from_dict(raw["candidate_spend_path"])
    has_geo_cand = isinstance(raw.get("candidate_spend_by_geo"), dict)
    has_scalar_cand = isinstance(raw.get("candidate_spend"), dict)
    if spend_path is not None and has_geo_cand:
        typer.secho(
            "scenario YAML: candidate_spend_path cannot be combined with candidate_spend_by_geo.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if has_geo_cand and has_scalar_cand:
        typer.secho(
            "scenario YAML: use either candidate_spend or candidate_spend_by_geo, not both.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if spend_path is None and not has_scalar_cand and not has_geo_cand:
        typer.secho(
            "scenario YAML must include candidate_spend, candidate_spend_by_geo, and/or candidate_spend_path.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    spend_plan_geo = None
    if spend_path is not None:
        if has_scalar_cand and isinstance(raw.get("candidate_spend"), dict):
            cand = {str(k): float(v) for k, v in raw["candidate_spend"].items()}
        else:
            tmp_df = counterfactual_piecewise_spend_panel(panel, schema, spend_path)
            cand = time_mean_spend_by_channel(tmp_df, schema)
    elif has_geo_cand:
        raw_geo = raw["candidate_spend_by_geo"]
        spend_plan_geo = {
            str(g): {str(c): float(v) for c, v in row.items()} for g, row in raw_geo.items() if isinstance(row, dict)
        }
        cand = channel_means_from_geo_plan(spend_plan_geo, schema, geos_cli)
    else:
        cs = raw.get("candidate_spend")
        if not isinstance(cs, dict):
            typer.secho("scenario YAML must include candidate_spend when candidate_spend_path is absent.", err=True)
            raise typer.Exit(code=1)
        cand = {str(k): float(v) for k, v in cs.items()}
    base_plan = None
    if isinstance(raw.get("baseline_spend"), dict):
        base_plan = locked_plan_baseline(
            {str(k): float(v) for k, v in raw["baseline_spend"].items()},
            source="scenario_yaml:baseline_spend",
            notes="baseline_spend from scenario YAML (non-BAU; labeled locked_plan).",
        )
    elif isinstance(raw.get("baseline_spend_by_geo"), dict):
        raw_bg = raw["baseline_spend_by_geo"]
        by_geo = {
            str(g): {str(c): float(v) for c, v in row.items()} for g, row in raw_bg.items() if isinstance(row, dict)
        }
        base_plan = locked_geo_plan_baseline(
            by_geo,
            source="scenario_yaml:baseline_spend_by_geo",
            notes="baseline_spend_by_geo from scenario YAML (non-BAU; labeled locked_plan).",
        )
    co_b = ControlOverlaySpec.from_dict(raw["control_overlay_baseline"]) if isinstance(
        raw.get("control_overlay_baseline"), dict
    ) else None
    co_p = ControlOverlaySpec.from_dict(raw["control_overlay_plan"]) if isinstance(
        raw.get("control_overlay_plan"), dict
    ) else None
    co_single = (
        ControlOverlaySpec.from_dict(raw["control_overlay"])
        if isinstance(raw.get("control_overlay"), dict)
        else None
    )
    ctrls_yaml = raw.get("controls_plan")
    sim = decision_simulate(
        cand,
        ctx,
        baseline_plan=base_plan,
        uncertainty_mode="point",
        spend_path_plan=spend_path,
        spend_plan_geo=spend_plan_geo,
        controls_plan=ctrls_yaml if co_p is None and co_single is None else None,
        control_overlay_baseline=co_b,
        control_overlay_plan=co_p,
        control_overlay=co_single if co_p is None else None,
    )
    text = json.dumps(sim.to_json(), indent=2, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def simulate_diagnostic_curves(
    config: Annotated[Path, typer.Argument(help="YAML config (channels must match curve data)")],
    scenario: Annotated[
        Path,
        typer.Option(
            "--scenario",
            help="YAML: SpendPlan (horizon_weeks, aggregate_steps) OR SpendScenario (baseline/proposed)",
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
    """Diagnostic only: curve / response-surface interpolation — not decision-grade full-panel Δμ."""
    cfg = load_config(config)
    _ = cfg
    from mmm.simulation.engine import (
        SpendPlan,
        SpendScenario,
        run_curve_bundle_scenario,
        run_stepped_scenario,
    )
    from mmm.simulation.engine import (
        simulate as simulate_engine,
    )

    gathered = None
    if curve_bundle is not None and curve_bundle.exists():
        gathered = gather_curve_bundles_from_path(curve_bundle)
    elif extension_report is not None and extension_report.exists():
        gathered = gather_curve_bundles_from_dict(json.loads(extension_report.read_text(encoding="utf-8")))
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
