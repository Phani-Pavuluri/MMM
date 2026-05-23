"""Large-scale performance certification (measure only; no algorithm changes)."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.extensions import PerformanceCertificationConfig
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

REPORT_VERSION = "mmm_performance_certification_v1"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Performance certification documents operating limits — it does not change training or decision algorithms.",
    "Memory estimates use tracemalloc peaks on synthetic panels; production workloads may differ.",
)


@dataclass(frozen=True)
class ScalingScenario:
    name: str
    n_geos: int
    n_channels: int
    n_weeks: int


SCENARIOS: tuple[ScalingScenario, ...] = (
    ScalingScenario("small", n_geos=20, n_channels=15, n_weeks=52),
    ScalingScenario("medium", n_geos=100, n_channels=40, n_weeks=104),
    ScalingScenario("large", n_geos=500, n_channels=100, n_weeks=156),
)


def _synthetic_panel(spec: ScalingScenario, seed: int) -> tuple[pd.DataFrame, PanelSchema]:
    channels = tuple(f"ch{i}" for i in range(spec.n_channels))
    betas = tuple([0.3 / max(spec.n_channels, 1)] * spec.n_channels)
    geo_spec = SyntheticGeoPanelSpec(
        n_geos=spec.n_geos,
        n_weeks=spec.n_weeks,
        channels=channels,
        betas=betas,
        noise=0.02,
    )
    return generate_geo_panel(geo_spec, seed=seed)


def _timed_stage(label: str, fn: Any) -> dict[str, Any]:
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn()
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001 — certification captures failures
        ok = False
        err = str(exc)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "stage": label,
        "elapsed_ms": round(elapsed_ms, 3),
        "peak_memory_kb": round(peak / 1024.0, 2),
        "success": ok,
        "error": err,
    }


def _run_scenario(spec: ScalingScenario, *, seed: int, n_trials: int) -> dict[str, Any]:
    panel, schema = _synthetic_panel(spec, seed)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=max(8, spec.n_weeks // 10), horizon_weeks=4),
        ridge_bo={"n_trials": n_trials},
        calibration={"use_replay_calibration": False},
        random_seed=seed,
    )
    stages: list[dict[str, Any]] = []

    def _train() -> None:
        RidgeBOMMMTrainer(cfg, schema).fit(panel)

    stages.append(_timed_stage("train", _train))

    art = RidgeBOMMMTrainer(cfg, schema).fit(panel) if stages[-1]["success"] else {}
    bp = {}
    if art.get("artifacts") is not None:
        bp = getattr(art["artifacts"], "best_params", {}) or {}

    def _design() -> None:
        build_design_matrix(
            panel,
            schema,
            cfg,
            decay=float(bp.get("decay", 0.5)),
            hill_half=float(bp.get("hill_half", 1.0)),
            hill_slope=float(bp.get("hill_slope", 2.0)),
        )

    stages.append(_timed_stage("design_matrix", _design))

    def _extensions_stub() -> None:
        from mmm.evaluation.extension_runner import run_post_fit_extensions

        if art:
            bundle = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
            yhat = np.exp(bundle.y_modeling[: len(panel)])
            run_post_fit_extensions(
                panel=panel,
                schema=schema,
                config=cfg,
                fit_out=art,
                yhat=yhat,
                store=None,
            )

    stages.append(_timed_stage("extension_overhead", _extensions_stub))

    total_ms = sum(s["elapsed_ms"] for s in stages if s["success"])
    peak_kb = max((s["peak_memory_kb"] for s in stages), default=0.0)
    n_rows = len(panel)
    return {
        "scenario": spec.name,
        "n_geos": spec.n_geos,
        "n_channels": spec.n_channels,
        "n_weeks": spec.n_weeks,
        "n_rows": n_rows,
        "runtime_by_stage": stages,
        "total_runtime_ms": round(total_ms, 3),
        "memory_estimate_kb": round(peak_kb, 2),
        "all_stages_success": all(s["success"] for s in stages),
    }


def build_performance_certification_report(
    config: MMMConfig,
    *,
    cfg_ext: PerformanceCertificationConfig | None = None,
) -> dict[str, Any]:
    """Run synthetic scaling scenarios and emit operating-limit guidance."""
    ext = cfg_ext or config.extensions.performance_certification
    if not ext.enabled:
        return {
            "report_version": REPORT_VERSION,
            "enabled": False,
            "diagnostic_only": True,
            "notes": ["performance_certification disabled"],
        }
    scenarios_out: list[dict[str, Any]] = []
    for i, spec in enumerate(SCENARIOS):
        if spec.name == "medium" and not ext.include_medium_scenario:
            continue
        if spec.name == "large" and not ext.include_large_scenario:
            continue
        scenarios_out.append(_run_scenario(spec, seed=ext.seed + i, n_trials=ext.n_trials_per_scenario))

    bottlenecks: list[str] = []
    for row in scenarios_out:
        for st in row.get("runtime_by_stage", []):
            if st.get("elapsed_ms", 0) > 30_000:
                bottlenecks.append(f"{row['scenario']}:{st['stage']} > 30s")
            if st.get("peak_memory_kb", 0) > 500_000:
                bottlenecks.append(f"{row['scenario']}:{st['stage']} memory > 500MB")

    recommendations: list[str] = []
    if bottlenecks:
        recommendations.append("Review panel width (geos × weeks × channels) before prod batch training.")
    else:
        recommendations.append("No scenario exceeded 30s per stage at certified synthetic scale.")
    recommendations.extend(list(GOVERNANCE_WARNINGS))

    return {
        "report_version": REPORT_VERSION,
        "enabled": True,
        "diagnostic_only": True,
        "prod_decisioning_allowed": False,
        "scenarios": scenarios_out,
        "runtime_by_stage": {r["scenario"]: r["runtime_by_stage"] for r in scenarios_out},
        "scaling_summary": {
            "smallest_rows": min((r["n_rows"] for r in scenarios_out), default=0),
            "largest_rows": max((r["n_rows"] for r in scenarios_out), default=0),
        },
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "certification_warnings": list(GOVERNANCE_WARNINGS),
    }
