"""Runtime profiling for extension and artifact paths (evidence-based optimization)."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.evaluation.drift_monitor import build_drift_report
from mmm.features.design_matrix import build_design_matrix


def _timed(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": label,
        "elapsed_ms": round(elapsed_ms, 3),
        "peak_memory_kb": round(peak / 1024.0, 2),
        "result_summary": getattr(result, "n_rows", type(result).__name__),
    }


def build_performance_report(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Measure hot paths without changing training behavior.

    Emits optimization hints based on observed timings (diagnostic only).
    """
    timings: list[dict[str, Any]] = []
    art = (fit_out or {}).get("artifacts")
    if art is not None:
        bp = art.best_params

        def _design() -> Any:
            return build_design_matrix(
                panel,
                schema,
                config,
                decay=bp["decay"],
                hill_half=bp["hill_half"],
                hill_slope=bp["hill_slope"],
            )

        timings.append(_timed("design_matrix_build", _design))

    def _drift() -> Any:
        return build_drift_report(panel=panel, schema=schema, config=config)

    timings.append(_timed("drift_report_build", _drift))

    hints: list[str] = []
    for row in timings:
        if row["elapsed_ms"] > 500:
            hints.append(f"{row['label']} exceeded 500ms — consider caching or vectorization.")
        if row["peak_memory_kb"] > 50_000:
            hints.append(f"{row['label']} peak memory >50MB — review panel size and artifact payload.")

    if not hints:
        hints.append("No dominant bottleneck flagged at current panel scale.")

    return {
        "diagnostic_only": True,
        "timings": timings,
        "optimization_hints": hints,
        "notes": [
            "Bootstrap identifiability and full extension suite are not timed here.",
            "Use this artifact to justify performance work with measured evidence.",
        ],
    }
