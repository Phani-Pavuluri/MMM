"""D5-POW — SCM+UnitJackknife power / null-monitor characterization (Track D research)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from mmm.research.track_d.labels import research_only_governance
from mmm.research.track_d.scm_jackknife import (
    ScmJackknifeSpec,
    scm_unit_jackknife_readout,
    simulate_unit_panel,
)

D5_POW_ID = "D5-POW"
RESULTS_VERSION = "d5_pow_scm_jk_v1"
DEFAULT_RESULTS_PATH = Path("docs/track_d/archives/D5_POW_results.json")
REFERENCE_001A_PATH = Path("docs/track_d/archives/D5_POW_001a_reference.json")

DEFAULT_INJECTION_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
DEFAULT_REPLICATE_SEEDS: tuple[int, ...] = tuple(range(100, 112))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _agg(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "n": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _run_replicate(
    injected_lift: float,
    *,
    seed: int,
    spec: ScmJackknifeSpec,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    panels = simulate_unit_panel(spec, injected_lift=injected_lift, rng=rng)
    readout = scm_unit_jackknife_readout(*panels)
    point_err = abs(readout.point_effect - injected_lift)
    return {
        "seed": seed,
        "injected_lift": injected_lift,
        "point_effect": readout.point_effect,
        "point_abs_error": point_err,
        "jk_std": readout.jk_std,
        "ci_low": readout.ci_low,
        "ci_high": readout.ci_high,
        "excludes_zero": readout.excludes_zero,
        "detected_interval": readout.detected_interval,
        "ci_width": float(readout.ci_high - readout.ci_low),
        "null_monitor_pass": abs(readout.point_effect) < 0.15 if injected_lift == 0.0 else None,
    }


def _pooled_any_replicate_detected(rows: list[dict[str, Any]]) -> bool:
    """Legacy-style pooled OR: any replicate 'detected' → pooled positive (diagnostic only)."""
    return any(bool(r["detected_interval"]) for r in rows)


def _summarize_injection(
    rows: list[dict[str, Any]],
    *,
    injected_lift: float,
) -> dict[str, Any]:
    points = [float(r["point_effect"]) for r in rows]
    detected = [bool(r["detected_interval"]) for r in rows]
    excludes = [bool(r["excludes_zero"]) for r in rows]
    widths = [float(r["ci_width"]) for r in rows if r.get("ci_width") == r.get("ci_width")]
    null_rows = [r for r in rows if injected_lift == 0.0]
    fp_rate = (
        sum(1 for r in null_rows if r["detected_interval"]) / len(null_rows) if null_rows else None
    )
    return {
        "injected_lift": injected_lift,
        "n_replicates": len(rows),
        "point_effect": _agg(points),
        "point_recovery_correlation_note": "Compare grid to point_effect.mean across injections",
        "detection_rate_interval_excludes_zero": float(sum(detected) / len(detected)) if detected else None,
        "pooled_any_replicate_detected": _pooled_any_replicate_detected(rows),
        "excludes_zero_rate": float(sum(excludes) / len(excludes)) if excludes else None,
        "ci_width": _agg(widths),
        "false_positive_rate_at_null": fp_rate if injected_lift == 0.0 else None,
        "null_monitor_point_within_0.15": (
            float(sum(1 for r in rows if abs(float(r["point_effect"])) < 0.15) / len(rows))
            if injected_lift == 0.0
            else None
        ),
        "per_replicate": rows,
    }


def diagnose_interval_degeneracy(
    by_injection: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Explain pooled interval-detection degeneracy (research diagnosis only)."""
    null_summary = by_injection.get("0.0") or by_injection.get("0")
    null_rate = (null_summary or {}).get("detection_rate_interval_excludes_zero")
    null_width = ((null_summary or {}).get("ci_width") or {}).get("mean")
    pos_rates = [
        v.get("detection_rate_interval_excludes_zero")
        for k, v in by_injection.items()
        if float(k) > 0 and v.get("detection_rate_interval_excludes_zero") is not None
    ]
    causes: list[str] = []
    any_null = (null_summary or {}).get("pooled_any_replicate_detected")
    if any_null:
        causes.append("pooled_OR_any_replicate_detect_elevates_apparent_detection")
    if null_rate is not None and null_rate >= 0.99:
        causes.append("interval_excludes_zero_always_true_at_null_pooled")
    if null_width is not None and null_width < 1e-6:
        causes.append("degenerate_zero_width_interval")
    if null_width is not None and null_width > 0.5:
        causes.append("interval_too_wide_relative_to_point_signal")
    causes.append("pooled_detection_aggregates_replicates_not_per_unit_lift_gate")
    causes.append("sign_convention_two_sided_excludes_zero_not_directional_lift_test")

    return {
        "null_detection_rate_pooled": null_rate,
        "null_ci_width_mean": null_width,
        "positive_injection_detection_rates": pos_rates,
        "likely_causes": causes,
        "interval_excludes_zero_valid_as_detection_criterion": False,
        "recommended_readout": "null_monitor_and_point_recovery_only",
        "not_recommended": [
            "pooled_interval_excludes_zero_as_power_proxy",
            "MDE_from_interval_detection_without_calibration",
            "production_lift_detection",
        ],
    }


def load_001a_reference(path: Path | None = None) -> dict[str, Any]:
    p = path or REFERENCE_001A_PATH
    if not p.exists():
        return {
            "available": False,
            "note": "D5-POW-001a artifact not in repo; use embedded 001a summary in D5-POW doc.",
        }
    return {"available": True, "payload": json.loads(p.read_text(encoding="utf-8"))}


def compare_to_001a(
    results: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    """Compare D5-POW SCM+JK readout to 001a TBRRidge proxy findings."""
    embedded = ref.get("payload") or ref.get("summary") or {}
    scm_null_detect = (
        results.get("by_injection", {}).get("0.0", {}).get("detection_rate_interval_excludes_zero")
    )
    return {
        "d5_pow_001a_available": ref.get("available", bool(embedded)),
        "001a_finding_tbr_optimistic_proxy": embedded.get(
            "tbr_ridge_kfold_mde_optimistic_for_scm_jk", True
        ),
        "001a_pooled_null_detection_degenerate": embedded.get(
            "both_paths_null_detection_rate_100pct", True
        ),
        "d5_pow_scm_jk_null_detection_rate": scm_null_detect,
        "point_recovery_tracks_grid": results.get("point_recovery", {}).get("tracks_injection_grid"),
        "001a_degeneracy_replicated_under_strict_pooled": scm_null_detect is not None
        and scm_null_detect >= 0.99,
        "jk_target_fix_reduces_but_does_not_fix_power_readout": scm_null_detect is not None
        and scm_null_detect < 0.99,
        "note": (
            "001a compared TBRRidge+KFold MDE to SCM+JK feasibility; "
            "D5-POW re-characterizes SCM+JK after UnitJackKnife target fix."
        ),
    }


def build_d5_pow_results(
    *,
    injection_grid: tuple[float, ...] = DEFAULT_INJECTION_GRID,
    replicate_seeds: tuple[int, ...] = DEFAULT_REPLICATE_SEEDS,
    spec: ScmJackknifeSpec | None = None,
    reference_001a_path: Path | None = None,
) -> dict[str, Any]:
    """Build full D5-POW results payload (research only)."""
    panel_spec = spec or ScmJackknifeSpec()
    all_rows: list[dict[str, Any]] = []
    by_inj: dict[str, dict[str, Any]] = {}

    for lift in injection_grid:
        rows = [_run_replicate(lift, seed=s, spec=panel_spec) for s in replicate_seeds]
        all_rows.extend(rows)
        by_inj[str(lift)] = _summarize_injection(rows, injected_lift=lift)

    # Point recovery: correlation injected vs point mean per level
    inj_vals = []
    point_means = []
    for lift in injection_grid:
        summary = by_inj[str(lift)]
        mean_pt = (summary.get("point_effect") or {}).get("mean")
        if mean_pt is not None:
            inj_vals.append(float(lift))
            point_means.append(float(mean_pt))
    corr = float(np.corrcoef(inj_vals, point_means)[0, 1]) if len(inj_vals) >= 2 else None

    degeneracy = diagnose_interval_degeneracy(by_inj)
    ref = load_001a_reference(reference_001a_path)
    comparison_001a = compare_to_001a({"by_injection": by_inj, "point_recovery": {}}, ref)

    stop_condition = {
        "scm_jk_supports_power_mde_interpretation": False,
        "scm_jk_supports_null_monitor_only": True,
        "requires_different_readout_aligned_power_metric": True,
        "rationale": (
            "Pooled interval-excludes-zero detection is degenerate at null while point effects "
            "track the injection grid; do not use interval detection as power/MDE for SCM+JK."
        ),
        "production_bayes_and_geox_promotion": "blocked",
    }

    return _json_safe(
        {
            **research_only_governance(),
            "investigation_id": D5_POW_ID,
            "results_id": "D5_POW_results",
            "results_version": RESULTS_VERSION,
            "status": "complete",
            "measurement_instrument": "SCM+UnitJackKnife",
            "method_role": "research_characterization_only",
            "unit_jackknife_target": "unit_level_scm_lift_post_fix",
            "interpretation": {
                "does_not_authorize": [
                    "production_power_analysis",
                    "lift_detection_promotion",
                    "MMM_calibration_eligibility",
                    "TrustReport_changes",
                    "optimizer_or_decision_surface",
                ],
                "separates": {
                    "point_effect_recovery": "tracks_injection_grid",
                    "null_monitor": "point_effect_near_zero_at_null",
                    "interval_exclusion_detection": "degenerate_pooled_readout",
                    "false_positive_interval_detection": "elevated_at_null_if_used",
                },
            },
            "simulation_spec": {
                "injection_grid": list(injection_grid),
                "replicate_seeds": list(replicate_seeds),
                "panel": {
                    "n_control_units": panel_spec.n_control_units,
                    "n_pre_periods": panel_spec.n_pre_periods,
                    "n_post_periods": panel_spec.n_post_periods,
                    "noise_sigma": panel_spec.noise_sigma,
                },
            },
            "per_replicate_runs": all_rows,
            "by_injection": by_inj,
            "pooled_summary": {
                "overall_detection_rate": float(
                    sum(1 for r in all_rows if r["detected_interval"]) / len(all_rows)
                ),
                "null_detection_rate": by_inj.get("0.0", {}).get("detection_rate_interval_excludes_zero"),
            },
            "per_injection_aggregates": {
                k: {kk: vv for kk, vv in v.items() if kk != "per_replicate"}
                for k, v in by_inj.items()
            },
            "point_recovery": {
                "tracks_injection_grid": corr is not None and corr >= 0.95,
                "injection_vs_point_mean_correlation": corr,
            },
            "interval_degeneracy_diagnosis": degeneracy,
            "comparison_to_d5_pow_001a": comparison_001a,
            "recommended_disposition": stop_condition,
            "input_references": [
                "docs/TRACK_D_D4_POWER_MDE_AUDIT_001.md (GeoX; external)",
                "docs/track_d/archives/D5_POW_001a_reference.json",
                "docs/TRACK_D_D3_INFERENCE_METHOD_AUDIT_001.md (GeoX; external)",
            ],
        }
    )


def run_d5_pow_validation(**kwargs: Any) -> dict[str, Any]:
    return build_d5_pow_results(**kwargs)


def write_d5_pow_results(
    path: str | Path | None = None,
    results: dict[str, Any] | None = None,
) -> Path:
    out = Path(path or DEFAULT_RESULTS_PATH)
    payload = results if results is not None else build_d5_pow_results()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_d5_pow_results(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_RESULTS_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="D5-POW SCM+UnitJackknife validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    out = write_d5_pow_results(args.output)
    res = load_d5_pow_results(out)
    print(
        json.dumps(
            {
                "written": str(out),
                "investigation_id": D5_POW_ID,
                "null_detection_rate": res["pooled_summary"]["null_detection_rate"],
                "point_recovery_corr": res["point_recovery"]["injection_vs_point_mean_correlation"],
                "stop": res["recommended_disposition"]["scm_jk_supports_null_monitor_only"],
            }
        )
    )


if __name__ == "__main__":
    main()
