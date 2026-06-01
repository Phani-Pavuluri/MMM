"""Bayes-H4b repeated recovery pilot — longer sampling, multiple seeds (research only)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import (
    _backend_metadata,
    _json_safe,
    _world_row_from_report,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    SAMPLER_EXTENDED,
    WORLD_BAYES_H4_SPARSE_GEO,
    get_recovery_world,
)

PILOT_ID = "BAYES_H4_REPEATED_PILOT_20260601"
PILOT_ID_PRIMARY = "BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601"
PILOT_VERSION = "bayes_h4_repeated_pilot_v1"
PILOT_VERSION_PRIMARY = "bayes_h4_repeated_pilot_v2_primary_metric"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H4_REPEATED_PILOT_20260601.json")
PRIMARY_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json")
DEFAULT_PANEL_SEED = 4400
DEFAULT_NUTS_SEEDS: tuple[int, ...] = (4400, 4401, 4402)

METRIC_DEFINITIONS: dict[str, Any] = {
    "shrinkage_ratio_sparse": {
        "role": "pooling_mechanics_primary",
        "pool_center": "posterior_mu_c",
        "interpret_lt_1": "posterior beta closer to learned mu_hat than outlier was",
        "not_a_gate": True,
    },
    "shrinkage_ratio_sparse_vs_true_mu": {
        "role": "true_effect_recovery_diagnostic",
        "pool_center": "true_mu_c",
        "interpret_lt_1": "posterior beta closer to generative mu_star (recovery check only)",
        "not_a_pooling_gate": True,
    },
    "beta_gc_mae": {"role": "true_effect_recovery"},
    "mu_c_mae": {"role": "true_effect_recovery"},
    "beta_gc_coverage_90": {"role": "true_effect_recovery_directional"},
}


def _agg_numeric(values: list[float]) -> dict[str, Any]:
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


def _run_row(
    world_id: str,
    *,
    nuts_seed: int,
    panel_seed: int,
    sampler: dict[str, Any],
) -> dict[str, Any]:
    spec = get_recovery_world(world_id)
    report = run_h4_recovery_world(
        world_id,
        fast_mcmc=False,
        sampler=sampler,
        nuts_seed=nuts_seed,
        panel_seed=panel_seed,
    )
    row = _world_row_from_report(report, spec)
    row["nuts_seed"] = nuts_seed
    row["panel_seed"] = panel_seed
    row["sampler"] = dict(sampler)
    return row


def classify_sparse_shrinkage_primary(
    per_run_ratios: list[float | None],
) -> dict[str, Any]:
    """Classify primary pooling metric (posterior beta vs posterior mu_hat)."""
    valid = [float(r) for r in per_run_ratios if r is not None and r == r]
    if not valid:
        return {
            "metric": "shrinkage_ratio_sparse",
            "classification": "inconclusive",
            "reason": "no valid primary shrinkage values",
            "fraction_lt_1": None,
        }
    frac_lt = sum(1 for r in valid if r < 1.0) / len(valid)
    spread = max(valid) - min(valid)
    result: dict[str, Any] = {
        "metric": "shrinkage_ratio_sparse",
        "classification": "inconclusive",
        "reason": "",
        "fraction_lt_1": frac_lt,
        "per_run_values": valid,
        "aggregate": _agg_numeric(valid),
        "expect_direction": "ratio_lt_1",
        "pool_center": "posterior_mu_c",
    }
    if frac_lt >= 0.67:
        result["classification"] = "pooling_toward_posterior_mu_stable"
        result["reason"] = "majority of runs show shrinkage toward learned mu_hat (primary < 1)"
    elif spread > 0.35:
        result["classification"] = "still_unstable_across_seeds"
        result["reason"] = "high variance across seeds on primary pooling metric"
    elif all(r >= 1.0 for r in valid):
        result["classification"] = "pooling_not_observed"
        result["reason"] = "primary ratio >= 1 on all runs; review world/prior"
    else:
        result["reason"] = "mixed primary ratios; monitor with recovery metrics"
    return result


def classify_sparse_shrinkage_legacy(
    per_run_ratios: list[float | None],
    *,
    h4a_fast_reference: float | None = 2.57,
) -> dict[str, Any]:
    """Classify legacy recovery diagnostic (posterior beta vs true mu_star)."""
    valid = [float(r) for r in per_run_ratios if r is not None and r == r]
    if not valid:
        return {
            "metric": "shrinkage_ratio_sparse_vs_true_mu",
            "classification": "inconclusive",
            "reason": "no valid legacy shrinkage values",
            "fraction_lt_1": None,
        }
    frac_lt = sum(1 for r in valid if r < 1.0) / len(valid)
    all_gte_1 = all(r >= 1.0 for r in valid)
    result: dict[str, Any] = {
        "metric": "shrinkage_ratio_sparse_vs_true_mu",
        "classification": "recovery_diagnostic_only",
        "reason": "legacy compares to true mu_star; not a pooling-mechanics gate",
        "fraction_lt_1": frac_lt,
        "per_run_values": valid,
        "aggregate": _agg_numeric(valid),
        "h4a_fast_reference": h4a_fast_reference,
        "expect_direction": "ratio_lt_1_for_recovery",
        "pool_center": "true_mu_c",
        "not_a_pooling_gate": True,
    }
    if all_gte_1 and h4a_fast_reference is not None and statistics.mean(valid) >= h4a_fast_reference * 0.9:
        result["classification"] = "weak_recovery_vs_true_mu"
        result["reason"] = (
            "posterior not close to generative mu_star on sparse geo; "
            "evaluate beta_gc_mae and mu_c_mae separately from pooling"
        )
    elif frac_lt >= 0.67:
        result["classification"] = "recovery_vs_true_mu_ok"
        result["reason"] = "legacy ratio < 1 on majority of runs"
    return result


def classify_sparse_shrinkage(
    per_run_ratios: list[float | None],
    *,
    h4a_fast_reference: float | None = 2.57,
) -> dict[str, Any]:
    """Backward-compatible alias: legacy classification on primary ratio list."""
    return classify_sparse_shrinkage_legacy(per_run_ratios, h4a_fast_reference=h4a_fast_reference)


def aggregate_world_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run metrics for one world."""

    def _vals(key: str) -> list[float]:
        out: list[float] = []
        for r in runs:
            v = r.get(key)
            if v is not None and v == v:
                out.append(float(v))
        return out

    conflict_pass = sum(1 for r in runs if r.get("conflict_warnings")) / len(runs) if runs else 0.0
    shrink_primary = _vals("shrinkage_ratio_sparse")
    shrink_legacy = _vals("shrinkage_ratio_sparse_vs_true_mu")
    agg: dict[str, Any] = {
        "n_runs": len(runs),
        "beta_gc_mae": _agg_numeric(_vals("beta_gc_mae")),
        "mu_c_mae": _agg_numeric(_vals("mu_c_mae")),
        "beta_gc_coverage_90": _agg_numeric(_vals("beta_gc_coverage_90")),
        "shrinkage_ratio_sparse": _agg_numeric(shrink_primary),
        "shrinkage_ratio_sparse_vs_true_mu": _agg_numeric(shrink_legacy),
        "rhat_max": _agg_numeric(
            [float(r["convergence"]["rhat_max"]) for r in runs if r.get("convergence", {}).get("rhat_max") is not None]
        ),
        "conflict_warning_pass_rate": conflict_pass,
        "sparse_shrinkage_warning_primary": any(v >= 1.0 for v in shrink_primary) if shrink_primary else None,
        "sparse_shrinkage_warning_legacy": any(v >= 1.0 for v in shrink_legacy) if shrink_legacy else None,
    }
    if shrink_primary:
        agg["sparse_shrinkage_interpretation_primary"] = classify_sparse_shrinkage_primary(shrink_primary)
    if shrink_legacy:
        agg["sparse_shrinkage_interpretation_legacy"] = classify_sparse_shrinkage_legacy(shrink_legacy)
    return agg


def build_repeated_pilot_summary(
    per_run_rows: list[dict[str, Any]],
    *,
    pilot_id: str = PILOT_ID_PRIMARY,
    pilot_version: str = PILOT_VERSION_PRIMARY,
    seeds: tuple[int, ...] = DEFAULT_NUTS_SEEDS,
    panel_seed: int = DEFAULT_PANEL_SEED,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler_settings = dict(sampler or SAMPLER_EXTENDED)
    by_world: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        by_world.setdefault(str(row["world_id"]), []).append(row)

    world_agg = {wid: aggregate_world_runs(by_world[wid]) for wid in sorted(by_world)}
    sparse_primary = [
        float(r["shrinkage_ratio_sparse"])
        for r in per_run_rows
        if r.get("world_id") == WORLD_BAYES_H4_SPARSE_GEO and r.get("shrinkage_ratio_sparse") is not None
    ]
    sparse_legacy = [
        float(r["shrinkage_ratio_sparse_vs_true_mu"])
        for r in per_run_rows
        if r.get("world_id") == WORLD_BAYES_H4_SPARSE_GEO and r.get("shrinkage_ratio_sparse_vs_true_mu") is not None
    ]
    sparse_world_agg = world_agg.get(WORLD_BAYES_H4_SPARSE_GEO, {})
    sparse_interp_primary = sparse_world_agg.get("sparse_shrinkage_interpretation_primary", {})
    sparse_interp_legacy = sparse_world_agg.get("sparse_shrinkage_interpretation_legacy", {})
    conflict_rows = [r for r in per_run_rows if "CONFLICTING" in str(r.get("world_id", ""))]
    conflict_pass_rate = (
        sum(1 for r in conflict_rows if r.get("conflict_warnings")) / len(conflict_rows) if conflict_rows else None
    )
    backends: dict[str, Any] = {}
    for wid in sorted({str(r["world_id"]) for r in per_run_rows}):
        backends[wid] = _backend_metadata(get_recovery_world(wid), fast_mcmc=False)

    return _json_safe(
        {
            "pilot_id": pilot_id,
            "pilot_version": pilot_version,
            "status": "complete",
            "label": "RESEARCH ONLY — NOT DECISION GRADE",
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "outputs_are_diagnostic_only": True,
            "metric_definitions": METRIC_DEFINITIONS,
            "interpretation": {
                "report_only": True,
                "hard_gate": False,
                "production_promotion": False,
                "inv_h4_001b_status": "closed",
                "primary_shrinkage_role": "pooling_mechanics_only",
                "legacy_shrinkage_role": "true_effect_recovery_diagnostic_not_pooling_gate",
                "true_effect_recovery": {
                    "status": "open",
                    "metrics": ["beta_gc_mae", "mu_c_mae", "beta_gc_coverage_90", "shrinkage_ratio_sparse_vs_true_mu"],
                    "note": "Primary shrinkage < 1 does not prove recovery of generative mu_star or beta_gc.",
                },
                "h4c_blocked": True,
                "h4c_unblock_requires": "INV-H4-001 disposition C+A explicitly accepted",
                "note": "Extended repeated pilot with corrected primary shrinkage vs posterior mu_hat.",
                "sparse_shrinkage_summary_primary": sparse_interp_primary,
                "sparse_shrinkage_summary_legacy": sparse_interp_legacy,
                "conflict_warning_pass_rate": conflict_pass_rate,
            },
            "seeds": list(seeds),
            "panel_seed": panel_seed,
            "sampler_settings": sampler_settings,
            "backend_defaults": backends,
            "world_ids": list(H4_WORLD_IDS),
            "sparse_shrinkage_distribution_primary": _agg_numeric(sparse_primary),
            "sparse_shrinkage_distribution_legacy": _agg_numeric(sparse_legacy),
            "conflict_warning_pass_rate": conflict_pass_rate,
            "per_run": per_run_rows,
            "aggregate_by_world": world_agg,
        }
    )


def run_h4_repeated_pilot(
    world_ids: tuple[str, ...] | None = None,
    *,
    nuts_seeds: tuple[int, ...] = DEFAULT_NUTS_SEEDS,
    panel_seed: int = DEFAULT_PANEL_SEED,
    sampler: dict[str, Any] | None = None,
    pilot_id: str = PILOT_ID_PRIMARY,
    pilot_version: str = PILOT_VERSION_PRIMARY,
) -> dict[str, Any]:
    """Run repeated H4 recovery pilots with extended sampling (research only)."""
    ids = world_ids or H4_WORLD_IDS
    sampler_settings = dict(sampler or SAMPLER_EXTENDED)
    rows: list[dict[str, Any]] = []
    for wid in ids:
        for nuts_seed in nuts_seeds:
            rows.append(
                _run_row(
                    wid,
                    nuts_seed=int(nuts_seed),
                    panel_seed=int(panel_seed),
                    sampler=sampler_settings,
                )
            )
    return build_repeated_pilot_summary(
        rows,
        pilot_id=pilot_id,
        pilot_version=pilot_version,
        seeds=nuts_seeds,
        panel_seed=panel_seed,
        sampler=sampler_settings,
    )


def write_h4_repeated_pilot_artifact(
    path: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path or PRIMARY_ARTIFACT_PATH)
    payload = summary if summary is not None else run_h4_repeated_pilot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_h4_repeated_pilot_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or PRIMARY_ARTIFACT_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bayes-H4b repeated recovery pilot")
    parser.add_argument("--output", type=Path, default=PRIMARY_ARTIFACT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_NUTS_SEEDS))
    args = parser.parse_args()
    out = write_h4_repeated_pilot_artifact(
        args.output,
        run_h4_repeated_pilot(nuts_seeds=tuple(args.seeds)),
    )
    print(json.dumps({"written": str(out), "pilot_id": PILOT_ID_PRIMARY}))


if __name__ == "__main__":
    main()
