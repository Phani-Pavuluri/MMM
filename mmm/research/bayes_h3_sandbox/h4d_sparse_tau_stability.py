"""Bayes-H4d sparse/τ tuning and recovery-candidate stability pilot (research only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import (
    RECOVERY_CANDIDATE_WORLDS,
    STRESS_DIAGNOSTIC_WORLDS,
    evaluate_world_against_policy,
    world_policy_role,
)
from mmm.research.bayes_h3_sandbox.h4_repeated_pilot import _agg_numeric
from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import (
    _backend_metadata,
    _json_safe,
    _world_row_from_report,
)
from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import (
    WORLD_BAYES_H4C_CLEAN_RECOVERY,
    WORLD_BAYES_H4C_SPARSE_RECOVERY,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    SAMPLER_EXTENDED,
    SAMPLER_FAST,
    SPARSE_POOLING_DIAGNOSTIC_WORLD_IDS,
    WORLD_BAYES_H4_SIMPLE_POOLING,
    WORLD_BAYES_H4_SPARSE_GEO,
    get_recovery_world,
)

PILOT_ID = "BAYES_H4D_SPARSE_TAU_STABILITY_20260601"
PILOT_VERSION = "bayes_h4d_sparse_tau_stability_v1"
INVESTIGATION_ID = "INV-H4D"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H4D_SPARSE_TAU_STABILITY_20260601.json")

DEFAULT_PANEL_SEED = 4400
DEFAULT_NUTS_SEEDS: tuple[int, ...] = (4400, 4401, 4402)

H4D_WORLD_IDS: tuple[str, ...] = (
    WORLD_BAYES_H4C_CLEAN_RECOVERY,
    WORLD_BAYES_H4C_SPARSE_RECOVERY,
    WORLD_BAYES_H4_SIMPLE_POOLING,
    WORLD_BAYES_H4_SPARSE_GEO,
)

# τ prior grid: None = model default (0.5 in model.py).
TAU_GRID: tuple[dict[str, Any], ...] = (
    {"label": "default", "tau_channel_prior_sigma": None},
    {"label": "tau_0.30", "tau_channel_prior_sigma": 0.30},
    {"label": "tau_0.20", "tau_channel_prior_sigma": 0.20},
    {"label": "tau_0.15", "tau_channel_prior_sigma": 0.15},
)

INPUT_ARTIFACT_REFERENCES: tuple[str, ...] = (
    "docs/05_validation/archives/BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json",
    "docs/05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json",
    "docs/05_validation/archives/BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json",
    "docs/05_validation/archives/BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json",
)

STABILITY_CV_THRESHOLD = 0.25
STABILITY_ABS_SPREAD_THRESHOLD = 0.12


def h4d_tau_grid() -> tuple[dict[str, Any], ...]:
    """Deterministic τ grid for tests and pilot."""
    return TAU_GRID


def h4d_world_ids(*, include_stress: bool = True) -> tuple[str, ...]:
    if include_stress:
        return H4D_WORLD_IDS
    return tuple(w for w in H4D_WORLD_IDS if w not in STRESS_DIAGNOSTIC_WORLDS)


def _warning_summary(warnings: list[str]) -> dict[str, Any]:
    types: set[str] = set()
    for w in warnings:
        if w.startswith("h4c:"):
            parts = w.split(":")
            if len(parts) >= 2:
                types.add(parts[1])
        elif w.startswith("conflict:"):
            types.add("conflict")
        elif ":" in w:
            types.add(w.split(":")[0])
    return {
        "count": len(warnings),
        "types": sorted(types),
        "messages": list(warnings),
    }


def classify_metric_stability(
    values: list[float],
    *,
    cv_threshold: float = STABILITY_CV_THRESHOLD,
    abs_spread_threshold: float = STABILITY_ABS_SPREAD_THRESHOLD,
) -> dict[str, Any]:
    """Seed stability for one metric (report-only; not a CI gate)."""
    if len(values) < 2:
        return {
            "stable": None,
            "n": len(values),
            "reason": "insufficient_runs_for_stability",
            "aggregate": _agg_numeric(values),
        }
    agg = _agg_numeric(values)
    mean = float(agg["mean"])
    std = float(agg["std"])
    spread = float(agg["max"]) - float(agg["min"])
    cv = std / mean if abs(mean) > 1e-9 else float("inf")
    stable = cv <= cv_threshold and spread <= abs_spread_threshold
    return {
        "stable": stable,
        "n": len(values),
        "cv": cv,
        "spread": spread,
        "cv_threshold": cv_threshold,
        "abs_spread_threshold": abs_spread_threshold,
        "aggregate": agg,
        "reason": "stable_across_seeds" if stable else "unstable_across_seeds",
    }


def _run_h4d_row(
    world_id: str,
    *,
    nuts_seed: int,
    panel_seed: int,
    tau_entry: dict[str, Any],
    sampler: dict[str, Any],
    fast_mcmc: bool,
    sparse_variant: str | None = None,
) -> dict[str, Any]:
    spec = get_recovery_world(world_id)
    tau_sigma = tau_entry.get("tau_channel_prior_sigma")
    overrides: dict[str, Any] | None = None
    if tau_sigma is not None:
        overrides = {"tau_channel_prior_sigma": float(tau_sigma)}

    report = run_h4_recovery_world(
        world_id,
        fast_mcmc=fast_mcmc,
        sampler=sampler,
        nuts_seed=nuts_seed,
        panel_seed=panel_seed,
        sandbox_model_overrides=overrides,
    )
    rec = report.get("h4_recovery") or {}
    row = _world_row_from_report(report, spec)
    h4c_warn = list(rec.get("h4c_diagnostic_warnings") or [])
    conflict_warn = list(rec.get("conflict_warnings") or [])
    all_warn = h4c_warn + [w for w in conflict_warn if w not in h4c_warn]

    metrics_for_policy = {
        "beta_gc_mae": rec.get("beta_gc_mae"),
        "mu_c_mae": rec.get("mu_c_mae"),
        "beta_gc_coverage_90": rec.get("beta_gc_coverage_90"),
        "beta_interval_width_90_mean": rec.get("beta_interval_width_90_mean"),
    }
    policy_eval = evaluate_world_against_policy(world_id, metrics_for_policy)

    row.update(
        {
            "nuts_seed": nuts_seed,
            "panel_seed": panel_seed,
            "tau_label": str(tau_entry["label"]),
            "tau_channel_prior_sigma": tau_sigma,
            "sampler": dict(sampler),
            "fast_mcmc_profile": fast_mcmc,
            "sparse_variant": sparse_variant,
            "world_role": world_policy_role(world_id),
            "h4c_classification": (spec.expected_diagnostic_behavior or {}).get("h4c_classification"),
            "beta_interval_width_90_mean": rec.get("beta_interval_width_90_mean"),
            "h4c_diagnostic_warnings": h4c_warn,
            "warning_summary": _warning_summary(all_warn),
            "policy_evaluation": policy_eval,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "hard_gate": False,
        }
    )
    return row


def aggregate_h4d_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run metrics for one (world, tau_label) group."""

    def _vals(key: str) -> list[float]:
        out: list[float] = []
        for r in runs:
            v = r.get(key)
            if v is not None and v == v:
                out.append(float(v))
        return out

    beta_vals = _vals("beta_gc_mae")
    mu_vals = _vals("mu_c_mae")
    return {
        "n_runs": len(runs),
        "beta_gc_mae": _agg_numeric(beta_vals),
        "mu_c_mae": _agg_numeric(mu_vals),
        "beta_gc_coverage_90": _agg_numeric(_vals("beta_gc_coverage_90")),
        "beta_interval_width_90_mean": _agg_numeric(_vals("beta_interval_width_90_mean")),
        "shrinkage_ratio_sparse": _agg_numeric(_vals("shrinkage_ratio_sparse")),
        "shrinkage_ratio_sparse_vs_true_mu": _agg_numeric(_vals("shrinkage_ratio_sparse_vs_true_mu")),
        "stability": {
            "beta_gc_mae": classify_metric_stability(beta_vals),
            "mu_c_mae": classify_metric_stability(mu_vals),
        },
        "policy_outcomes": sorted({str(r.get("policy_evaluation", {}).get("outcome")) for r in runs}),
        "warning_summary_aggregate": {
            "max_warning_count": max((r.get("warning_summary") or {}).get("count", 0) for r in runs) if runs else 0,
            "warning_types_union": sorted({t for r in runs for t in (r.get("warning_summary") or {}).get("types", [])}),
        },
    }


def _group_key(row: dict[str, Any]) -> tuple[str, str, str | None]:
    return (
        str(row["world_id"]),
        str(row["tau_label"]),
        row.get("sparse_variant"),
    )


def recommend_disposition(
    aggregated: dict[str, dict[str, Any]],
    *,
    clean_world: str = WORLD_BAYES_H4C_CLEAN_RECOVERY,
    sparse_recovery_world: str = WORLD_BAYES_H4C_SPARSE_RECOVERY,
    stress_world: str = WORLD_BAYES_H4_SPARSE_GEO,
) -> dict[str, Any]:
    """
    Report-only disposition from aggregated stability (not production promotion).

    Decision logic (INV-H4D):
    - CLEAN unstable → model/spec work before thresholds
    - CLEAN stable, SPARSE-RECOVERY unstable → keep sparse claim restricted
    - Stronger τ helps sparse without hurting clean → recommend sandbox τ research update
    - SPARSE-GEO remains bad → stress diagnostic only
    """

    def _beta_stable(world: str, tau_label: str = "default") -> bool | None:
        key = f"{world}|{tau_label}|None"
        entry = aggregated.get(key)
        if not entry:
            return None
        return (entry.get("stability") or {}).get("beta_gc_mae", {}).get("stable")

    def _beta_mean(world: str, tau_label: str) -> float | None:
        key = f"{world}|{tau_label}|None"
        entry = aggregated.get(key)
        if not entry:
            return None
        return (entry.get("beta_gc_mae") or {}).get("mean")

    clean_stable = _beta_stable(clean_world, "default")
    sparse_rec_stable = _beta_stable(sparse_recovery_world, "default")
    clean_default_mae = _beta_mean(clean_world, "default")
    clean_tau15_mae = _beta_mean(clean_world, "tau_0.15")
    sparse_default_mae = _beta_mean(sparse_recovery_world, "default")
    sparse_tau15_mae = _beta_mean(sparse_recovery_world, "tau_0.15")
    stress_default_mae = _beta_mean(stress_world, "default")

    tau_helps_sparse = (
        sparse_default_mae is not None and sparse_tau15_mae is not None and sparse_tau15_mae < sparse_default_mae * 0.95
    )
    tau_hurts_clean = (
        clean_default_mae is not None and clean_tau15_mae is not None and clean_tau15_mae > clean_default_mae * 1.10
    )

    disposition = "continue_report_only_monitoring"
    reasons: list[str] = []

    if clean_stable is False:
        disposition = "model_spec_work_needed_before_thresholds"
        reasons.append("CLEAN-RECOVERY beta_gc_mae unstable across seeds at default τ")
    elif sparse_rec_stable is False:
        disposition = "keep_sparse_claim_restricted"
        reasons.append("SPARSE-RECOVERY unstable across seeds; do not generalize sparse recovery claim")
    elif tau_helps_sparse and not tau_hurts_clean:
        disposition = "recommend_tau_prior_sandbox_research_update"
        reasons.append("τ=0.15 improves SPARSE-RECOVERY MAE without materially hurting CLEAN-RECOVERY")
    else:
        reasons.append("No strong τ tuning signal; maintain INV-071 report-only thresholds")

    if stress_default_mae is not None and stress_default_mae > 0.40:
        reasons.append(
            f"{stress_world} remains elevated (beta_gc_mae≈{stress_default_mae:.3f}); keep as stress_diagnostic only"
        )

    return {
        "disposition": disposition,
        "reasons": reasons,
        "clean_recovery_stable": clean_stable,
        "sparse_recovery_stable": sparse_rec_stable,
        "tau_helps_sparse_recovery": tau_helps_sparse,
        "tau_hurts_clean_recovery": tau_hurts_clean,
        "sparse_stress_separate_role": True,
        "hard_gate": False,
        "production_promotion": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "does_not_authorize_production": True,
    }


def build_h4d_summary(
    per_run_rows: list[dict[str, Any]],
    *,
    pilot_id: str = PILOT_ID,
    nuts_seeds: tuple[int, ...] = DEFAULT_NUTS_SEEDS,
    panel_seed: int = DEFAULT_PANEL_SEED,
    sampler: dict[str, Any] | None = None,
    fast_mcmc: bool = True,
    tau_grid: tuple[dict[str, Any], ...] | None = None,
    include_sparse_variants: bool = False,
) -> dict[str, Any]:
    """Build H4d stability summary from per-run rows."""
    sampler_settings = dict(sampler or (SAMPLER_FAST if fast_mcmc else SAMPLER_EXTENDED))
    grid = tau_grid or TAU_GRID

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        wid, tau, var = _group_key(row)
        key = f"{wid}|{tau}|{var}"
        by_group.setdefault(key, []).append(row)

    aggregated = {k: aggregate_h4d_runs(v) for k, v in sorted(by_group.items())}

    world_roles = {wid: world_policy_role(wid) for wid in sorted({str(r["world_id"]) for r in per_run_rows})}
    backends = {
        wid: _backend_metadata(get_recovery_world(wid), fast_mcmc=fast_mcmc)
        for wid in sorted({str(r["world_id"]) for r in per_run_rows})
    }

    disposition = recommend_disposition(aggregated)

    return _json_safe(
        {
            "pilot_id": pilot_id,
            "pilot_version": PILOT_VERSION,
            "investigation_id": INVESTIGATION_ID,
            "status": "complete",
            "label": "RESEARCH ONLY — NOT DECISION GRADE",
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "hard_gate": False,
            "outputs_are_diagnostic_only": True,
            "input_artifact_references": list(INPUT_ARTIFACT_REFERENCES),
            "interpretation": {
                "report_only": True,
                "hard_gate": False,
                "production_promotion": False,
                "does_not_claim_global_recovery": True,
                "pooling_not_true_effect_gate": True,
                "sparse_recovery_vs_sparse_stress_separate": True,
                "inv_071_thresholds_remain_report_only": True,
                "question": (
                    "Are recovery_candidate worlds stable across seeds, "
                    "and does τ prior tuning help sparse recovery without hurting clean recovery?"
                ),
                "not_redoing_h4c": "H4c reliability map complete; H4d tunes τ and tests seed stability",
            },
            "world_ids": list(H4D_WORLD_IDS),
            "recovery_candidate_worlds": sorted(RECOVERY_CANDIDATE_WORLDS),
            "stress_diagnostic_worlds": sorted(STRESS_DIAGNOSTIC_WORLDS),
            "world_roles": world_roles,
            "nuts_seeds": list(nuts_seeds),
            "panel_seed": panel_seed,
            "tau_grid": list(grid),
            "sampler_settings": sampler_settings,
            "fast_mcmc_profile": fast_mcmc,
            "include_sparse_diagnostic_variants": include_sparse_variants,
            "backend_defaults": backends,
            "per_run": per_run_rows,
            "aggregated_by_world_tau": aggregated,
            "recommended_disposition": disposition,
        }
    )


def run_h4d_stability_pilot(
    *,
    world_ids: tuple[str, ...] | None = None,
    tau_grid: tuple[dict[str, Any], ...] | None = None,
    nuts_seeds: tuple[int, ...] = DEFAULT_NUTS_SEEDS,
    panel_seed: int = DEFAULT_PANEL_SEED,
    fast_mcmc: bool = True,
    sampler: dict[str, Any] | None = None,
    include_sparse_diagnostic_variants: bool = False,
) -> dict[str, Any]:
    """Run H4d τ × seed stability pilot (research only)."""
    ids = world_ids or H4D_WORLD_IDS
    grid = tau_grid or TAU_GRID
    sampler_settings = dict(sampler or (SAMPLER_FAST if fast_mcmc else SAMPLER_EXTENDED))

    rows: list[dict[str, Any]] = []
    for wid in ids:
        for tau_entry in grid:
            for nuts_seed in nuts_seeds:
                rows.append(
                    _run_h4d_row(
                        wid,
                        nuts_seed=nuts_seed,
                        panel_seed=panel_seed,
                        tau_entry=tau_entry,
                        sampler=sampler_settings,
                        fast_mcmc=fast_mcmc,
                    )
                )

    if include_sparse_diagnostic_variants:
        for variant_wid in SPARSE_POOLING_DIAGNOSTIC_WORLD_IDS:
            for tau_entry in grid:
                for nuts_seed in nuts_seeds:
                    rows.append(
                        _run_h4d_row(
                            variant_wid,
                            nuts_seed=nuts_seed,
                            panel_seed=panel_seed,
                            tau_entry=tau_entry,
                            sampler=sampler_settings,
                            fast_mcmc=fast_mcmc,
                            sparse_variant=variant_wid,
                        )
                    )

    return build_h4d_summary(
        rows,
        nuts_seeds=nuts_seeds,
        panel_seed=panel_seed,
        sampler=sampler_settings,
        fast_mcmc=fast_mcmc,
        tau_grid=grid,
        include_sparse_variants=include_sparse_diagnostic_variants,
    )


def write_h4d_stability_artifact(
    path: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path or DEFAULT_ARTIFACT_PATH)
    payload = summary if summary is not None else run_h4d_stability_pilot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_h4d_stability_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_ARTIFACT_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bayes-H4d sparse/τ stability pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--fast-mcmc", action="store_true", default=True)
    parser.add_argument("--extended-mcmc", action="store_true", default=False)
    parser.add_argument("--include-sparse-variants", action="store_true", default=False)
    args = parser.parse_args()
    fast = not args.extended_mcmc
    summary = run_h4d_stability_pilot(
        fast_mcmc=fast,
        include_sparse_diagnostic_variants=args.include_sparse_variants,
    )
    out = write_h4d_stability_artifact(args.output, summary)
    print(json.dumps({"written": str(out), "pilot_id": PILOT_ID, "n_runs": len(summary["per_run"])}))


if __name__ == "__main__":
    main()
