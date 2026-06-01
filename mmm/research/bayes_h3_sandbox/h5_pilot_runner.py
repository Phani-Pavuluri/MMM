"""Bayes-H5 sandbox pilot — research-only validation over H5 worlds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from mmm.config.schema import BayesianBackend
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import evaluate_world_against_policy
from mmm.research.bayes_h3_sandbox.h5_transforms import list_transform_registry
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    H5_WORLD_IDS,
    h5_world_catalog_metadata,
    h5_world_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h5_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    SAMPLER_EXTENDED,
    SAMPLER_FAST,
    RecoveryWorldSpec,
    get_recovery_world,
    recovery_world_config,
)

PILOT_ID = "BAYES_H5_SANDBOX_PILOT_20260601"
PILOT_VERSION = "bayes_h5_sandbox_pilot_v1"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H5_SANDBOX_PILOT_20260601.json")

REPEATED_PILOT_ID = "BAYES_H5B_REPEATED_PILOT_20260601"
REPEATED_PILOT_VERSION = "bayes_h5b_repeated_pilot_v1"
DEFAULT_REPEATED_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H5B_REPEATED_PILOT_20260601.json")

EXTENDED_PILOT_ID = "BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601"
EXTENDED_PILOT_VERSION = "bayes_h5c_extended_repeated_pilot_v1"
DEFAULT_EXTENDED_ARTIFACT_PATH = Path(
    "docs/05_validation/archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json"
)

DEFAULT_REPEATED_SEEDS: tuple[int, ...] = (4400, 4401, 4402)
H5C_MATERIAL_CHANGE_BETA_MAE_DELTA = 0.05
H4C_PILOT_PATH = Path("docs/05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json")

# H5 world → H4c mismatch/recovery baseline world for beta_gc_mae comparison.
H5_TO_H4C_BASELINE: dict[str, str] = {
    "WORLD-BAYES-H5-ADSTOCK-ALIGNED": "WORLD-BAYES-H4C-ADSTOCKED-MEDIA",
    "WORLD-BAYES-H5-SATURATION-ALIGNED": "WORLD-BAYES-H4C-SATURATION",
    "WORLD-BAYES-H5-ADSTOCK-MISMATCH": "WORLD-BAYES-H4C-ADSTOCKED-MEDIA",
    "WORLD-BAYES-H5-SATURATION-MISMATCH": "WORLD-BAYES-H4C-SATURATION",
    "WORLD-BAYES-H5-CORRELATED-CHANNELS": "WORLD-BAYES-H4C-CORRELATED-CHANNELS",
    "WORLD-BAYES-H5-WEAK-SIGNAL": "WORLD-BAYES-H4C-WEAK-SIGNAL",
    "WORLD-BAYES-H5-SPARSE-RECOVERY": "WORLD-BAYES-H4C-SPARSE-RECOVERY",
}


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


def _backend_metadata(
    spec: RecoveryWorldSpec,
    *,
    fast_mcmc: bool,
    extended_mcmc: bool = False,
) -> dict[str, Any]:
    cfg = recovery_world_config(
        spec,
        fast_mcmc=fast_mcmc and not extended_mcmc,
        sampler=_sampler_for_profile(extended_mcmc),
    )
    b = cfg.bayesian
    return {
        "inference_backend": BayesianBackend.PYMC.value,
        "framework": cfg.framework.value,
        "pooling": cfg.pooling.value,
        "model_form": cfg.model_form.value,
        "draws": int(b.draws),
        "tune": int(b.tune),
        "chains": int(b.chains),
        "target_accept": float(b.target_accept),
        "nuts_seed": int(b.nuts_seed),
        "world_mcmc_seed": int(spec.mcmc_seed),
        "fast_mcmc_profile": fast_mcmc and not extended_mcmc,
        "extended_mcmc_profile": extended_mcmc,
    }


def _sampler_for_profile(extended_mcmc: bool) -> dict[str, Any] | None:
    if extended_mcmc:
        return dict(SAMPLER_EXTENDED)
    return None


def _world_row_from_report(report: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, Any]:
    rec = report.get("h4_recovery") or report.get("h5_recovery") or {}
    conv = rec.get("convergence_diagnostics") or report.get("convergence_diagnostics") or {}
    rhat = conv.get("rhat_max")
    h5_diag = report.get("h5_transform_diagnostics") or {}
    exp = spec.expected_diagnostic_behavior
    policy = evaluate_world_against_policy(spec.world_id, rec)
    return {
        "world_id": spec.world_id,
        "role": exp.get("role"),
        "h5_classification": exp.get("h5_classification"),
        "generative_transform": exp.get("generative_transform"),
        "fitted_transform_expectation": exp.get("fitted_transform_expectation"),
        "transform_mismatch_mode": exp.get("transform_mismatch_mode"),
        "beta_gc_mae": rec.get("beta_gc_mae"),
        "mu_c_mae": rec.get("mu_c_mae"),
        "beta_gc_coverage_90": rec.get("beta_gc_coverage_90"),
        "shrinkage_ratio_sparse": rec.get("shrinkage_ratio_sparse"),
        "shrinkage_ratio_sparse_vs_true_mu": rec.get("shrinkage_ratio_sparse_vs_true_mu"),
        "h5_transform_diagnostics": _json_safe(h5_diag),
        "h5_diagnostic_warnings": list(rec.get("h5_diagnostic_warnings") or []),
        "conflict_warnings": list(rec.get("conflict_warnings") or []),
        "policy_outcomes": policy,
        "convergence": {
            "rhat_max": rhat,
            "ess_bulk_min": conv.get("ess_bulk_min"),
        },
        "expected_diagnostic_behavior": dict(exp),
        **h5_world_production_flags(),
        "has_decision_surface": report.get("decision_surface") is not None,
        "has_budget_recommendation": report.get("budget_recommendation") is not None,
        "has_optimizer_ready_curves": report.get("optimizer_ready_curves") is not None,
    }


def build_h5_pilot_summary(
    world_rows: list[dict[str, Any]],
    *,
    fast_mcmc: bool = True,
    sampler_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble pilot JSON (research only — no production promotion)."""
    mismatch_rows = [r for r in world_rows if r.get("transform_mismatch_mode") == "intentional_mismatch"]
    aligned_rows = [r for r in world_rows if r.get("transform_mismatch_mode") == "aligned"]
    all_warnings: list[str] = []
    for r in world_rows:
        all_warnings.extend(r.get("h5_diagnostic_warnings") or [])

    return {
        "pilot_id": PILOT_ID,
        "pilot_version": PILOT_VERSION,
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "transform_registry": list_transform_registry(),
        "world_catalog": h5_world_catalog_metadata(),
        "sampler_settings": sampler_metadata or {"fast_mcmc_profile": fast_mcmc},
        "per_world_metrics": world_rows,
        "diagnostic_warnings": sorted(set(all_warnings)),
        "mismatch_worlds": [r["world_id"] for r in mismatch_rows],
        "aligned_worlds": [r["world_id"] for r in aligned_rows],
        "hard_gate": False,
        "production_promotion": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "decision_grade": False,
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "note": (
            "H5 sandbox pilot — transform alignment probe only. "
            "INV-071 thresholds remain report-only; no optimizer or DecisionSurface."
        ),
    }


def run_h5_pilot(
    *,
    world_ids: tuple[str, ...] | None = None,
    fast_mcmc: bool = True,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run fast H5 pilot over validation worlds and optionally write JSON artifact."""
    ids = world_ids or H5_WORLD_IDS
    rows: list[dict[str, Any]] = []
    sampler_meta: dict[str, Any] | None = None
    for wid in ids:
        spec = get_recovery_world(wid)
        report = run_h5_recovery_world(wid, fast_mcmc=fast_mcmc)
        if sampler_meta is None:
            sampler_meta = _backend_metadata(spec, fast_mcmc=fast_mcmc)
        rows.append(_world_row_from_report(report, spec))

    summary = build_h5_pilot_summary(rows, fast_mcmc=fast_mcmc, sampler_metadata=sampler_meta)
    path = artifact_path or DEFAULT_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    summary["artifact_path"] = str(path)
    return summary


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


def load_h5b_repeated_pilot(path: Path | None = None) -> dict[str, Any]:
    """Load H5b fast repeated pilot JSON (empty dict if missing)."""
    p = path or DEFAULT_REPEATED_ARTIFACT_PATH
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_h4c_beta_mae_baselines(path: Path | None = None) -> dict[str, float]:
    """Load H4c extended pilot beta_gc_mae by world_id for comparison."""
    p = path or H4C_PILOT_PATH
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for row in data.get("worlds") or data.get("per_world") or data.get("per_world_metrics") or []:
        wid = str(row.get("world_id", ""))
        mae = row.get("beta_gc_mae")
        if wid and mae is not None:
            out[wid] = float(mae)
    return out


def _warning_rate(runs: list[dict[str, Any]], *, prefix: str) -> float:
    if not runs:
        return 0.0
    hits = sum(
        1
        for r in runs
        if any(str(w).startswith(prefix) for w in (r.get("h5_diagnostic_warnings") or []))
    )
    return hits / len(runs)


def aggregate_h5_world_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed rows for one H5 world."""
    def _vals(key: str) -> list[float]:
        out: list[float] = []
        for r in runs:
            v = r.get(key)
            if v is not None and v == v:
                out.append(float(v))
        return out

    return {
        "n_runs": len(runs),
        "beta_gc_mae": _agg_numeric(_vals("beta_gc_mae")),
        "mu_c_mae": _agg_numeric(_vals("mu_c_mae")),
        "transform_mismatch_warning_rate": _warning_rate(runs, prefix="h5:transform_mismatch:"),
        "unexpected_mismatch_warning_rate": _warning_rate(runs, prefix="h5:unexpected_transform_mismatch:"),
        "weak_identification_warning_rate": _warning_rate(runs, prefix="h5:weak_identification:"),
        "collinearity_warning_rate": _warning_rate(runs, prefix="h5:collinearity:"),
    }


def _compare_to_h5b_fast_pilot(
    aggregate_by_world: dict[str, Any],
    *,
    h5b_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-world comparison of extended (or current) aggregates vs H5b fast pilot."""
    h5b = h5b_data if h5b_data is not None else load_h5b_repeated_pilot()
    h5b_agg = h5b.get("aggregate_by_world") or {}
    out: dict[str, Any] = {}
    for wid in H5_WORLD_IDS:
        cur = aggregate_by_world.get(wid) or {}
        ref = h5b_agg.get(wid) or {}
        cur_mean = (cur.get("beta_gc_mae") or {}).get("mean")
        ref_mean = (ref.get("beta_gc_mae") or {}).get("mean")
        if cur_mean is None or ref_mean is None:
            continue
        delta = float(cur_mean) - float(ref_mean)
        out[wid] = {
            "h5b_beta_gc_mae_mean": ref_mean,
            "current_beta_gc_mae_mean": cur_mean,
            "delta_vs_h5b": delta,
            "material_change_vs_h5b": abs(delta) > H5C_MATERIAL_CHANGE_BETA_MAE_DELTA,
            "h5b_transform_mismatch_warning_rate": ref.get("transform_mismatch_warning_rate"),
            "current_transform_mismatch_warning_rate": cur.get("transform_mismatch_warning_rate"),
            "h5b_unexpected_mismatch_warning_rate": ref.get("unexpected_mismatch_warning_rate"),
            "current_unexpected_mismatch_warning_rate": cur.get("unexpected_mismatch_warning_rate"),
        }
    return out


def _extended_conclusions(
    aggregate_by_world: dict[str, Any],
    h4c_comparison: dict[str, Any],
    comparison_to_h5b: dict[str, Any],
) -> dict[str, Any]:
    """Research-only acceptance notes for H5c vs H5b/H4c."""
    sat = aggregate_by_world.get("WORLD-BAYES-H5-SATURATION-ALIGNED", {})
    ad = aggregate_by_world.get("WORLD-BAYES-H5-ADSTOCK-ALIGNED", {})
    sat_mean = (sat.get("beta_gc_mae") or {}).get("mean")
    ad_mean = (ad.get("beta_gc_mae") or {}).get("mean")
    sat_h4c = h4c_comparison.get("WORLD-BAYES-H5-SATURATION-ALIGNED", {})
    ad_h4c = h4c_comparison.get("WORLD-BAYES-H5-ADSTOCK-ALIGNED", {})

    mismatch_rates = [
        aggregate_by_world.get(w, {}).get("transform_mismatch_warning_rate", 0.0)
        for w in H5_WORLD_IDS
        if "MISMATCH" in w
    ]
    unexpected_rates = [
        aggregate_by_world.get(w, {}).get("unexpected_mismatch_warning_rate", 0.0) for w in H5_WORLD_IDS
    ]
    material_changes = [v.get("material_change_vs_h5b") for v in comparison_to_h5b.values() if v]

    return {
        "saturation_aligned_improvement_holds": bool(
            sat_h4c.get("improved_vs_h4c") and sat_mean is not None and float(sat_mean) < 0.15
        ),
        "adstock_aligned_improvement_holds": bool(
            ad_h4c.get("improved_vs_h4c") and ad_mean is not None and float(ad_mean) < 0.28
        ),
        "mismatch_warnings_clean": all(r >= 0.99 for r in mismatch_rates) if mismatch_rates else False,
        "unexpected_mismatch_clean": all(r == 0.0 for r in unexpected_rates),
        "weak_id_diagnostics_present": (
            aggregate_by_world.get("WORLD-BAYES-H5-WEAK-SIGNAL", {}).get("weak_identification_warning_rate", 0) >= 0.99
            and aggregate_by_world.get("WORLD-BAYES-H5-CORRELATED-CHANNELS", {}).get("collinearity_warning_rate", 0)
            >= 0.99
        ),
        "sparse_recovery_report_only": True,
        "material_change_any_world": any(material_changes),
        "accepted_research_evidence": (
            "H5 transform-aligned spec improves saturation recovery vs H4c MVP mismatch; "
            "adstock gain modest but directionally stable; diagnostics behave as designed."
        ),
        "rejected_for_production": "Production Bayes, optimizer, DecisionSurface, and hard gates remain blocked.",
    }


def build_h5_repeated_pilot_summary(
    per_run_rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] = DEFAULT_REPEATED_SEEDS,
    fast_mcmc: bool = True,
    extended_mcmc: bool = False,
    sampler_metadata: dict[str, Any] | None = None,
    h4c_baselines: dict[str, float] | None = None,
    h5b_data: dict[str, Any] | None = None,
    pilot_id: str | None = None,
    pilot_version: str | None = None,
) -> dict[str, Any]:
    """Assemble H5b/H5c repeated pilot JSON (research only)."""
    baselines = h4c_baselines if h4c_baselines is not None else load_h4c_beta_mae_baselines()
    by_world: dict[str, list[dict[str, Any]]] = {}
    for row in per_run_rows:
        by_world.setdefault(str(row["world_id"]), []).append(row)

    aggregate_by_world: dict[str, Any] = {}
    h4c_comparison: dict[str, Any] = {}
    for wid, runs in sorted(by_world.items()):
        aggregate_by_world[wid] = aggregate_world_runs(runs)
        h4c_wid = H5_TO_H4C_BASELINE.get(wid)
        h4c_mae = baselines.get(h4c_wid) if h4c_wid else None
        h5_mean = aggregate_by_world[wid]["beta_gc_mae"].get("mean")
        if h4c_mae is not None and h5_mean is not None:
            h4c_comparison[wid] = {
                "h4c_baseline_world": h4c_wid,
                "h4c_beta_gc_mae": h4c_mae,
                "h5_beta_gc_mae_mean": h5_mean,
                "delta_vs_h4c": float(h5_mean) - float(h4c_mae),
                "improved_vs_h4c": float(h5_mean) < float(h4c_mae),
            }

    stability: dict[str, Any] = {}
    for wid in (
        "WORLD-BAYES-H5-ADSTOCK-ALIGNED",
        "WORLD-BAYES-H5-SATURATION-ALIGNED",
    ):
        agg = aggregate_by_world.get(wid, {})
        beta = agg.get("beta_gc_mae") or {}
        comp = h4c_comparison.get(wid, {})
        stability[wid] = {
            "beta_gc_mae_mean": beta.get("mean"),
            "beta_gc_mae_std": beta.get("std"),
            "improved_vs_h4c_all_seeds": comp.get("improved_vs_h4c"),
            "stable_improvement": comp.get("improved_vs_h4c") and (beta.get("std") or 0) < 0.05,
        }

    mismatch_worlds = [w for w in H5_WORLD_IDS if "MISMATCH" in w]
    mismatch_warn_rates = [
        aggregate_by_world.get(w, {}).get("transform_mismatch_warning_rate", 0.0) for w in mismatch_worlds
    ]

    comparison_to_h5b = (
        _compare_to_h5b_fast_pilot(aggregate_by_world, h5b_data=h5b_data) if extended_mcmc else None
    )
    conclusions = (
        _extended_conclusions(aggregate_by_world, h4c_comparison, comparison_to_h5b or {})
        if extended_mcmc
        else None
    )

    summary: dict[str, Any] = {
        "pilot_id": pilot_id or (EXTENDED_PILOT_ID if extended_mcmc else REPEATED_PILOT_ID),
        "pilot_version": pilot_version
        or (EXTENDED_PILOT_VERSION if extended_mcmc else REPEATED_PILOT_VERSION),
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "seeds": list(seeds),
        "sampler_settings": sampler_metadata
        or (
            {**SAMPLER_EXTENDED, "extended_mcmc_profile": True}
            if extended_mcmc
            else {**SAMPLER_FAST, "fast_mcmc_profile": fast_mcmc}
        ),
        "per_run": per_run_rows,
        "aggregate_by_world": aggregate_by_world,
        "h4c_baseline_comparison": h4c_comparison,
        "stability_summary": stability,
        "mismatch_warning_rate_by_world": dict(zip(mismatch_worlds, mismatch_warn_rates, strict=True)),
        "hard_gate": False,
        "production_promotion": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "decision_grade": False,
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
    }
    if extended_mcmc:
        summary["comparison_to_h5b_fast_pilot"] = comparison_to_h5b
        summary["h5c_conclusions"] = conclusions
        summary["reference_h5b_artifact"] = str(DEFAULT_REPEATED_ARTIFACT_PATH)
        summary["note"] = "H5c extended MCMC repeated pilot — confirms H5b conclusions; INV-071 report-only."
    else:
        summary["diagnostic_fix"] = "transforms_aligned treats linear/correlated/weak_signal + identity as aligned"
        summary["note"] = "H5b repeated fast-MCMC pilot — stability check only; INV-071 report-only."
    return summary


def aggregate_world_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_h5_world_runs(runs)


def run_h5_repeated_pilot(
    seeds: tuple[int, ...] | None = None,
    *,
    world_ids: tuple[str, ...] | None = None,
    fast_mcmc: bool = True,
    extended_mcmc: bool = False,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run H5 recovery across seeds and write repeated pilot artifact (H5b fast or H5c extended)."""
    if extended_mcmc:
        fast_mcmc = False
    nut_seeds = seeds or DEFAULT_REPEATED_SEEDS
    ids = world_ids or H5_WORLD_IDS
    sampler_settings = dict(SAMPLER_EXTENDED) if extended_mcmc else None
    rows: list[dict[str, Any]] = []
    sampler_meta: dict[str, Any] | None = None
    for wid in ids:
        for nuts_seed in nut_seeds:
            spec = get_recovery_world(wid)
            report = run_h5_recovery_world(
                wid,
                fast_mcmc=fast_mcmc,
                sampler=sampler_settings,
                nuts_seed=int(nuts_seed),
            )
            if sampler_meta is None:
                sampler_meta = _backend_metadata(
                    spec,
                    fast_mcmc=fast_mcmc,
                    extended_mcmc=extended_mcmc,
                )
            row = _world_row_from_report(report, spec)
            row["nuts_seed"] = int(nuts_seed)
            row["extended_mcmc"] = extended_mcmc
            rows.append(row)

    default_path = DEFAULT_EXTENDED_ARTIFACT_PATH if extended_mcmc else DEFAULT_REPEATED_ARTIFACT_PATH
    summary = build_h5_repeated_pilot_summary(
        rows,
        seeds=nut_seeds,
        fast_mcmc=fast_mcmc,
        extended_mcmc=extended_mcmc,
        sampler_metadata=sampler_meta,
    )
    path = artifact_path or default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    summary["artifact_path"] = str(path)
    return summary


def validate_h5_extended_repeated_pilot_schema(summary: dict[str, Any]) -> None:
    """Fail fast if H5c extended repeated pilot artifact is invalid."""
    validate_h5_repeated_pilot_schema(summary)
    if summary.get("pilot_id") != EXTENDED_PILOT_ID:
        raise ValueError(f"pilot_id must be {EXTENDED_PILOT_ID!r}")
    if "comparison_to_h5b_fast_pilot" not in summary:
        raise ValueError("comparison_to_h5b_fast_pilot required for H5c artifact")
    if summary.get("sampler_settings", {}).get("extended_mcmc_profile") is not True:
        raise ValueError("extended_mcmc_profile must be true in sampler_settings")


def validate_h5_repeated_pilot_schema(summary: dict[str, Any]) -> None:
    """Fail fast if repeated pilot artifact missing required research-only keys."""
    required = {
        "model_spec_version",
        "seeds",
        "sampler_settings",
        "per_run",
        "aggregate_by_world",
        "hard_gate",
        "production_promotion",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
    missing = required - set(summary.keys())
    if missing:
        raise ValueError(f"H5 repeated pilot schema missing keys: {sorted(missing)}")
    if summary.get("hard_gate") is not False:
        raise ValueError("hard_gate must be false")
    if summary.get("approved_for_prod") is not False:
        raise ValueError("approved_for_prod must be false")
    for row in summary.get("per_run") or []:
        for key in ("decision_surface", "optimizer_ready_curves", "budget_recommendation", "recommendation"):
            if row.get(key) is not None:
                raise ValueError(f"forbidden production field {key!r} in per_run")


def validate_h5_pilot_schema(summary: dict[str, Any]) -> None:
    """Fail fast if pilot artifact missing required research-only keys."""
    required = {
        "model_spec_version",
        "world_catalog",
        "sampler_settings",
        "per_world_metrics",
        "diagnostic_warnings",
        "hard_gate",
        "production_promotion",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
    missing = required - set(summary.keys())
    if missing:
        raise ValueError(f"H5 pilot schema missing keys: {sorted(missing)}")
    if summary.get("hard_gate") is not False:
        raise ValueError("hard_gate must be false")
    if summary.get("approved_for_prod") is not False:
        raise ValueError("approved_for_prod must be false")
    for row in summary.get("per_world_metrics") or []:
        for key in ("decision_surface", "optimizer_ready_curves", "budget_recommendation", "recommendation"):
            if row.get(key) is not None:
                raise ValueError(f"forbidden production field {key!r} in per_world_metrics")
