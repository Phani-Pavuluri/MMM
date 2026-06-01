"""Bayes-H4c extended recovery pilot — reliability map (research only, no promotion)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import (
    _backend_metadata,
    _json_safe,
    _world_row_from_report,
)
from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import H4C_WORLD_IDS
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    SAMPLER_EXTENDED,
    SAMPLER_FAST,
    WORLD_BAYES_H4_SPARSE_GEO,
    get_recovery_world,
)

PILOT_ID = "BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601"
PILOT_VERSION = "bayes_h4c_extended_recovery_pilot_v1"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json")

RECOVERY_MAE_GOOD = 0.30
RECOVERY_MAE_POOR = 0.45


def classify_h4c_world(row: dict[str, Any], spec: Any) -> dict[str, Any]:
    """Assign reliability-map classification from spec + observed metrics (report only)."""
    base = str(spec.expected_diagnostic_behavior.get("h4c_classification", "inconclusive"))
    beta_mae = row.get("beta_gc_mae")
    mu_mae = row.get("mu_c_mae")
    warnings = list(row.get("h4c_diagnostic_warnings") or row.get("conflict_warnings") or [])

    classification = base
    reason = f"designated {base} from world spec"

    if base == "recovery_candidate" and beta_mae is not None:
        if float(beta_mae) <= RECOVERY_MAE_GOOD and (mu_mae is None or float(mu_mae) <= RECOVERY_MAE_GOOD):
            classification = "recovery_candidate"
            reason = "beta_gc_mae and mu_c_mae within report-only good band on this pilot run"
        elif float(beta_mae) > RECOVERY_MAE_POOR:
            classification = "recovery_degraded"
            reason = f"high beta_gc_mae={float(beta_mae):.3f} on recovery candidate world"
        else:
            classification = "recovery_moderate"
            reason = "partial recovery on candidate world; not a pass gate"

    if base == "transform_mismatch" or any("transform_mismatch" in w for w in warnings):
        classification = "transform_mismatch"
        reason = "generative transform differs from MVP semi_log on raw media"

    if base == "weak_identification" or any(
        k in w for w in warnings for k in ("collinearity", "weak_identification")
    ):
        if any("collinearity" in w for w in warnings):
            classification = "weak_identification"
            reason = "channel collinearity or weak signal limits channel-level recovery"
        elif base == "weak_identification":
            classification = "weak_identification"
            reason = "low SNR / identification stress world"

    if any("conflict:" in w for w in warnings):
        classification = "conflict_warning"

    return {
        "classification": classification,
        "design_classification": base,
        "reason": reason,
        "hard_gate": False,
        "production_promotion": False,
    }


def _run_world_row(world_id: str, *, fast_mcmc: bool = True) -> dict[str, Any]:
    spec = get_recovery_world(world_id)
    report = run_h4_recovery_world(world_id, fast_mcmc=fast_mcmc, panel_seed=4400)
    rec = report.get("h4_recovery") or {}
    row = _world_row_from_report(report, spec)
    row["h4c_classification"] = rec.get("h4c_classification")
    row["h4c_diagnostic_warnings"] = list(rec.get("h4c_diagnostic_warnings") or [])
    row["beta_interval_width_90_mean"] = rec.get("beta_interval_width_90_mean")
    row["sparse_shrinkage_decomposition"] = rec.get("sparse_shrinkage_decomposition")
    row["reliability_map"] = classify_h4c_world(row, spec)
    row["approved_for_prod"] = False
    row["prod_decisioning_allowed"] = False
    return row


def build_h4c_pilot_summary(
    world_rows: list[dict[str, Any]],
    *,
    pilot_id: str = PILOT_ID,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    backends = {wid: _backend_metadata(get_recovery_world(wid), fast_mcmc=fast_mcmc) for wid in H4C_WORLD_IDS}
    by_class: dict[str, list[str]] = {}
    for row in world_rows:
        rm = row.get("reliability_map") or {}
        cls = str(rm.get("classification", "unknown"))
        by_class.setdefault(cls, []).append(str(row["world_id"]))

    return _json_safe(
        {
            "pilot_id": pilot_id,
            "pilot_version": PILOT_VERSION,
            "status": "complete",
            "label": "RESEARCH ONLY — NOT DECISION GRADE",
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "outputs_are_diagnostic_only": True,
            "interpretation": {
                "report_only": True,
                "hard_gate": False,
                "production_promotion": False,
                "question": "Where does the Bayes-H3 sandbox recover synthetic truth, and where does it fail?",
                "not_asking": "Is Bayesian MMM ready for production?",
                "pooling_mechanics": "understood (H4b-disposition C+A)",
                "sparse_stress_reference": WORLD_BAYES_H4_SPARSE_GEO,
                "sparse_stress_role": "report-only stress diagnostic — not a global H4c pass/fail gate",
                "true_effect_recovery": "open under INV-071",
                "reliability_map_by_classification": by_class,
            },
            "world_ids": list(H4C_WORLD_IDS),
            "sampler_settings": dict(SAMPLER_FAST if fast_mcmc else SAMPLER_EXTENDED),
            "fast_mcmc_profile": fast_mcmc,
            "backend_defaults": backends,
            "worlds": world_rows,
        }
    )


def run_h4c_extended_recovery_pilot(
    world_ids: tuple[str, ...] | None = None,
    *,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    """Run H4c extended recovery worlds and build reliability-map summary (research only)."""
    ids = world_ids or H4C_WORLD_IDS
    rows = [_run_world_row(wid, fast_mcmc=fast_mcmc) for wid in ids]
    return build_h4c_pilot_summary(rows, fast_mcmc=fast_mcmc)


def write_h4c_extended_recovery_pilot_artifact(
    path: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path or DEFAULT_ARTIFACT_PATH)
    payload = summary if summary is not None else run_h4c_extended_recovery_pilot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_h4c_extended_recovery_pilot_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_ARTIFACT_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bayes-H4c extended recovery pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--fast-mcmc", action="store_true", default=True)
    args = parser.parse_args()
    out = write_h4c_extended_recovery_pilot_artifact(
        args.output,
        run_h4c_extended_recovery_pilot(fast_mcmc=args.fast_mcmc),
    )
    print(json.dumps({"written": str(out), "pilot_id": PILOT_ID}))


if __name__ == "__main__":
    main()
