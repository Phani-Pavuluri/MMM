"""Phase 1 experiment evidence diagnostic reports (opt-in; no replay execution)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.experiments.compatibility import ExperimentCompatibilityResolver, ModelPanelContext
from mmm.experiments.evidence import ExperimentEvidence
from mmm.experiments.evidence_quality import EvidenceQualityContext, score_evidence_quality
from mmm.experiments.evidence_registry import ExperimentEvidenceRegistry, registry_from_dict
from mmm.experiments.shock_plan import CounterfactualShockPlanner


def _evidence_extensions_enabled(config: MMMConfig) -> bool:
    cal = config.calibration
    return bool(
        cal.evidence_registry_path
        or cal.compatibility_resolver_enabled
        or cal.evidence_weighting_enabled
        or cal.replay_mode == "evidence_registry"
    )


def load_evidence_from_path(path: str | Path) -> list[ExperimentEvidence]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [ExperimentEvidence.model_validate(x) for x in raw]
    if isinstance(raw, dict) and "experiments" in raw:
        return registry_from_dict(raw).list_all()
    raise ValueError(f"unsupported evidence JSON shape at {p}")


def _panel_context(panel: pd.DataFrame, schema: PanelSchema, config: MMMConfig) -> ModelPanelContext:
    from mmm.calibration.evidence_replay import _panel_context as _ctx

    return _ctx(panel, schema, config)


def build_experiment_evidence_reports(
    config: MMMConfig,
    *,
    panel: pd.DataFrame | None = None,
    schema: PanelSchema | None = None,
) -> dict[str, Any]:
    """
    Build diagnostic-only experiment evidence artifacts.

    Does not run experiments, auto-calibrate, or change default replay behavior.
    """
    if not _evidence_extensions_enabled(config):
        return {"skipped": True, "reason": "evidence_extensions_not_enabled"}

    path = config.calibration.evidence_registry_path
    if not path:
        return {"skipped": True, "reason": "evidence_registry_path_not_set"}

    evidence_list = load_evidence_from_path(path)
    reg = ExperimentEvidenceRegistry()
    for ev in evidence_list:
        try:
            reg.register(ev, allow_duplicate=True)
        except ValueError as exc:
            return {"skipped": True, "reason": "registry_validation_failed", "error": str(exc)}

    resolver = ExperimentCompatibilityResolver()
    planner = CounterfactualShockPlanner()
    ctx_panel = _panel_context(panel, schema, config) if panel is not None and schema is not None else None
    target_kpi = config.calibration.experiment_target_kpi or (
        schema.target_column if schema else None
    )

    compatibility_rows: list[dict[str, Any]] = []
    weighting_rows: list[dict[str, Any]] = []
    shock_plans: list[dict[str, Any]] = []
    rejected: list[str] = []
    diagnostic_only: list[str] = []

    for ev in evidence_list:
        compat_dec = None
        if ctx_panel is not None:
            compat_dec = resolver.resolve(ev, ctx_panel, target_kpi=target_kpi)
            compatibility_rows.append(
                {
                    "experiment_id": ev.experiment_id,
                    **compat_dec.to_json(),
                }
            )
            if compat_dec.compatibility_status.value == "rejected":
                rejected.append(ev.experiment_id)
            elif not compat_dec.supports_model_level_calibration:
                diagnostic_only.append(ev.experiment_id)

        qctx = EvidenceQualityContext(
            target_kpi=target_kpi,
            channel_match=ev.channel in (schema.channel_columns if schema else (ev.channel,)),
            compatibility=compat_dec,
            allow_missing_se=config.run_environment.value != "prod",
        )
        qscore = score_evidence_quality(ev, qctx)
        weighting_rows.append({"experiment_id": ev.experiment_id, **qscore.to_json()})

        if compat_dec is not None:
            plan = planner.plan(ev, compat_dec)
            shock_plans.append({"experiment_id": ev.experiment_id, **plan.to_json()})

    coverage = reg.coverage().to_json()
    return {
        "experiment_compatibility_report": {
            "n_experiments": len(evidence_list),
            "decisions": compatibility_rows,
            "rejected_experiments": rejected,
            "diagnostic_only_experiments": diagnostic_only,
            "policy_note": "Compatibility is diagnostic until evidence_registry replay path is validated in prod.",
        },
        "evidence_weighting_report": {
            "n_experiments": len(evidence_list),
            "scores": weighting_rows,
            "evidence_weighting_enabled": config.calibration.evidence_weighting_enabled,
        },
        "counterfactual_shock_plan": {
            "plans": shock_plans,
            "guardrail": "Allocated shocks are computational bridges only; not experimental subgeo truth.",
        },
        "experiment_evidence_registry_coverage": coverage,
        "governance_unsupported_claims": [
            "subgeo_lift_claims_from_aggregate_experiments",
            "causal_claims_from_observational_mmm",
            "experiment_level_claims_from_allocated_shocks",
            "production_monetary_cis_without_coverage_validation",
        ],
    }
