"""Post-fit diagnostics — orchestrates registered extensions only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.artifacts.base import ArtifactStoreBase
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import get_extension_registry
from mmm.evaluation.performance_certification import build_performance_certification_report
from mmm.governance.calibration_readiness import (
    apply_calibration_readiness_to_model_release,
    build_calibration_readiness_report,
)
from mmm.governance.decision_stress import build_decision_stress_report
from mmm.governance.production_readiness import build_production_readiness_report
from mmm.governance.reproducibility_certification import (
    build_reproducibility_certification_report,
    extract_reproducibility_snapshot,
)
from mmm.governance.synthetic_certification import run_synthetic_certification_suite
from mmm.hierarchy.diagnostics import hierarchy_enabled
from mmm.optimization.optimizer_certification import build_optimizer_certification_report
from mmm.optimization.robust import build_robust_optimization_research
from mmm.services.calibration_service import run_hierarchy_post_fit_reports
from mmm.uncertainty.propagation_report import (
    build_legacy_uncertainty_buckets,
    build_uncertainty_propagation_report,
)
from mmm.validation import build_continuous_validation_report, build_decision_validation_report

_REPLAY_DISCLOSURE_KEYS = (
    "calibration_refit_mode",
    "replay_uses_full_panel_refit",
    "replay_overfit_warning",
    "replay_training_units",
    "replay_holdout_units",
    "replay_holdout_available",
    "replay_train_loss",
    "replay_holdout_loss",
    "replay_generalization_gap",
    "replay_generalization_gap_severity",
    "replay_transform_mode",
    "legacy_replay_upgrade_warnings",
    "legacy_replay_warnings",
    "replay_refit_mode",
    "fold_replay_losses",
    "fold_replay_units_used",
    "fold_replay_units_skipped",
    "replay_fold_alignment_warnings",
)


def _enrich_calibration_summary(ctx: ExtensionContext) -> None:
    cal_summary = ctx.out.get("calibration_summary")
    if not isinstance(cal_summary, dict):
        return
    replay_meta = cal_summary.get("replay_meta")
    if isinstance(replay_meta, dict):
        if replay_meta.get("evidence_weighted_replay_summary"):
            ctx.out["evidence_weighted_replay_summary"] = replay_meta["evidence_weighted_replay_summary"]
        for key in _REPLAY_DISCLOSURE_KEYS:
            if key in replay_meta:
                cal_summary[key] = replay_meta[key]
    best_detail = ctx.fit_out.get("best_detail")
    if isinstance(best_detail, dict):
        for key in _REPLAY_DISCLOSURE_KEYS:
            if key in best_detail:
                cal_summary[key] = best_detail[key]
    ctx.out["calibration_summary"] = cal_summary


def _patch_model_release_replay_gap(ctx: ExtensionContext) -> None:
    mr = ctx.out.get("model_release")
    if not isinstance(mr, dict):
        return
    replay_invalidation: list[str] = list(mr.get("invalidation_reasons") or [])
    _bd_gap = ctx.fit_out.get("best_detail")
    if (
        ctx.config.calibration.block_on_severe_replay_gap
        and isinstance(_bd_gap, dict)
        and str(_bd_gap.get("replay_generalization_gap_severity", "")) == "severe"
        and "severe_replay_generalization_gap" not in replay_invalidation
    ):
        replay_invalidation.append("severe_replay_generalization_gap")
        from mmm.governance.model_release import infer_model_release_state

        gov_js = ctx.out.get("governance") or {}
        pq_js = ctx.out.get("panel_qa") or {}
        post_fit_validation = ctx.out.get("post_fit_validation")
        post_fit_validation_js = post_fit_validation if isinstance(post_fit_validation, dict) else None
        operational_health = ctx.out.get("operational_health")
        operational_health_js = operational_health if isinstance(operational_health, dict) else None
        ctx.out["model_release"] = infer_model_release_state(
            config=ctx.config,
            panel_qa_max_severity=str(pq_js.get("max_severity", "info")),
            governance_approved_for_optimization=bool(gov_js.get("approved_for_optimization")),
            governance_approved_for_reporting=bool(gov_js.get("approved_for_reporting")),
            ridge_fit_summary_present=bool(ctx.out.get("ridge_fit_summary")),
            invalidation_reasons=replay_invalidation,
            post_fit_validation=post_fit_validation_js,
            operational_health=operational_health_js,
        )


def _run_research_extension_reports(ctx: ExtensionContext) -> None:
    """Optional research diagnostics not yet registered as plugins."""
    config = ctx.config
    fit_out = ctx.fit_out
    out = ctx.out
    panel_s = ctx.panel_s
    schema = ctx.schema
    rng = ctx.rng

    if hierarchy_enabled(config) and config.framework == Framework.RIDGE_BO and fit_out.get("artifacts"):
        art = fit_out["artifacts"]
        n_media = len(schema.channel_columns)
        coef = np.asarray(art.coef, dtype=float).ravel()[:n_media]
        out.update(run_hierarchy_post_fit_reports(config, panel=panel_s, schema=schema, coef=coef))

    if config.framework == Framework.BAYESIAN:
        exp_lr = fit_out.get("bayesian_experiment_likelihood_report")
        if isinstance(exp_lr, dict):
            out["bayesian_experiment_likelihood_report"] = exp_lr
        hier_lr = fit_out.get("bayesian_hierarchy_report")
        if isinstance(hier_lr, dict):
            out["bayesian_hierarchy_report"] = hier_lr

    propagation_report = build_uncertainty_propagation_report(
        config,
        fit_out=fit_out,
        extension_report=out,
    )
    out["uncertainty_propagation_report"] = propagation_report
    out["robust_optimization_research"] = build_robust_optimization_research(
        config,
        panel=panel_s,
        schema=schema,
        fit_out=fit_out,
        extension_report=out,
        rng=rng,
    )
    if config.extensions.continuous_validation.enabled:
        out["continuous_validation_report"] = build_continuous_validation_report(
            config,
            current_extension_report=out,
        )
    if config.extensions.decision_validation.enabled:
        out["decision_validation_report"] = build_decision_validation_report(config)
    readiness = build_calibration_readiness_report(config, out)
    out["calibration_readiness_report"] = readiness
    patched_mr = apply_calibration_readiness_to_model_release(config, out, readiness)
    if patched_mr is not None:
        out["model_release"] = patched_mr
    if config.extensions.reproducibility_certification.enabled:
        snap = extract_reproducibility_snapshot(fit_out=fit_out, extension_report=out)
        out["reproducibility_certification_report"] = build_reproducibility_certification_report(
            reference=snap,
            reference_run_path=config.extensions.reproducibility_certification.reference_run_path,
        )
    if config.extensions.performance_certification.enabled:
        out["performance_certification_report"] = build_performance_certification_report(config)
    id_json = out.get("identifiability", {})
    out["uncertainty_decomposition_legacy"] = build_legacy_uncertainty_buckets(
        config,
        propagation_report,
        ident=id_json if isinstance(id_json, dict) else {},
    )
    _run_certification_extension_reports(ctx)


def _run_certification_extension_reports(ctx: ExtensionContext) -> None:
    """Production certification rollups (warnings by default; no auto budget actions)."""
    config = ctx.config
    out = ctx.out
    out["synthetic_certification_report"] = run_synthetic_certification_suite()
    opt_report = None
    run_optimizer_cert = (
        config.extensions.optimizer_certification.enabled
        or config.run_environment == RunEnvironment.PROD
    )
    if run_optimizer_cert:
        opt_report = build_optimizer_certification_report(seed=int(config.random_seed))
        out["optimizer_certification_report"] = opt_report
    out["decision_stress_report"] = build_decision_stress_report(
        config, out, panel=ctx.panel_s, schema=ctx.schema
    )
    out["production_readiness_report"] = build_production_readiness_report(
        config,
        out,
        synthetic_certification=out["synthetic_certification_report"],
        optimizer_certification=opt_report,
    )


def run_post_fit_extensions(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    yhat: np.ndarray,
    store: ArtifactStoreBase | None,
) -> dict[str, Any]:
    """Run all registered post-fit extensions in deterministic order."""
    seed_resolution = resolve_seed_contract(config)
    panel_s = sort_panel_for_modeling(panel, schema)
    ctx = ExtensionContext(
        panel=panel,
        panel_s=panel_s,
        schema=schema,
        config=config,
        fit_out=fit_out,
        yhat=yhat,
        store=store,
        out={"seed_resolution": seed_resolution},
        rng=np.random.default_rng(int(config.extension_seed)),
        ext=config.extensions,
        seed_resolution=seed_resolution,
    )
    get_extension_registry().run_all(ctx)
    _enrich_calibration_summary(ctx)
    from mmm.governance.replay_refit_prod_policy import replay_refit_prod_governance_note

    replay_note = replay_refit_prod_governance_note(ctx.config)
    if replay_note:
        ctx.out["replay_refit_prod_governance"] = replay_note
    _patch_model_release_replay_gap(ctx)
    _run_research_extension_reports(ctx)
    _attach_ridge_production_diagnostics(ctx)
    if store:
        store.log_dict("extension_report", ctx.out)
    return ctx.out


def _attach_ridge_production_diagnostics(ctx: ExtensionContext) -> None:
    """H7: attach Ridge decision-safety diagnostics (no optimizer / DecisionSurface changes)."""
    if ctx.config.framework != Framework.RIDGE_BO:
        return
    vertical_id: str | None = None
    product = getattr(ctx.ext, "product", None)
    if isinstance(product, dict):
        vertical_id = product.get("vertical_id")
    rd = getattr(ctx.ext, "ridge_diagnostics", None)
    if rd is not None and getattr(rd, "vertical_id", None):
        vertical_id = rd.vertical_id
    cal_signals_path: str | None = None
    if rd is not None and getattr(rd, "calibration_signals_path", None):
        cal_signals_path = rd.calibration_signals_path
    try:
        from mmm.diagnostics.ridge_diagnostics import attach_ridge_diagnostics_to_extension_report

        trainer = ctx.fit_out.get("_ridge_trainer")
        ctx.out = attach_ridge_diagnostics_to_extension_report(
            ctx.out,
            ctx.panel_s,
            ctx.schema,
            ctx.config,
            ctx.fit_out,
            trainer=trainer,
            vertical_id=vertical_id,
            calibration_evidence_available=bool(
                ctx.config.calibration.enabled and ctx.config.calibration.use_replay_calibration
            ),
            calibration_signals_path=cal_signals_path,
        )
    except Exception:
        ctx.out.setdefault("ridge_production_diagnostics_report", {"status": "unavailable"})
