"""Post-fit handlers: drift, control governance, curve alignment, model card."""

from __future__ import annotations

from typing import Any

from mmm.artifacts.compatibility import MODEL_CARD_FILENAME
from mmm.artifacts.factory import artifact_backend_disclosure
from mmm.config.schema import RunEnvironment
from mmm.data.fingerprint import fingerprint_panel
from mmm.evaluation.curve_decision_alignment import (
    evaluate_alignment_scenarios,
    evaluate_curve_decision_alignment,
)
from mmm.evaluation.drift_history import load_historical_reference
from mmm.evaluation.drift_monitor import build_drift_report
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.run_registry import AcceptedRunRegistry
from mmm.governance.control_diagnostics import build_control_governance_report
from mmm.reporting.model_card import generate_model_card


def _run_fingerprint_and_drift(ctx: ExtensionContext) -> None:
    if not (
        ctx.config.artifacts.write_data_fingerprint
        or ctx.config.run_environment == RunEnvironment.PROD
    ):
        return
    ctx.fingerprint = fingerprint_panel(
        ctx.panel_s,
        ctx.schema,
        config=ctx.config,
        seed_resolution=ctx.seed_resolution,
    )
    ctx.out["data_fingerprint"] = ctx.fingerprint
    ctx.out["artifact_backend"] = artifact_backend_disclosure(ctx.config)
    dh = ctx.config.extensions.drift_historical
    historical = None
    if dh.use_registry and dh.registry_dir:
        historical = AcceptedRunRegistry(dh.registry_dir).historical_reference_for_drift(
            exclude_run_id=ctx.config.run_id,
            exclude_run_dir=str(ctx.store.run_path) if ctx.store else None,
        )
    elif dh.prior_run_dir:
        historical = load_historical_reference(dh.prior_run_dir)
    post = ctx.out.get("post_fit_validation") or {}
    model_outputs: dict[str, Any] = {}
    if isinstance(post, dict) and post.get("in_sample_rmse") is not None:
        model_outputs["in_sample_rmse"] = post.get("in_sample_rmse")
    ident = ctx.out.get("identifiability") or {}
    if isinstance(ident, dict) and ident.get("mean_cv_score") is not None:
        model_outputs["mean_cv_score"] = ident.get("mean_cv_score")
    ctx.out["drift_report"] = build_drift_report(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        reference_fingerprint=ctx.fingerprint if historical is None else None,
        reference_panel=ctx.panel_s if historical is None else None,
        historical_reference=historical,
        current_model_outputs=model_outputs or None,
        calibration_summary=ctx.out.get("calibration_summary"),
        seed_resolution=ctx.seed_resolution,
        current_run_id=ctx.config.run_id,
    )


def _run_control_governance(ctx: ExtensionContext) -> None:
    ctx.out["control_governance"] = build_control_governance_report(
        config=ctx.config,
        schema=ctx.schema,
        panel=ctx.panel_s,
    )


def _run_curve_decision_alignment(ctx: ExtensionContext) -> None:
    if not ctx.curve_bundle.get("curve_bundles"):
        return
    bundles = ctx.curve_bundle["curve_bundles"]
    ctx.out["curve_decision_alignment"] = evaluate_curve_decision_alignment(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
        curve_bundles=bundles,
    )
    ch_cols = list(ctx.schema.channel_columns)
    scenarios = [{"label": "default", "spend_delta_frac": 0.05}]
    if len(ch_cols) > 1:
        scenarios.append({"label": "second_channel", "channel": ch_cols[1], "spend_delta_frac": 0.08})
    scenarios.append({"label": "larger_bump", "spend_delta_frac": 0.12})
    ctx.out["curve_decision_alignment_matrix"] = evaluate_alignment_scenarios(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
        curve_bundles=bundles,
        scenarios=scenarios,
    )


def _run_model_card(ctx: ExtensionContext) -> None:
    ctx.out["model_card_md"] = generate_model_card(
        extension_report=ctx.out,
        decision_bundle=ctx.out.get("decision_bundle"),
    )
    if ctx.store:
        card_path = ctx.store.run_path / MODEL_CARD_FILENAME
        card_path.write_text(ctx.out["model_card_md"], encoding="utf-8")
        ctx.store.log_artifact("model_card", card_path)
