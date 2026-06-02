"""Bayes-H5n research-only shadow-policy recommender (diagnostics → governed policy)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.data.schema import PanelSchema, validate_panel
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics import (
    DATASET_SNAPSHOT_ID,
    DEFAULT_PANEL_PATH,
    PANEL_ID,
    inspect_collinearity_diagnostics,
    inspect_sparsity_diagnostics,
)
from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    HIERARCHY_FIXED_TAU,
    HIERARCHY_FULL_GEO_CHANNEL,
    HIERARCHY_POOLED_CHANNEL,
    PARAMETERIZATION_NON_CENTERED,
    is_ablation_only_geometry,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_COMPOSITE,
    CHANNEL_POLICY_DROP_COLLINEAR,
    CHANNEL_POLICY_KEEP_ALL,
    CHANNEL_POLICY_POOLED,
    CHANNEL_POLICY_SINGLE,
    compute_media_correlation_matrix,
    detect_collinear_channel_groups,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_policy import load_shadow_policy
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    classify_convergence_status,
    evidence_promotion_allowed,
    research_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED

INVESTIGATION_ID = "INV-H5N_SHADOW_POLICY_RECOMMENDER"
ARTIFACT_ID = "BAYES_H5N_SHADOW_POLICY_RECOMMENDATION_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601"
DEFAULT_OUTPUT = Path(f"docs/05_validation/archives/{ARTIFACT_ID}.json")
DEFAULT_H5M_POLICY = Path("docs/06_investigations/h5m_sample_panel_shadow_policy.json")
DEFAULT_H5J_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json"
)
DEFAULT_H5L_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601.json"
)
DEFAULT_H5M_REPLAY = Path(
    "docs/05_validation/archives/BAYES_H5M_SHADOW_POLICY_REPLAY_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json"
)

STATUS_RECOMMENDED = "recommended"
STATUS_ALLOWED_ALTERNATIVE = "allowed_alternative"
STATUS_BLOCKED = "blocked"
STATUS_REQUIRES_EXTERNAL_CALIBRATION = "requires_external_calibration"
STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
STATUS_DO_NOT_RUN = "do_not_run"

CHANNEL_KEEP_ALL_WEAK_ID = "keep_all_channels_with_weak_id_warning"
CHANNEL_DROP_COLLINEAR = "drop_collinear_channels"
CHANNEL_COMPOSITE = "composite_media_channel"
CHANNEL_SINGLE_DIAGNOSTIC = "single_channel_diagnostic"
CHANNEL_EXTERNAL_CALIBRATION = "external_calibration_required"
CHANNEL_DO_NOT_RUN = "do_not_run_h5_shadow_until_panel_fixed"

GEOM_FULL_NC_SIGMA_FLOOR = "full_hierarchy_non_centered_sigma_floor"
GEOM_FULL_CURRENT_WARNING = "full_hierarchy_current_with_warning"
GEOM_ABLATION_BENCHMARK = "ablation_benchmark_only"
GEOM_DO_NOT_PROMOTE_ABLATION = "do_not_use_ablation_for_promotion"

DEFAULT_CORR_THRESHOLD = 0.95
SPARSE_NEAR_ZERO_THRESHOLD = 0.99


class H5ShadowPolicyRecommenderError(ValueError):
    """Recommender input validation failed — fail closed."""


@dataclass
class ShadowPolicyRecommendationInput:
    panel_id: str
    dataset_snapshot_id: str
    panel_schema: dict[str, Any]
    collinearity_diagnostics: dict[str, Any]
    sparsity_diagnostics: dict[str, Any] | None = None
    panel_diagnostics: dict[str, Any] | None = None
    convergence_experiment_results: list[dict[str, Any]] | None = None
    business_metadata: dict[str, Any] | None = None
    calibration_evidence_available: bool = False
    frozen_policy_reference: dict[str, Any] | None = None
    corr_threshold: float = DEFAULT_CORR_THRESHOLD
    collinear_groups: list[dict[str, Any]] | None = None
    artifact_id: str | None = None


def panel_recommendation_artifact_id(panel_id: str, *, date_suffix: str = "20260601") -> str:
    """Stable archive artifact id for a panel recommendation."""
    slug = panel_id.upper().replace("-", "_")
    return f"BAYES_H5O_SHADOW_POLICY_RECOMMENDATION_{slug}_{date_suffix}"


def recommendation_is_runnable(artifact: dict[str, Any]) -> bool:
    """True when recommender permits a shadow run (not do_not_run)."""
    status = (artifact.get("recommended_shadow_policy") or {}).get("status")
    return status not in (STATUS_DO_NOT_RUN, STATUS_REQUIRES_EXTERNAL_CALIBRATION)


def _require_non_empty_dict(val: Any, name: str) -> dict[str, Any]:
    if not val or not isinstance(val, dict):
        raise H5ShadowPolicyRecommenderError(f"{name} is required")
    return val


def _experiment_index(results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not results:
        return []
    return [r for r in results if isinstance(r, dict)]


def _variant_is_ablation(row: dict[str, Any]) -> bool:
    geom = row.get("geometry_config") or row.get("geometry_diagnostics_from_fit") or {}
    if row.get("ablation_only") is True:
        return True
    if row.get("hierarchy_faithful") is False:
        return True
    hier = geom.get("hierarchy_policy") if isinstance(geom, dict) else None
    if hier in (HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU):
        return True
    if geom and is_ablation_only_geometry(geom):
        return True
    cp = row.get("channel_config") or row.get("channel_policy") or {}
    if cp.get("mode") == CHANNEL_POLICY_POOLED:
        return True
    rec = row.get("channel_policy_record") or {}
    if rec.get("mode") == "pooled_channel_effects":
        return True
    return False


def _faithful_converged_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in results:
        if _variant_is_ablation(row):
            continue
        status = row.get("convergence_status")
        if status == "converged_diagnostic_only":
            out.append(row)
    return out


def _drop_collinear_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in results:
        cp = row.get("channel_config") or row.get("channel_policy") or {}
        rec = row.get("channel_policy_record") or {}
        if cp.get("mode") == CHANNEL_POLICY_DROP_COLLINEAR or rec.get("mode") == CHANNEL_POLICY_DROP_COLLINEAR:
            out.append(row)
    return out


def _keep_all_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in results:
        cp = row.get("channel_config") or row.get("channel_policy") or {}
        if cp.get("mode") == CHANNEL_POLICY_KEEP_ALL:
            out.append(row)
        elif not cp and "BASELINE" in str(row.get("variant_id", "")):
            out.append(row)
    return out


def _composite_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in results:
        cp = row.get("channel_config") or row.get("channel_policy") or {}
        if cp.get("mode") == CHANNEL_POLICY_COMPOSITE:
            out.append(row)
    return out


def _base_forbidden_claims() -> list[str]:
    return [
        "No production Bayes claim",
        "No optimizer or budget recommendation use",
        "No DecisionSurface or budget recommendations",
        "No Ridge replacement",
        "Shadow policy recommendations are not business decisions",
    ]


def _interpretation_for_channel_policy(
    channel_policy: dict[str, Any],
    *,
    collinear_groups: list[dict[str, Any]],
    retained_may_absorb: list[tuple[str, str]] | None = None,
) -> list[str]:
    notes: list[str] = []
    mode = channel_policy.get("mode")
    if mode == CHANNEL_POLICY_DROP_COLLINEAR:
        for ch in channel_policy.get("dropped_channels") or []:
            notes.append(f"Dropped channel {ch!r}: forbid separate effect claim for this channel.")
        for ch in channel_policy.get("kept_channels") or []:
            notes.append(f"Retained channel {ch!r}: partial-pooling geo×channel coefficients only.")
        if retained_may_absorb:
            for a, b in retained_may_absorb:
                notes.append(
                    f"Channels {a!r} and {b!r} were highly correlated; forbid clean isolated "
                    f"effect claim for either without external calibration."
                )
    elif mode == CHANNEL_POLICY_COMPOSITE:
        out_ch = channel_policy.get("output_channel", "composite_media")
        notes.append(
            f"Composite channel {out_ch!r}: interpretation is combined-media-block effect, "
            "not separate source channel effects."
        )
    elif mode == CHANNEL_POLICY_KEEP_ALL:
        notes.append(
            "Keep-all under collinearity or weak ID: forbid channel-level budget or causal decision use."
        )
    elif mode == CHANNEL_POLICY_SINGLE:
        ch = channel_policy.get("channel", "?")
        notes.append(f"Single-channel diagnostic on {ch!r} only — not a multi-channel MMM claim.")
    return notes


def _recommendation_option(
    recommendation_id: str,
    status: str,
    *,
    rationale: str,
    channel_policy: dict[str, Any] | None = None,
    h5_geometry_config: dict[str, Any] | None = None,
    sampler_profile: dict[str, Any] | None = None,
    evidence_promotion_allowed: bool | None = None,
    hierarchy_faithful: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "recommendation_id": recommendation_id,
        "status": status,
        "rationale": rationale,
    }
    if channel_policy is not None:
        row["channel_policy"] = channel_policy
    if h5_geometry_config is not None:
        row["h5_geometry_config"] = h5_geometry_config
    if sampler_profile is not None:
        row["sampler_profile"] = sampler_profile
    if evidence_promotion_allowed is not None:
        row["evidence_promotion_allowed"] = evidence_promotion_allowed
    if hierarchy_faithful is not None:
        row["hierarchy_faithful"] = hierarchy_faithful
    if extra:
        row.update(extra)
    return row


def _default_sampler_profile() -> dict[str, Any]:
    return {"profile": "extended_mcmc", **dict(SAMPLER_EXTENDED)}


def _geometry_sigma_floor() -> dict[str, Any]:
    return {
        "parameterization": PARAMETERIZATION_NON_CENTERED,
        "hierarchy_policy": HIERARCHY_FULL_GEO_CHANNEL,
        "hierarchy_strength_policy": "learned_tau",
        "tau_parameterization": "current",
        "beta_prior_policy": "current_default",
        "sigma_policy": "sigma_floor",
        "sigma_floor": 0.05,
        "likelihood_scale_policy": "prescaled_log_outcome",
    }


def recommend_shadow_policy(inp: ShadowPolicyRecommendationInput) -> dict[str, Any]:
    """
    Convert diagnostics and prior experiment evidence into shadow policy recommendations.

    Does not run PyMC or shadow fits — policy suggestion only.
    """
    if not inp.panel_id or not inp.dataset_snapshot_id:
        raise H5ShadowPolicyRecommenderError("panel_id and dataset_snapshot_id are required")
    col = _require_non_empty_dict(inp.collinearity_diagnostics, "collinearity_diagnostics")
    if inp.panel_schema is None:
        raise H5ShadowPolicyRecommenderError("panel_schema is required")

    max_corr = float(col.get("max_abs_correlation") or col.get("max_abs_off_diagonal") or 0.0)
    threshold = float(inp.corr_threshold)
    high_collinearity = max_corr >= threshold
    experiments = _experiment_index(inp.convergence_experiment_results)
    sparsity = inp.sparsity_diagnostics or {}
    business = inp.business_metadata or {}

    channels = list(inp.panel_schema.get("media_columns") or [])
    collinear_groups = inp.collinear_groups or col.get("collinear_groups") or []

    allowed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    interpretation_changes: list[str] = []
    forbidden_claims = _base_forbidden_claims()

    faithful_converged = _faithful_converged_variants(experiments)
    ablation_converged = [
        r
        for r in experiments
        if _variant_is_ablation(r) and r.get("convergence_status") == "converged_diagnostic_only"
    ]
    drop_variants = _drop_collinear_variants(experiments)
    keep_all_rows = _keep_all_variants(experiments)
    composite_rows = _composite_variants(experiments)

    keep_all_failed = all(
        r.get("convergence_status") not in ("converged_diagnostic_only",) for r in keep_all_rows
    ) if keep_all_rows else high_collinearity

    governed_drop_converged = any(
        not _variant_is_ablation(r) and r.get("convergence_status") == "converged_diagnostic_only"
        for r in drop_variants
    ) or (
        inp.frozen_policy_reference is not None
        and (inp.frozen_policy_reference.get("channel_policy") or {}).get("mode")
        == CHANNEL_POLICY_DROP_COLLINEAR
    )
    only_ablation_converged = bool(ablation_converged) and not faithful_converged

    # --- SPARSITY ---
    sparse_channels: list[str] = []
    for ch, stats in (sparsity.get("by_channel") or {}).items():
        nz = float(stats.get("near_zero_share") or 0.0)
        if nz >= SPARSE_NEAR_ZERO_THRESHOLD:
            sparse_channels.append(ch)
            forbidden_claims.append(
                f"Channel {ch!r} has near-zero variation: block separate coefficient claim."
            )

    if sparse_channels and not high_collinearity:
        allowed.append(
            _recommendation_option(
                CHANNEL_SINGLE_DIAGNOSTIC,
                STATUS_DIAGNOSTIC_ONLY,
                rationale="Sparse channel detected — single-channel diagnostic only if explicitly governed.",
                channel_policy={
                    "mode": CHANNEL_POLICY_SINGLE,
                    "channel": sparse_channels[0],
                    "no_silent_dropping": True,
                },
                evidence_promotion_allowed=False,
                hierarchy_faithful=True,
            )
        )

    # --- COLLINEARITY & CONVERGENCE ---
    primary_channel_policy: dict[str, Any] | None = None
    primary_geometry = _geometry_sigma_floor()
    primary_status = STATUS_DO_NOT_RUN
    primary_id = CHANNEL_DO_NOT_RUN
    rationale = "Insufficient evidence for governed shadow policy."
    evidence_status: dict[str, Any] = {
        "evidence_promotion_allowed": False,
        "hierarchy_faithful": False,
        "convergence_status": None,
        "note": "research_only — not production promotion",
    }

    inseparable = business.get("channels_strategically_inseparable") or business.get(
        "inseparable_channel_groups"
    )
    separation_critical = business.get("channel_separation_business_critical") is True

    if separation_critical and high_collinearity:
        blocked.append(
            _recommendation_option(
                CHANNEL_EXTERNAL_CALIBRATION,
                STATUS_REQUIRES_EXTERNAL_CALIBRATION,
                rationale="Business requires separable channel claims but collinearity prevents identification.",
                evidence_promotion_allowed=False,
            )
        )
        primary_status = STATUS_REQUIRES_EXTERNAL_CALIBRATION
        primary_id = CHANNEL_EXTERNAL_CALIBRATION
        rationale = "Collinearity with business-critical channel separation — external calibration required."
        forbidden_claims.append("Do not run H5 shadow for channel-level business claims without calibration.")

    elif inseparable and high_collinearity:
        comp_policy = {
            "mode": CHANNEL_POLICY_COMPOSITE,
            "source_channels": list(inseparable[0]) if isinstance(inseparable, list) and inseparable else ["social", "tv"],
            "method": "first_principal_component",
            "output_channel": business.get("composite_output_channel", "social_tv_pc1"),
            "remaining_channels": [c for c in channels if c not in (inseparable[0] if inseparable else [])],
            "no_silent_dropping": True,
        }
        allowed.append(
            _recommendation_option(
                CHANNEL_COMPOSITE,
                STATUS_ALLOWED_ALTERNATIVE,
                rationale="Channels marked strategically inseparable — composite media block is allowed.",
                channel_policy=comp_policy,
                h5_geometry_config=primary_geometry,
                evidence_promotion_allowed=False,
                hierarchy_faithful=True,
            )
        )
        interpretation_changes.extend(_interpretation_for_channel_policy(comp_policy, collinear_groups=collinear_groups))

    if not high_collinearity:
        keep_policy = {"mode": CHANNEL_POLICY_KEEP_ALL, "no_silent_dropping": True}
        allowed.append(
            _recommendation_option(
                CHANNEL_KEEP_ALL_WEAK_ID,
                STATUS_RECOMMENDED if not sparse_channels else STATUS_ALLOWED_ALTERNATIVE,
                rationale="Max |ρ| below threshold — keep all channels with routine weak-ID monitoring.",
                channel_policy=keep_policy,
                h5_geometry_config=primary_geometry,
                evidence_promotion_allowed=bool(faithful_converged),
                hierarchy_faithful=True,
            )
        )
        if not separation_critical and not inseparable:
            primary_channel_policy = keep_policy
            primary_status = STATUS_RECOMMENDED
            primary_id = CHANNEL_KEEP_ALL_WEAK_ID
            rationale = f"Collinearity below threshold (max |ρ|={max_corr:.3f} < {threshold})."
            evidence_status["evidence_promotion_allowed"] = evidence_promotion_allowed(
                "converged_diagnostic_only" if faithful_converged else "weak_convergence"
            )
            evidence_status["hierarchy_faithful"] = True

    elif governed_drop_converged or inp.frozen_policy_reference:
        ref = inp.frozen_policy_reference or {}
        ref_cp = ref.get("channel_policy") or {}
        drop_policy = {
            "mode": CHANNEL_POLICY_DROP_COLLINEAR,
            "max_abs_corr_threshold": threshold,
            "dropped_channels": list(ref_cp.get("dropped_channels") or []),
            "kept_channels": list(ref_cp.get("kept_channels") or []),
            "no_silent_dropping": True,
            "reason": ref_cp.get("reason")
            or "Governed explicit drop from prior ablation/frozen policy with hierarchy-faithful convergence.",
        }
        if not drop_policy["dropped_channels"]:
            best_drop = next(
                (r for r in faithful_converged if r in drop_variants),
                faithful_converged[0] if faithful_converged else None,
            )
            rec = (best_drop or {}).get("channel_policy_record") or {}
            drop_policy["dropped_channels"] = [
                d.get("channel") for d in rec.get("dropped_channels") or [] if d.get("channel")
            ]
            drop_policy["kept_channels"] = list(rec.get("kept_channels") or [])

        if not drop_policy["dropped_channels"] and collinear_groups:
            members = collinear_groups[0].get("channels", channels)
            drop_policy["dropped_channels"] = [members[-1]]
            drop_policy["kept_channels"] = [c for c in channels if c not in drop_policy["dropped_channels"]]

        primary_channel_policy = drop_policy
        primary_geometry = dict(ref.get("h5_geometry_config") or _geometry_sigma_floor())
        primary_status = STATUS_RECOMMENDED
        primary_id = CHANNEL_DROP_COLLINEAR
        rationale = (
            f"High collinearity (max |ρ|={max_corr:.3f} ≥ {threshold}); "
            "governed explicit drop converged under hierarchy-faithful geometry (H5m/H5L-B pattern)."
        )
        evidence_status = {
            "evidence_promotion_allowed": True,
            "hierarchy_faithful": True,
            "convergence_status": "converged_diagnostic_only",
            "source_milestone": ref.get("policy_id") or "prior_experiments",
            "note": "research eligibility only — production Bayes blocked",
        }

        allowed.append(
            _recommendation_option(
                CHANNEL_KEEP_ALL_WEAK_ID,
                STATUS_BLOCKED if keep_all_failed else STATUS_ALLOWED_ALTERNATIVE,
                rationale="Keep-all failed or weak under collinearity — not for evidence promotion.",
                channel_policy={"mode": CHANNEL_POLICY_KEEP_ALL, "no_silent_dropping": True},
                evidence_promotion_allowed=False,
                hierarchy_faithful=True,
            )
        )

        for comp in composite_rows:
            allowed.append(
                _recommendation_option(
                    CHANNEL_COMPOSITE,
                    STATUS_ALLOWED_ALTERNATIVE,
                    rationale="Composite PC1 probe available but not selected over governed explicit drop.",
                    channel_policy=comp.get("channel_config") or comp.get("channel_policy"),
                    evidence_promotion_allowed=False,
                    hierarchy_faithful=True,
                )
            )

    elif high_collinearity and not governed_drop_converged and primary_status != STATUS_REQUIRES_EXTERNAL_CALIBRATION:
        primary_status = STATUS_DO_NOT_RUN
        primary_id = CHANNEL_DO_NOT_RUN
        rationale = (
            f"Collinearity present (max |ρ|={max_corr:.3f}) but no hierarchy-faithful governed remedy "
            "has reached converged_diagnostic_only."
        )
        blocked.append(
            _recommendation_option(
                CHANNEL_KEEP_ALL_WEAK_ID,
                STATUS_BLOCKED,
                rationale="Keep-all under high collinearity without convergence — blocked for evidence.",
                evidence_promotion_allowed=False,
            )
        )

    # Ablation benchmarks
    for ab in ablation_converged:
        allowed.append(
            _recommendation_option(
                GEOM_ABLATION_BENCHMARK,
                STATUS_DIAGNOSTIC_ONLY,
                rationale=f"Ablation variant {ab.get('variant_id')} converged — benchmark only, not promotable.",
                h5_geometry_config=ab.get("geometry_config"),
                evidence_promotion_allowed=False,
                hierarchy_faithful=False,
                extra={"variant_id": ab.get("variant_id")},
            )
        )
    blocked.append(
        _recommendation_option(
            GEOM_DO_NOT_PROMOTE_ABLATION,
            STATUS_BLOCKED,
            rationale="Pooled/fixed-τ ablations must not be recommended as promotable shadow policies.",
            evidence_promotion_allowed=False,
            hierarchy_faithful=False,
        )
    )

    if only_ablation_converged:
        evidence_status["evidence_promotion_allowed"] = False
        evidence_status["note"] = "Only ablation configs converged — no promotable faithful policy."
        primary_status = STATUS_DO_NOT_RUN
        primary_id = CHANNEL_DO_NOT_RUN
        rationale = "Only pooled/fixed-τ (or other ablation) variants converged — do not promote."

    if inp.calibration_evidence_available:
        allowed.append(
            _recommendation_option(
                CHANNEL_EXTERNAL_CALIBRATION,
                STATUS_ALLOWED_ALTERNATIVE,
                rationale="External calibration available to support channel identification if required.",
                evidence_promotion_allowed=False,
            )
        )
    elif high_collinearity and primary_id == CHANNEL_DROP_COLLINEAR:
        forbidden_claims.append(
            "No external calibration on pilot — collinearity interpretation caveats apply."
        )

    retained_pairs: list[tuple[str, str]] = []
    if primary_channel_policy and primary_channel_policy.get("mode") == CHANNEL_POLICY_DROP_COLLINEAR:
        kept = primary_channel_policy.get("kept_channels") or []
        dropped = primary_channel_policy.get("dropped_channels") or []
        if "social" in kept and "tv" in dropped:
            retained_pairs.append(("social", "tv"))
        elif "search" in kept and "social" in kept:
            retained_pairs.append(("search", "social"))
        for ch in dropped:
            forbidden_claims.append(f"No separate {ch} channel effect claim after explicit drop.")
        if "social" in kept and "tv" in dropped:
            forbidden_claims.append(
                "No clean isolated social-only effect claim without external calibration "
                "(retained social may absorb shared social-tv movement)."
            )

    if primary_channel_policy:
        interpretation_changes.extend(
            _interpretation_for_channel_policy(
                primary_channel_policy,
                collinear_groups=collinear_groups,
                retained_may_absorb=retained_pairs,
            )
        )

    recommended_policy_candidate: dict[str, Any] = {
        "channel_recommendation_id": primary_id,
        "channel_policy": primary_channel_policy,
        "geometry_recommendation_id": GEOM_FULL_NC_SIGMA_FLOOR,
        "h5_geometry_config": primary_geometry,
        "sampler_profile": _default_sampler_profile(),
        "prescale": {"media_prescale": "zscore_panel", "outcome_prescale": "zscore_log"},
        "status": primary_status,
        "rationale": rationale,
    }
    if inp.frozen_policy_reference and primary_status == STATUS_RECOMMENDED:
        recommended_policy_candidate["aligns_with_frozen_policy_id"] = inp.frozen_policy_reference.get(
            "policy_id"
        )

    artifact_id = inp.artifact_id or ARTIFACT_ID
    return {
        "artifact_id": artifact_id,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": inp.panel_id,
        "dataset_snapshot_id": inp.dataset_snapshot_id,
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "recommender_version": "bayes_h5n_shadow_policy_recommender_v1",
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "collinearity_summary": {
            "max_abs_correlation": max_corr,
            "threshold": threshold,
            "high_collinearity": high_collinearity,
            "collinear_groups": collinear_groups,
        },
        "sparsity_summary": sparsity,
        "recommended_shadow_policy": recommended_policy_candidate,
        "allowed_alternatives": allowed,
        "blocked_options": blocked,
        "interpretation_changes": interpretation_changes,
        "forbidden_claims": sorted(set(forbidden_claims)),
        "evidence_status": evidence_status,
        "calibration_evidence_available": inp.calibration_evidence_available,
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
        "excluded_fields": sorted(
            {
                "optimizer",
                "DecisionSurface",
                "decision_surface",
                "recommendations",
                "budget_recommendation",
            }
        ),
    }


def load_prior_experiment_results(
    *,
    h5j_path: Path = DEFAULT_H5J_ARTIFACT,
    h5l_path: Path = DEFAULT_H5L_ARTIFACT,
    h5m_replay_path: Path = DEFAULT_H5M_REPLAY,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, key in (
        (h5j_path, "ablation_results"),
        (h5l_path, "variants"),
        (h5m_replay_path, None),
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if key:
            rows.extend(data.get(key) or [])
        else:
            rows.append(
                {
                    "variant_id": "H5M-FROZEN-POLICY-REPLAY",
                    "convergence_status": data.get("convergence_status"),
                    "rhat_max": data.get("rhat_max"),
                    "divergence_count": data.get("divergence_count"),
                    "evidence_promotion_allowed": data.get("evidence_promotion_allowed"),
                    "hierarchy_faithful": True,
                    "ablation_only": False,
                    "channel_policy": data.get("channel_policy_declared"),
                    "channel_policy_record": data.get("channel_policy_applied"),
                    "geometry_config": data.get("h5_geometry_config_applied"),
                }
            )
    return rows


def _default_panel_schema() -> dict[str, Any]:
    return {
        "geo_column": "geo_id",
        "week_column": "week_start_date",
        "date_column": "week_start_date",
        "outcome_column": "revenue",
        "target_column": "revenue",
        "media_columns": ["search", "social", "tv"],
        "control_columns": [],
    }


def build_panel_recommendation(
    *,
    panel_path: str | Path,
    panel_id: str,
    dataset_snapshot_id: str,
    panel_schema: dict[str, Any] | None = None,
    frozen_policy_path: str | Path | None = None,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    include_sample_panel_prior_evidence: bool = False,
    calibration_evidence_available: bool = False,
    business_metadata: dict[str, Any] | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Build recommendation artifact for an arbitrary in-repo real panel."""
    from mmm.research.bayes_h3_sandbox.h5_shadow_runner import load_panel_from_path

    path = Path(panel_path)
    if not path.is_file():
        raise H5ShadowPolicyRecommenderError(f"panel not found: {path}")

    frozen: dict[str, Any] | None = None
    if frozen_policy_path and Path(frozen_policy_path).is_file():
        frozen = load_shadow_policy(frozen_policy_path)
        if frozen.get("panel_id") and frozen.get("panel_id") != panel_id:
            frozen = None

    schema_raw = dict(panel_schema or frozen.get("panel_schema") if frozen else _default_panel_schema())
    channels = tuple(schema_raw.get("media_columns") or ["search", "social", "tv"])
    schema = PanelSchema(
        schema_raw.get("geo_column", "geo_id"),
        schema_raw.get("week_column") or schema_raw.get("date_column", "week_start_date"),
        schema_raw.get("target_column") or schema_raw.get("outcome_column", "revenue"),
        channels,
        tuple(schema_raw.get("control_columns") or ()),
    )
    df = load_panel_from_path(path)
    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)

    col_diag = inspect_collinearity_diagnostics(df, schema)
    col_diag["collinear_groups"] = detect_collinear_channel_groups(
        df, channels, max_abs_corr_threshold=corr_threshold
    )
    col_diag["media_correlation_matrix"] = compute_media_correlation_matrix(df, channels)

    experiments: list[dict[str, Any]] = []
    if include_sample_panel_prior_evidence:
        experiments = load_prior_experiment_results()

    inp = ShadowPolicyRecommendationInput(
        panel_id=panel_id,
        dataset_snapshot_id=dataset_snapshot_id,
        panel_schema=schema_raw,
        collinearity_diagnostics=col_diag,
        sparsity_diagnostics=inspect_sparsity_diagnostics(df, schema),
        panel_diagnostics={"n_rows": len(df), "n_geos": int(df[schema.geo_column].nunique())},
        convergence_experiment_results=experiments,
        calibration_evidence_available=calibration_evidence_available,
        frozen_policy_reference=frozen,
        corr_threshold=corr_threshold,
        collinear_groups=col_diag.get("collinear_groups"),
        business_metadata=business_metadata,
        artifact_id=artifact_id or panel_recommendation_artifact_id(panel_id),
    )
    artifact = recommend_shadow_policy(inp)
    artifact["source_evidence"] = {"panel_path": str(path)}
    if include_sample_panel_prior_evidence:
        artifact["source_evidence"].update(
            {
                "h5j_artifact": str(DEFAULT_H5J_ARTIFACT),
                "h5l_artifact": str(DEFAULT_H5L_ARTIFACT),
                "h5m_replay_artifact": str(DEFAULT_H5M_REPLAY),
            }
        )
    if frozen:
        artifact["recommended_frozen_policy_id"] = frozen.get("policy_id")
        artifact["source_evidence"]["frozen_policy"] = str(frozen_policy_path)
    return artifact


def build_sample_panel_recommendation(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    frozen_policy_path: str | Path = DEFAULT_H5M_POLICY,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> dict[str, Any]:
    """Build recommendation artifact for the examples sample panel (H5n default)."""
    artifact = build_panel_recommendation(
        panel_path=panel_path,
        panel_id=PANEL_ID,
        dataset_snapshot_id=DATASET_SNAPSHOT_ID,
        frozen_policy_path=frozen_policy_path,
        corr_threshold=corr_threshold,
        include_sample_panel_prior_evidence=True,
        artifact_id=ARTIFACT_ID,
    )
    return artifact


def validate_recommendation_artifact(artifact: dict[str, Any]) -> None:
    for key, val in research_production_flags().items():
        if artifact.get(key) is not val:
            raise H5ShadowPolicyRecommenderError(f"artifact.{key} must be {val!r}")
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        if artifact.get(forbidden) is not None:
            raise H5ShadowPolicyRecommenderError(f"forbidden field on artifact: {forbidden!r}")
    if not artifact.get("forbidden_claims"):
        raise H5ShadowPolicyRecommenderError("forbidden_claims required")
    rec = artifact.get("recommended_shadow_policy") or {}
    if rec.get("status") == STATUS_RECOMMENDED:
        cp = rec.get("channel_policy") or {}
        if cp.get("mode") == CHANNEL_POLICY_DROP_COLLINEAR:
            if not cp.get("dropped_channels") or not cp.get("kept_channels"):
                raise H5ShadowPolicyRecommenderError("explicit drop/keep channels required")
    for alt in artifact.get("allowed_alternatives") or []:
        if alt.get("status") == STATUS_RECOMMENDED and _variant_is_ablation(alt):
            raise H5ShadowPolicyRecommenderError("ablation cannot be status=recommended")
    promo = artifact.get("evidence_status", {}).get("evidence_promotion_allowed")
    if promo and artifact.get("recommended_shadow_policy", {}).get("status") == STATUS_DO_NOT_RUN:
        raise H5ShadowPolicyRecommenderError("do_not_run cannot have evidence_promotion_allowed")


def write_panel_recommendation_artifact(
    *,
    panel_path: str | Path,
    panel_id: str,
    dataset_snapshot_id: str,
    output_path: str | Path,
    panel_schema: dict[str, Any] | None = None,
    frozen_policy_path: str | Path | None = None,
    include_sample_panel_prior_evidence: bool = False,
) -> dict[str, Any]:
    artifact = build_panel_recommendation(
        panel_path=panel_path,
        panel_id=panel_id,
        dataset_snapshot_id=dataset_snapshot_id,
        panel_schema=panel_schema,
        frozen_policy_path=frozen_policy_path,
        include_sample_panel_prior_evidence=include_sample_panel_prior_evidence,
    )
    validate_recommendation_artifact(artifact)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    artifact["artifact_path"] = str(out)
    return artifact


def write_sample_panel_recommendation_artifact(
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact = build_sample_panel_recommendation()
    validate_recommendation_artifact(artifact)
    out = Path(output_path or DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    artifact["artifact_path"] = str(out)
    return artifact


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayes-H5n shadow-policy recommender (research only)")
    p.add_argument("--panel-path", type=str, default=None)
    p.add_argument("--panel-id", type=str, default=None)
    p.add_argument("--dataset-snapshot-id", type=str, default=None)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument(
        "--include-sample-panel-prior-evidence",
        action="store_true",
        help="Attach H5j/H5l/H5m experiment rows (sample panel only)",
    )
    p.add_argument(
        "--frozen-policy-path",
        type=str,
        default=None,
        help="Optional frozen policy when panel_id matches",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if args.panel_path and args.panel_id and args.dataset_snapshot_id:
        out = args.output_path or (
            f"docs/05_validation/archives/"
            f"{panel_recommendation_artifact_id(args.panel_id)}.json"
        )
        artifact = write_panel_recommendation_artifact(
            panel_path=args.panel_path,
            panel_id=args.panel_id,
            dataset_snapshot_id=args.dataset_snapshot_id,
            output_path=out,
            frozen_policy_path=args.frozen_policy_path,
            include_sample_panel_prior_evidence=args.include_sample_panel_prior_evidence,
        )
    else:
        artifact = write_sample_panel_recommendation_artifact(
            output_path=args.output_path,
        )
    print(artifact["artifact_path"])
    if not recommendation_is_runnable(artifact):
        print(
            "STOP: recommended_shadow_policy status="
            f"{artifact['recommended_shadow_policy']['status']!r} — do not force shadow run",
            file=__import__('sys').stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
