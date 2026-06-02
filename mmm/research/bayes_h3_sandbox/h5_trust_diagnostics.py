"""Bayes-H5 research-only TrustReport diagnostic mapping (not production TrustReport)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import H5_WORLD_IDS, get_h5_validation_world

MAPPING_VERSION = "bayes_h5d_trust_diagnostic_mapping_v1"
MAPPING_ID = "BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601"
DEFAULT_SOURCE_ARTIFACT = Path("docs/05_validation/archives/BAYES_H5C_EXTENDED_REPEATED_PILOT_20260601.json")
DEFAULT_OUTPUT_ARTIFACT = Path("docs/05_validation/archives/BAYES_H5D_TRUST_DIAGNOSTIC_MAPPING_20260601.json")

# Stable warning taxonomy (research-only; not production gate codes).
WARNING_TAXONOMY: tuple[str, ...] = (
    "h5:transform_mismatch:adstock",
    "h5:transform_mismatch:saturation",
    "h5:transform_assumption:identity",
    "h5:transform_assumption:adstock",
    "h5:transform_assumption:saturation",
    "h5:transform_unknown:real_panel",
    "h5:convergence:failed",
    "h5:convergence:weak",
    "h5:evidence:blocked",
    "h5:weak_identification:collinearity",
    "h5:weak_identification:weak_signal_generative",
    "h5:sparse_recovery:report_only",
    "h5:recovery_candidate:stable_research_only",
    "h5:production:block",
)

CONVERGENCE_RHAT_CONVERGED_MAX = 1.05
CONVERGENCE_RHAT_WEAK_MAX = 1.10
CONVERGENCE_WEAK_DIVERGENCE_MAX = 5

TRANSFORM_ASSUMPTION_CODE_BY_ID: dict[str, str] = {
    "identity": "h5:transform_assumption:identity",
    "geometric_adstock": "h5:transform_assumption:adstock",
    "adstock": "h5:transform_assumption:adstock",
    "hill_saturation": "h5:transform_assumption:saturation",
    "saturation": "h5:transform_assumption:saturation",
    "adstock_then_saturation": "h5:transform_assumption:adstock",
}

PANEL_CONTEXT_REAL = "real_panel"
PANEL_CONTEXT_SYNTHETIC_FIXTURE = "synthetic_fixture"
PANEL_CONTEXT_SYNTHETIC_WORLD = "synthetic_world"

KNOWN_WORLD_ROLES: frozenset[str] = frozenset(
    {
        "recovery_candidate",
        "transform_mismatch",
        "weak_identification",
    }
)

TRUST_REPORT_CANDIDATE_FIELDS: tuple[str, ...] = (
    "model_spec_version",
    "world_id",
    "transform_alignment_status",
    "transform_mismatch_detected",
    "weak_identification_status",
    "sparse_recovery_status",
    "beta_gc_mae_mean",
    "mu_c_mae_mean",
    "beta_gc_coverage_90_mean",
    "shrinkage_ratio_sparse_mean",
    "warning_codes",
    "h5_classification",
    "policy_outcome",
    "convergence_rhat_max",
    "convergence_ess_bulk_min",
)

FIELDS_EXCLUDED_FROM_PRODUCTION: tuple[str, ...] = (
    "decision_surface",
    "optimizer_ready_curves",
    "budget_recommendation",
    "recommendation",
    "production_decision_surface",
    "production_recommendation",
    "approved_for_prod",
    "prod_decisioning_allowed",
    "hard_gate",
    "production_promotion",
    "decision_grade",
    "idata",
    "linear_coef_draws",
)


class H5TrustDiagnosticMappingError(ValueError):
    """H5 trust diagnostic mapping failed — fail closed."""


def research_production_flags() -> dict[str, bool]:
    """Production flags for H5d mapping artifacts (always false)."""
    return {
        "hard_gate": False,
        "production_promotion": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "research_only": True,
        "decision_grade": False,
        "production_decision_surface": False,
    }


def classify_convergence_status(
    *,
    rhat_max: float | None,
    divergence_count: int | None,
) -> str:
    """
    Report-only convergence class for shadow runs (not a production gate).

    - converged_diagnostic_only: rhat_max <= 1.05 and divergence_count == 0
    - weak_convergence: rhat_max <= 1.10 and divergence_count <= 5
    - failed_convergence: otherwise
    """
    rhat = float(rhat_max) if rhat_max is not None else float("inf")
    div = int(divergence_count) if divergence_count is not None else 0
    if rhat <= CONVERGENCE_RHAT_CONVERGED_MAX and div == 0:
        return "converged_diagnostic_only"
    if rhat <= CONVERGENCE_RHAT_WEAK_MAX and div <= CONVERGENCE_WEAK_DIVERGENCE_MAX:
        return "weak_convergence"
    return "failed_convergence"


def evidence_promotion_allowed(convergence_status: str) -> bool:
    """Shadow evidence may not be promoted unless diagnostic convergence is clean."""
    return convergence_status == "converged_diagnostic_only"


def derive_real_panel_transform_warning_codes(
    transform_config: dict[str, Any],
    *,
    h5_diag: dict[str, Any] | None = None,
) -> list[str]:
    """Real panels: declared transform assumptions + unknown generative truth (no synthetic mismatch)."""
    codes: list[str] = []
    mismatch_mode = str(transform_config.get("transform_mismatch_mode", "aligned"))
    by_channel = transform_config.get("media_transforms_by_channel") or {}

    if mismatch_mode == "intentional_mismatch":
        h5_diag = h5_diag or {}
        if h5_diag.get("transform_mismatch_detected"):
            gen = str(h5_diag.get("generative_transform_expected", ""))
            if "adstock" in gen:
                codes.append("h5:transform_mismatch:adstock")
            if "saturation" in gen or "hill" in gen:
                codes.append("h5:transform_mismatch:saturation")
    else:
        for transform_id in set(by_channel.values()):
            code = TRANSFORM_ASSUMPTION_CODE_BY_ID.get(str(transform_id))
            if code:
                codes.append(code)
        codes.append("h5:transform_unknown:real_panel")

    codes.append("h5:production:block")
    return sorted(set(codes))


def derive_synthetic_transform_warning_codes(
    transform_config: dict[str, Any],
    h5_diag: dict[str, Any],
) -> list[str]:
    """Synthetic worlds / fixture: generative truth known — mismatch warnings when detected."""
    mismatch_mode = str(transform_config.get("transform_mismatch_mode", "aligned"))
    mismatch_detected = bool(h5_diag.get("transform_mismatch_detected"))
    codes: list[str] = []

    if mismatch_detected and mismatch_mode == "intentional_mismatch":
        for ch_tid in transform_config.get("media_transforms_by_channel", {}).values():
            if ch_tid == "identity":
                gen = str(h5_diag.get("generative_transform_expected", ""))
                if "adstock" in gen:
                    codes.append("h5:transform_mismatch:adstock")
                if "saturation" in gen or "hill" in gen:
                    codes.append("h5:transform_mismatch:saturation")
                break
    elif mismatch_detected:
        codes.append("h5:transform_mismatch:adstock")

    if not mismatch_detected and mismatch_mode == "aligned":
        codes.append("h5:recovery_candidate:stable_research_only")

    codes.append("h5:production:block")
    return sorted(set(codes))


def derive_convergence_warning_codes(convergence_status: str) -> list[str]:
    codes: list[str] = []
    if convergence_status == "failed_convergence":
        codes.extend(["h5:convergence:failed", "h5:evidence:blocked"])
    elif convergence_status == "weak_convergence":
        codes.extend(["h5:convergence:weak", "h5:evidence:blocked"])
    return codes


def build_shadow_trust_diagnostics(
    artifact: dict[str, Any],
    transform_config: dict[str, Any],
    *,
    panel_context: str,
    convergence_status: str | None = None,
) -> dict[str, Any]:
    """TrustReport candidate block for H5 shadow runs (real panel vs synthetic)."""
    from mmm.research.bayes_h3_sandbox.labels import RESEARCH_ONLY_LABEL

    h5_diag = artifact.get("h5_transform_diagnostics") or {}
    conv = artifact.get("convergence_diagnostics") or {}
    post = artifact.get("posterior_summary") or {}

    if convergence_status is None:
        convergence_status = classify_convergence_status(
            rhat_max=conv.get("rhat_max"),
            divergence_count=conv.get("divergence_count"),
        )

    if panel_context == PANEL_CONTEXT_REAL:
        codes = derive_real_panel_transform_warning_codes(transform_config, h5_diag=h5_diag)
        alignment = (
            "intentional_mismatch"
            if transform_config.get("transform_mismatch_mode") == "intentional_mismatch"
            else "assumption_only"
        )
        mismatch_detected = bool(h5_diag.get("transform_mismatch_detected"))
    else:
        codes = derive_synthetic_transform_warning_codes(transform_config, h5_diag)
        mismatch_detected = bool(h5_diag.get("transform_mismatch_detected"))
        alignment = "intentional_mismatch" if mismatch_detected else "aligned"

    codes = sorted(set(codes + derive_convergence_warning_codes(convergence_status)))

    return {
        "mapping_version": MAPPING_VERSION,
        "warning_codes": codes,
        "trust_report_candidate_fields": {
            "transform_alignment_status": alignment,
            "transform_mismatch_detected": mismatch_detected,
            "panel_context": panel_context,
            "convergence_status": convergence_status,
            "evidence_promotion_allowed": evidence_promotion_allowed(convergence_status),
            "beta_gc_mae_mean": None,
            "mu_c_mae_mean": None,
            "mu_channel_mean": dict(post.get("mu_channel_mean") or {}),
            "convergence_rhat_max": conv.get("rhat_max"),
            "convergence_ess_bulk_min": conv.get("ess_bulk_min"),
            "convergence_divergence_count": conv.get("divergence_count"),
        },
        "production_trust_report": None,
        "recommended_interpretation": (
            f"{RESEARCH_ONLY_LABEL} Shadow-run diagnostic mapping only; not production TrustReport."
        ),
    }


def build_sampler_diagnostics(
    fit_artifact: dict[str, Any] | None,
    config: Any,
    *,
    sampler_profile: str,
) -> dict[str, Any]:
    """Sampler report block for real-panel shadow artifacts."""
    conv = (fit_artifact or {}).get("convergence_diagnostics") or {}
    bayesian = getattr(config, "bayesian", None)
    draws = tune = chains = target_accept = None
    if bayesian is not None:
        draws = getattr(bayesian, "draws", None)
        tune = getattr(bayesian, "tune", None)
        chains = getattr(bayesian, "chains", None)
        target_accept = getattr(bayesian, "target_accept", None)

    rhat_max = conv.get("rhat_max")
    divergence_count = conv.get("divergence_count")
    status = classify_convergence_status(
        rhat_max=rhat_max,
        divergence_count=divergence_count,
    )
    return {
        "sampler_profile": sampler_profile,
        "draws": draws if draws is not None else conv.get("draws_per_chain"),
        "tune": tune,
        "chains": chains if chains is not None else conv.get("chains"),
        "target_accept": target_accept,
        "rhat_max": rhat_max,
        "ess_min": conv.get("ess_bulk_min"),
        "divergence_count": divergence_count,
        "convergence_status": status,
        "evidence_promotion_allowed": evidence_promotion_allowed(status),
        "report_only": True,
    }


def resolve_world_role(world_id: str, *, pilot_row: dict[str, Any] | None = None) -> str:
    """Resolve canonical world role; fail closed on unknown worlds/roles."""
    if world_id not in H5_WORLD_IDS:
        raise H5TrustDiagnosticMappingError(f"unknown H5 world_id: {world_id!r}")
    spec = get_h5_validation_world(world_id)
    exp = spec.expected_diagnostic_behavior
    role = str((pilot_row or {}).get("role") or exp.get("role") or exp.get("h5_classification") or "")
    if role not in KNOWN_WORLD_ROLES:
        raise H5TrustDiagnosticMappingError(
            f"unknown world role for {world_id!r}: {role!r}; known: {sorted(KNOWN_WORLD_ROLES)}"
        )
    return role


def transform_alignment_status(
    world_id: str,
    *,
    aggregate: dict[str, Any],
) -> str:
    """aligned | intentional_mismatch | unknown (fail closed)."""
    spec = get_h5_validation_world(world_id)
    mode = str(spec.expected_diagnostic_behavior.get("transform_mismatch_mode", "aligned"))
    if mode == "intentional_mismatch":
        return "intentional_mismatch"
    if mode == "aligned":
        if float(aggregate.get("unexpected_mismatch_warning_rate") or 0.0) > 0.0:
            return "unknown"
        return "aligned"
    return "unknown"


def weak_identification_status(world_id: str, *, aggregate: dict[str, Any]) -> str:
    """none | collinearity | weak_signal | unknown."""
    if world_id == "WORLD-BAYES-H5-CORRELATED-CHANNELS":
        if float(aggregate.get("collinearity_warning_rate") or 0.0) >= 0.99:
            return "collinearity"
        return "unknown"
    if world_id == "WORLD-BAYES-H5-WEAK-SIGNAL":
        if float(aggregate.get("weak_identification_warning_rate") or 0.0) >= 0.99:
            return "weak_signal"
        return "unknown"
    if float(aggregate.get("collinearity_warning_rate") or 0.0) > 0.0:
        return "collinearity"
    if float(aggregate.get("weak_identification_warning_rate") or 0.0) > 0.0:
        return "weak_signal"
    return "none"


def sparse_recovery_status(world_id: str) -> str:
    if world_id == "WORLD-BAYES-H5-SPARSE-RECOVERY":
        return "report_only"
    return "not_applicable"


def derive_warning_codes(
    world_id: str,
    *,
    aggregate: dict[str, Any],
    role: str,
) -> list[str]:
    """Deterministic warning codes from world design + aggregate rates."""
    codes: list[str] = []
    tm_rate = float(aggregate.get("transform_mismatch_warning_rate") or 0.0)

    if world_id == "WORLD-BAYES-H5-ADSTOCK-MISMATCH" and tm_rate >= 0.99:
        codes.append("h5:transform_mismatch:adstock")
    if world_id == "WORLD-BAYES-H5-SATURATION-MISMATCH" and tm_rate >= 0.99:
        codes.append("h5:transform_mismatch:saturation")

    if world_id == "WORLD-BAYES-H5-CORRELATED-CHANNELS" and float(
        aggregate.get("collinearity_warning_rate") or 0.0
    ) >= 0.99:
        codes.append("h5:weak_identification:collinearity")

    if world_id == "WORLD-BAYES-H5-WEAK-SIGNAL" and float(
        aggregate.get("weak_identification_warning_rate") or 0.0
    ) >= 0.99:
        codes.append("h5:weak_identification:weak_signal_generative")

    if world_id == "WORLD-BAYES-H5-SPARSE-RECOVERY":
        codes.append("h5:sparse_recovery:report_only")
    elif role == "recovery_candidate":
        codes.append("h5:recovery_candidate:stable_research_only")

    codes.append("h5:production:block")
    return sorted(set(codes))


def recommended_interpretation(
    world_id: str,
    *,
    role: str,
    alignment: str,
    warning_codes: list[str],
    h4c_comparison: dict[str, Any] | None,
) -> str:
    parts: list[str] = ["RESEARCH ONLY — NOT DECISION GRADE."]
    if "h5:transform_mismatch:adstock" in warning_codes or "h5:transform_mismatch:saturation" in warning_codes:
        parts.append("Transform mismatch probe: diagnostic warn only; not a production fail.")
    if "h5:weak_identification:collinearity" in warning_codes:
        parts.append("Collinear channels: weak identification expected.")
    if "h5:weak_identification:weak_signal_generative" in warning_codes:
        parts.append("Weak-signal generative: poor recovery expected.")
    if "h5:sparse_recovery:report_only" in warning_codes:
        parts.append("Sparse recovery: report-only shrinkage/recovery metrics.")
    if "h5:recovery_candidate:stable_research_only" in warning_codes:
        comp = h4c_comparison or {}
        if comp.get("improved_vs_h4c"):
            parts.append("Aligned transform world improved vs H4c MVP mismatch baseline (research).")
        else:
            parts.append("Recovery candidate: monitor; not production pass.")
    parts.append("Production Bayes and optimizer remain blocked.")
    return " ".join(parts)


def build_world_trust_diagnostic_payload(
    world_id: str,
    *,
    aggregate: dict[str, Any],
    pilot_source: dict[str, Any],
    sample_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one H5 world's pilot aggregates to a research TrustReport candidate payload."""
    role = resolve_world_role(world_id, pilot_row=sample_run)
    alignment = transform_alignment_status(world_id, aggregate=aggregate)
    weak_id = weak_identification_status(world_id, aggregate=aggregate)
    sparse = sparse_recovery_status(world_id)
    codes = derive_warning_codes(world_id, aggregate=aggregate, role=role)
    h4c = (pilot_source.get("h4c_baseline_comparison") or {}).get(world_id) or {}

    beta_agg = aggregate.get("beta_gc_mae") or {}
    mu_agg = aggregate.get("mu_c_mae") or {}
    h5_diag = (sample_run or {}).get("h5_transform_diagnostics") or {}
    policy = (sample_run or {}).get("policy_outcomes") or {}
    conv = (sample_run or {}).get("convergence") or {}
    h5_class = (sample_run or {}).get("h5_classification") or get_h5_validation_world(
        world_id
    ).expected_diagnostic_behavior.get("h5_classification")
    mismatch_detected = bool(
        h5_diag.get("transform_mismatch_detected")
        if h5_diag
        else alignment == "intentional_mismatch"
    )

    candidate_fields = {
        field: payload_field_value(
            field,
            world_id=world_id,
            beta_mean=beta_agg.get("mean"),
            mu_mean=mu_agg.get("mean"),
            alignment=alignment,
            weak_id=weak_id,
            sparse=sparse,
            codes=codes,
            h5_classification=h5_class,
            policy_outcome=policy.get("outcome"),
            rhat=conv.get("rhat_max"),
            ess=conv.get("ess_bulk_min"),
            mismatch_detected=mismatch_detected,
        )
        for field in TRUST_REPORT_CANDIDATE_FIELDS
    }

    return {
        "model_spec_version": pilot_source.get("model_spec_version", H5_MODEL_SPEC_VERSION),
        "world_id": world_id,
        "world_role": role,
        "h5_classification": h5_class,
        "transform_alignment_status": alignment,
        "transform_mismatch_detected": mismatch_detected,
        "weak_identification_status": weak_id,
        "sparse_recovery_status": sparse,
        "recovery_metric_summary": {
            "beta_gc_mae": dict(beta_agg),
            "mu_c_mae": dict(mu_agg),
            "beta_gc_coverage_90": None,
            "shrinkage_ratio_sparse": aggregate.get("shrinkage_ratio_sparse"),
            "h4c_baseline_comparison": dict(h4c) if h4c else None,
            "transform_mismatch_warning_rate": aggregate.get("transform_mismatch_warning_rate"),
            "unexpected_mismatch_warning_rate": aggregate.get("unexpected_mismatch_warning_rate"),
            "collinearity_warning_rate": aggregate.get("collinearity_warning_rate"),
            "weak_identification_warning_rate": aggregate.get("weak_identification_warning_rate"),
        },
        "warning_codes": codes,
        "trust_report_candidate_fields": candidate_fields,
        "recommended_interpretation": recommended_interpretation(
            world_id,
            role=role,
            alignment=alignment,
            warning_codes=codes,
            h4c_comparison=h4c,
        ),
        "production_flags": research_production_flags(),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "outputs_are_diagnostic_only": True,
    }


def payload_field_value(
    field: str,
    *,
    world_id: str,
    beta_mean: Any,
    mu_mean: Any,
    alignment: str,
    weak_id: str,
    sparse: str,
    codes: list[str],
    h5_classification: Any,
    policy_outcome: Any,
    rhat: Any,
    ess: Any,
    mismatch_detected: Any,
) -> Any:
    mapping: dict[str, Any] = {
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "world_id": world_id,
        "transform_alignment_status": alignment,
        "transform_mismatch_detected": mismatch_detected,
        "weak_identification_status": weak_id,
        "sparse_recovery_status": sparse,
        "beta_gc_mae_mean": beta_mean,
        "mu_c_mae_mean": mu_mean,
        "beta_gc_coverage_90_mean": None,
        "shrinkage_ratio_sparse_mean": None,
        "warning_codes": list(codes),
        "h5_classification": h5_classification,
        "policy_outcome": policy_outcome,
        "convergence_rhat_max": rhat,
        "convergence_ess_bulk_min": ess,
    }
    return mapping.get(field)


def _sample_run_for_world(pilot: dict[str, Any], world_id: str) -> dict[str, Any] | None:
    for row in pilot.get("per_run") or []:
        if row.get("world_id") == world_id:
            return row
    return None


def build_trust_diagnostic_mapping(
    pilot: dict[str, Any],
    *,
    source_artifact: str | Path,
) -> dict[str, Any]:
    """Build full H5d mapping artifact from H5c (or compatible) repeated pilot JSON."""
    aggregate_by_world = pilot.get("aggregate_by_world") or {}
    per_world: list[dict[str, Any]] = []
    warning_rate_agg: dict[str, float] = {}

    for wid in H5_WORLD_IDS:
        if wid not in aggregate_by_world:
            raise H5TrustDiagnosticMappingError(f"pilot missing aggregate for {wid!r}")
        agg = aggregate_by_world[wid]
        sample = _sample_run_for_world(pilot, wid)
        payload = build_world_trust_diagnostic_payload(
            wid,
            aggregate=agg,
            pilot_source=pilot,
            sample_run=sample,
        )
        per_world.append(payload)
        for code in payload["warning_codes"]:
            if code == "h5:production:block":
                continue
            warning_rate_agg[code] = warning_rate_agg.get(code, 0.0) + 1.0

    n = len(H5_WORLD_IDS)
    for code, count in list(warning_rate_agg.items()):
        warning_rate_agg[code] = count / n

    return {
        "mapping_id": MAPPING_ID,
        "mapping_version": MAPPING_VERSION,
        "model_spec_version": pilot.get("model_spec_version", H5_MODEL_SPEC_VERSION),
        "source_artifact": str(source_artifact),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "warning_taxonomy": list(WARNING_TAXONOMY),
        "trust_report_candidate_field_catalog": list(TRUST_REPORT_CANDIDATE_FIELDS),
        "fields_intentionally_excluded_from_production": list(FIELDS_EXCLUDED_FROM_PRODUCTION),
        "per_world_diagnostic_payloads": per_world,
        "aggregate_warning_rates": warning_rate_agg,
        "note": (
            "Research-only TrustReport candidate mapping from H5c extended pilot. "
            "Not wired to production TrustReport, optimizer, or DecisionSurface."
        ),
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def load_pilot_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_SOURCE_ARTIFACT)
    if not p.is_file():
        raise H5TrustDiagnosticMappingError(f"source pilot artifact not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_trust_diagnostic_mapping_artifact(
    *,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load H5c pilot, emit H5d trust diagnostic mapping JSON."""
    src = Path(source_path or DEFAULT_SOURCE_ARTIFACT)
    out = Path(output_path or DEFAULT_OUTPUT_ARTIFACT)
    pilot = load_pilot_artifact(src)
    mapping = build_trust_diagnostic_mapping(pilot, source_artifact=src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    mapping["artifact_path"] = str(out)
    return mapping


def validate_trust_diagnostic_mapping_artifact(mapping: dict[str, Any]) -> None:
    """Fail closed if mapping artifact violates research-only schema."""
    required = {
        "mapping_id",
        "mapping_version",
        "model_spec_version",
        "source_artifact",
        "per_world_diagnostic_payloads",
        "aggregate_warning_rates",
        "fields_intentionally_excluded_from_production",
        "hard_gate",
        "production_promotion",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
    missing = required - set(mapping.keys())
    if missing:
        raise H5TrustDiagnosticMappingError(f"mapping schema missing keys: {sorted(missing)}")

    flags = research_production_flags()
    for key, expected in flags.items():
        if mapping.get(key) is not expected:
            raise H5TrustDiagnosticMappingError(f"mapping.{key} must be {expected!r}, got {mapping.get(key)!r}")

    forbidden = set(FIELDS_EXCLUDED_FROM_PRODUCTION)
    for payload in mapping.get("per_world_diagnostic_payloads") or []:
        for key in forbidden:
            if key in payload and payload.get(key) is not None:
                raise H5TrustDiagnosticMappingError(
                    f"forbidden production field {key!r} in per_world payload"
                )
        wc = payload.get("warning_codes") or []
        if "h5:production:block" not in wc:
            raise H5TrustDiagnosticMappingError("each world must include h5:production:block")
        for code in wc:
            if code not in WARNING_TAXONOMY:
                raise H5TrustDiagnosticMappingError(f"unknown warning code: {code!r}")
        pf = payload.get("production_flags") or {}
        if pf.get("approved_for_prod") is not False:
            raise H5TrustDiagnosticMappingError("per_world production_flags.approved_for_prod must be false")
