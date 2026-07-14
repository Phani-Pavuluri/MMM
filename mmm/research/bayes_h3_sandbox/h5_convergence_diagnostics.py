"""Bayes-H5i real-panel convergence diagnostics (research only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema, validate_panel
from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import (
    _config_from_panel,
    _infer_schema,
    load_panel_from_path,
    load_transform_config,
)
from mmm.research.bayes_h3_sandbox.h5_transforms import apply_media_transforms_matrix
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    classify_convergence_status,
    evidence_promotion_allowed,
    research_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_FAST
from mmm.utils.math import safe_log

INVESTIGATION_ID = "INV-H5I_REAL_PANEL_CONVERGENCE_DIAGNOSTICS"
DIAGNOSTICS_ARTIFACT_ID = "BAYES_H5I_CONVERGENCE_DIAGNOSTICS_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601"
MATRIX_ARTIFACT_ID = "BAYES_H5I_CONVERGENCE_EXPERIMENT_MATRIX_20260601"

DEFAULT_PANEL_PATH = Path("examples/sample_panel.csv")
DEFAULT_TRANSFORM_CONFIG = Path("docs/06_investigations/h5g_sample_panel_transform_config.json")
DEFAULT_H5H_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5H_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json"
)
DEFAULT_H5G_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5G_SHADOW_RUN_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json"
)

PANEL_ID = "examples_mmm_sample_panel_v1"
DATASET_SNAPSHOT_ID = "mmm-examples-sample-panel-frozen-2022"

FORBIDDEN_ARTIFACT_FIELDS = frozenset(
    {
        "decision_surface",
        "optimizer_ready_curves",
        "budget_recommendation",
        "recommendation",
        "production_decision_surface",
    }
)


class H5ConvergenceDiagnosticError(ValueError):
    """H5 convergence diagnostic failed — fail closed."""


@dataclass(frozen=True)
class ConvergenceExperimentSpec:
    variant_id: str
    changed_factor: str
    sampler_profile: str
    sampler_overrides: dict[str, Any]
    model_overrides: dict[str, Any]
    channel_subset: tuple[str, ...] | None = None
    notes: str = ""


def _load_panel_and_schema(
    panel_path: str | Path,
    transform_config: dict[str, Any],
) -> tuple[pd.DataFrame, PanelSchema]:
    df = load_panel_from_path(panel_path)
    schema = _infer_schema(df, transform_config)
    if missing := [c for c in (schema.geo_column, schema.week_column, schema.target_column) if c not in df.columns]:
        raise H5ConvergenceDiagnosticError(f"panel missing required columns: {missing}")
    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)
    return df, schema


def inspect_panel_shape(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    n_rows = len(df)
    geos = df[schema.geo_column].unique().tolist()
    weeks = df[schema.week_column].nunique()
    return {
        "n_rows": n_rows,
        "n_geos": len(geos),
        "n_weeks_unique": int(weeks),
        "geo_ids": geos,
        "rows_per_geo": {str(g): int((df[schema.geo_column] == g).sum()) for g in geos},
        "missingness": {
            schema.target_column: int(df[schema.target_column].isna().sum()),
            **{ch: int(df[ch].isna().sum()) for ch in schema.channel_columns},
        },
    }


def inspect_scale_diagnostics(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    y = df[schema.target_column].to_numpy(dtype=float)
    log_y = safe_log(y)
    media_stats: dict[str, Any] = {}
    for ch in schema.channel_columns:
        col = df[ch].to_numpy(dtype=float)
        media_stats[ch] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "cv": float(np.std(col) / (np.mean(col) + 1e-9)),
        }
    return {
        "outcome_level": {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        },
        "outcome_log": {
            "min": float(np.min(log_y)),
            "max": float(np.max(log_y)),
            "mean": float(np.mean(log_y)),
            "std": float(np.std(log_y)),
        },
        "media_level_by_channel": media_stats,
        "likelihood_note": "semi_log uses log(outcome) with Normal likelihood; sigma scale must match log-y residual",
    }


def inspect_collinearity_diagnostics(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    media = df[list(schema.channel_columns)].to_numpy(dtype=float)
    if media.shape[1] < 2:
        return {"pairwise_correlations": {}, "max_abs_correlation": 0.0}
    corr = np.corrcoef(media.T)
    pairs: dict[str, float] = {}
    cols = list(schema.channel_columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs[f"{cols[i]}|{cols[j]}"] = float(corr[i, j])
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    return {
        "pairwise_correlations": pairs,
        "max_abs_correlation": float(np.max(np.abs(off_diag))) if len(off_diag) else 0.0,
    }


def inspect_sparsity_diagnostics(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    by_channel: dict[str, Any] = {}
    for ch in schema.channel_columns:
        col = df[ch].to_numpy(dtype=float)
        zero_share = float(np.mean(col <= 0.0))
        near_zero_share = float(np.mean(col < 1e-6))
        by_channel[ch] = {
            "zero_share": zero_share,
            "near_zero_share": near_zero_share,
            "min_positive": float(np.min(col[col > 0])) if np.any(col > 0) else None,
        }
    return {"by_channel": by_channel, "any_hard_zeros": any(v["zero_share"] > 0 for v in by_channel.values())}


def inspect_geo_sample_sizes(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    counts = df.groupby(schema.geo_column).size()
    return {
        "min_rows_per_geo": int(counts.min()),
        "max_rows_per_geo": int(counts.max()),
        "counts_by_geo": {str(k): int(v) for k, v in counts.items()},
    }


def inspect_transform_output_scale(
    df: pd.DataFrame,
    schema: PanelSchema,
    transform_config: dict[str, Any],
) -> dict[str, Any]:
    channels = list(schema.channel_columns)
    transforms = dict(transform_config.get("media_transforms_by_channel") or {})
    params = dict(transform_config.get("transform_params_by_channel") or {})
    raw_x = df[channels].to_numpy(dtype=float)
    x = apply_media_transforms_matrix(raw_x, channels, transforms, transform_params_by_channel=params)
    out: dict[str, Any] = {}
    for i, ch in enumerate(channels):
        col = x[:, i]
        out[ch] = {
            "transform_id": transforms.get(ch, "identity"),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return {
        "post_transform_by_channel": out,
        "note": "identity transform applies per-column standardization in H5 registry",
    }


def summarize_posterior_diagnostics(
    fit_artifact: dict[str, Any] | None,
    *,
    sampler_profile: str | None = None,
) -> dict[str, Any]:
    if not fit_artifact:
        return {"status": "not_available"}
    conv = fit_artifact.get("convergence_diagnostics") or {}
    post = fit_artifact.get("posterior_summary") or {}
    status = classify_convergence_status(
        rhat_max=conv.get("rhat_max"),
        divergence_count=conv.get("divergence_count"),
    )
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in conv.get("per_parameter") or []:
        fam = str(row.get("family") or "other")
        by_family.setdefault(fam, []).append(row)

    return {
        "sampler_profile": sampler_profile,
        "rhat_max": conv.get("rhat_max"),
        "ess_bulk_min": conv.get("ess_bulk_min"),
        "divergence_count": conv.get("divergence_count"),
        "chains": conv.get("chains"),
        "draws_per_chain": conv.get("draws_per_chain"),
        "convergence_status": status,
        "evidence_promotion_allowed": evidence_promotion_allowed(status),
        "sigma_mean": post.get("sigma_mean"),
        "worst_rhat_parameters": conv.get("worst_rhat_parameters") or [],
        "parameters_by_family": {k: v for k, v in by_family.items()},
    }


def infer_suspected_failure_modes(
    panel_summary: dict[str, Any],
    scale_diag: dict[str, Any],
    collinearity_diag: dict[str, Any],
    sparsity_diag: dict[str, Any],
    posterior_diag: dict[str, Any],
) -> list[dict[str, str]]:
    modes: list[dict[str, str]] = []

    max_corr = float(collinearity_diag.get("max_abs_correlation") or 0.0)
    if max_corr >= 0.85:
        modes.append(
            {
                "mode": "media_collinearity",
                "severity": "high",
                "note": f"Max |channel correlation|={max_corr:.3f} — weak identification for geo-specific betas",
            }
        )

    n_geos = int(panel_summary.get("n_geos") or 0)
    n_channels = len(scale_diag.get("media_level_by_channel") or {})
    if n_geos <= 3 and n_channels >= 3:
        modes.append(
            {
                "mode": "overparameterized_for_panel_size",
                "severity": "high",
                "note": (
                    f"{n_geos} geos × {n_channels} channels with partial pooling — "
                    "many hierarchical offsets per row"
                ),
            }
        )

    sigma_mean = posterior_diag.get("sigma_mean")
    if sigma_mean is not None and float(sigma_mean) < 0.05:
        modes.append(
            {
                "mode": "tight_likelihood_scale",
                "severity": "medium",
                "note": f"Posterior sigma_mean≈{float(sigma_mean):.4f} on log scale — funnel / sampler stress",
            }
        )

    worst = posterior_diag.get("worst_rhat_parameters") or []
    if worst:
        top = worst[0]
        modes.append(
            {
                "mode": "worst_rhat_parameter_family",
                "severity": "medium",
                "note": f"Worst r_hat {top.get('parameter')} ({top.get('family')}) = {top.get('r_hat')}",
            }
        )

    div = int(posterior_diag.get("divergence_count") or 0)
    if div > 0:
        modes.append(
            {
                "mode": "nuts_divergences",
                "severity": "high" if div > 10 else "medium",
                "note": f"{div} divergences — often tau/beta geometry or step size; not fixed by label semantics alone",
            }
        )

    if not modes:
        modes.append(
            {"mode": "unknown", "severity": "low", "note": "No dominant mode inferred from static diagnostics"}
        )
    return modes


def recommend_next_experiments(suspected_modes: list[dict[str, str]], matrix_rows: list[dict[str, Any]]) -> list[str]:
    recs = [
        "Do not run additional real panels until a variant reaches converged_diagnostic_only or documented "
        "weak_convergence.",
        "Prefer geometry fixes (scaling, collinearity reduction, simpler hierarchy) before aggressive sampler "
        "tuning alone.",
    ]
    mode_ids = {m["mode"] for m in suspected_modes}
    if "media_collinearity" in mode_ids:
        recs.append("Try single-channel or orthogonalized media subsets to localize collinearity impact.")
    if "overparameterized_for_panel_size" in mode_ids:
        recs.append("Try pooled/no-geo-offset probe or fewer channels; 3 geos is marginal for partial pooling.")
    if "tight_likelihood_scale" in mode_ids or "nuts_divergences" in mode_ids:
        recs.append("Try outcome/media prescale experiment and tighter tau priors; monitor sigma posterior.")

    best = min(
        (r for r in matrix_rows if r.get("rhat_max") is not None),
        key=lambda r: (float(r["rhat_max"]), int(r.get("divergence_count") or 999)),
        default=None,
    )
    if best:
        recs.append(
            f"Best matrix variant so far: {best.get('variant_id')} "
            f"(rhat_max={best.get('rhat_max')}, divergences={best.get('divergence_count')})."
        )
    return recs


def build_convergence_diagnostics_artifact(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    source_h5h_artifact_path: str | Path = DEFAULT_H5H_ARTIFACT,
    source_h5g_artifact_path: str | Path = DEFAULT_H5G_ARTIFACT,
    fit_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transform_config = load_transform_config(transform_config_path)
    df, schema = _load_panel_and_schema(panel_path, transform_config)

    panel_summary = inspect_panel_shape(df, schema)
    scale_diag = inspect_scale_diagnostics(df, schema)
    collinearity_diag = inspect_collinearity_diagnostics(df, schema)
    sparsity_diag = inspect_sparsity_diagnostics(df, schema)
    geo_diag = inspect_geo_sample_sizes(df, schema)
    transform_scale = inspect_transform_output_scale(df, schema, transform_config)

    h5h = json.loads(Path(source_h5h_artifact_path).read_text(encoding="utf-8"))
    h5g = (
        json.loads(Path(source_h5g_artifact_path).read_text(encoding="utf-8"))
        if Path(source_h5g_artifact_path).is_file()
        else {}
    )
    shadow_h5h = h5h.get("shadow_run") or {}
    pd_h5h = shadow_h5h.get("posterior_diagnostics") or {}
    post_h5h = summarize_posterior_diagnostics(
        {
            "convergence_diagnostics": pd_h5h.get("convergence_diagnostics"),
            "posterior_summary": pd_h5h.get("posterior_summary"),
        },
        sampler_profile=h5h.get("sampler_profile") or "extended",
    )

    if fit_artifact is not None:
        post_h5h = summarize_posterior_diagnostics(fit_artifact, sampler_profile="replay")

    suspected = infer_suspected_failure_modes(panel_summary, scale_diag, collinearity_diag, sparsity_diag, post_h5h)

    return {
        "artifact_id": DIAGNOSTICS_ARTIFACT_ID,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": PANEL_ID,
        "dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "panel_path": str(panel_path),
        "source_artifacts": {
            "h5h_shadow_run": str(source_h5h_artifact_path),
            "h5g_shadow_run": str(source_h5g_artifact_path) if Path(source_h5g_artifact_path).is_file() else None,
        },
        "reference_runs": {
            "h5g_fast": _extract_run_summary(h5g),
            "h5h_extended": _extract_run_summary(h5h),
        },
        "panel_summary": panel_summary,
        "scale_diagnostics": scale_diag,
        "collinearity_diagnostics": collinearity_diag,
        "sparsity_diagnostics": sparsity_diag,
        "geo_sample_diagnostics": geo_diag,
        "transform_output_scale": transform_scale,
        "posterior_diagnostic_summary": post_h5h,
        "suspected_failure_modes": suspected,
        "recommended_next_experiments": recommend_next_experiments(suspected, []),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def _extract_run_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    shadow = artifact.get("shadow_run") or {}
    post = shadow.get("posterior_diagnostics") or {}
    conv = post.get("convergence_diagnostics") or {}
    real = shadow.get("real_panel_diagnostics") or {}
    return {
        "sampler_profile": artifact.get("sampler_profile"),
        "fast_mcmc_profile": artifact.get("fast_mcmc_profile"),
        "extended_mcmc_profile": artifact.get("extended_mcmc_profile"),
        "rhat_max": conv.get("rhat_max"),
        "divergence_count": conv.get("divergence_count"),
        "convergence_status": real.get("convergence_status")
        or (shadow.get("trust_report_candidate_diagnostics") or {})
        .get("trust_report_candidate_fields", {})
        .get("convergence_status"),
    }


def default_experiment_specs() -> list[ConvergenceExperimentSpec]:
    return [
        ConvergenceExperimentSpec(
            variant_id="H5I-BASELINE-FAST-REPLAY",
            changed_factor="baseline_sampler",
            sampler_profile="fast",
            sampler_overrides=dict(SAMPLER_FAST),
            model_overrides={"h5_panel_context": "real_panel", "h5_real_panel": True},
            notes="Replay H5g-style fast profile on sample panel for matrix comparison",
        ),
        ConvergenceExperimentSpec(
            variant_id="H5I-SCALING-ZSCORE-PRESCALE",
            changed_factor="scaling_prescale",
            sampler_profile="fast",
            sampler_overrides=dict(SAMPLER_FAST),
            model_overrides={
                "h5_panel_context": "real_panel",
                "h5_real_panel": True,
                "media_prescale": "zscore_panel",
                "outcome_prescale": "zscore_log",
            },
            notes="Z-score raw media + z-score log outcome before identity transform",
        ),
        ConvergenceExperimentSpec(
            variant_id="H5I-SAMPLER-TARGET-099",
            changed_factor="sampler_target_accept",
            sampler_profile="fast_target_099",
            sampler_overrides={**SAMPLER_FAST, "target_accept": 0.99, "tune": 400},
            model_overrides={"h5_panel_context": "real_panel", "h5_real_panel": True},
            notes="Higher target_accept — sampler-only probe",
        ),
        ConvergenceExperimentSpec(
            variant_id="H5I-PRIOR-TIGHT-TAU",
            changed_factor="prior_tau_tightening",
            sampler_profile="fast",
            sampler_overrides=dict(SAMPLER_FAST),
            model_overrides={
                "h5_panel_context": "real_panel",
                "h5_real_panel": True,
                "tau_channel_prior_sigma": 0.15,
            },
            notes="Tighter tau_channel HalfNormal(0.15)",
        ),
        ConvergenceExperimentSpec(
            variant_id="H5I-PRIOR-TIGHT-TAU-MU",
            changed_factor="prior_tau_mu_tightening",
            sampler_profile="fast",
            sampler_overrides=dict(SAMPLER_FAST),
            model_overrides={
                "h5_panel_context": "real_panel",
                "h5_real_panel": True,
                "tau_channel_prior_sigma": 0.15,
                "mu_channel_prior_sigma": 0.2,
                "sigma_prior_sigma": 0.5,
            },
            notes="Tighter tau and mu_channel priors + smaller sigma prior",
        ),
        ConvergenceExperimentSpec(
            variant_id="H5I-SINGLE-CHANNEL-SEARCH",
            changed_factor="model_simplified_channels",
            sampler_profile="fast",
            sampler_overrides=dict(SAMPLER_FAST),
            model_overrides={"h5_panel_context": "real_panel", "h5_real_panel": True},
            channel_subset=("search",),
            notes="Single media channel to localize geometry / collinearity",
        ),
    ]


def run_convergence_experiment(
    spec: ConvergenceExperimentSpec,
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    execute_fit: bool = True,
) -> dict[str, Any]:
    transform_config = load_transform_config(transform_config_path)
    df, schema = _load_panel_and_schema(panel_path, transform_config)

    if spec.channel_subset:
        channels = tuple(spec.channel_subset)
        transform_config = {
            **transform_config,
            "media_transforms_by_channel": {
                ch: transform_config["media_transforms_by_channel"][ch] for ch in channels
            },
        }
        schema = PanelSchema(
            schema.geo_column,
            schema.week_column,
            schema.target_column,
            channels,
            schema.control_columns,
        )
        df = df[[schema.geo_column, schema.week_column, schema.target_column, *channels]]

    cfg, _ = _config_from_panel(df, schema, fast_mcmc=True, extended_mcmc=False)
    cfg = cfg.model_copy(update={"bayesian": cfg.bayesian.model_copy(update=spec.sampler_overrides)})

    fit_artifact: dict[str, Any] | None = None
    if execute_fit:
        fit_artifact = run_sandbox_fit(
            cfg,
            schema,
            df,
            sandbox_model_overrides={
                "media_transforms_by_channel": dict(transform_config["media_transforms_by_channel"]),
                "transform_params_by_channel": dict(transform_config.get("transform_params_by_channel") or {}),
                "h5_transform_mismatch_mode": str(transform_config.get("transform_mismatch_mode", "aligned")),
                **spec.model_overrides,
            },
            model_spec_version=H5_MODEL_SPEC_VERSION,
            enable_h5_sandbox=True,
            research_only=True,
        )

    post = summarize_posterior_diagnostics(fit_artifact, sampler_profile=spec.sampler_profile)
    return {
        "variant_id": spec.variant_id,
        "changed_factor": spec.changed_factor,
        "sampler_profile": spec.sampler_profile,
        "sampler_settings": spec.sampler_overrides,
        "model_overrides": spec.model_overrides,
        "channel_subset": list(spec.channel_subset) if spec.channel_subset else None,
        "rhat_max": post.get("rhat_max"),
        "ess_bulk_min": post.get("ess_bulk_min"),
        "divergence_count": post.get("divergence_count"),
        "convergence_status": post.get("convergence_status"),
        "evidence_promotion_allowed": post.get("evidence_promotion_allowed"),
        "worst_rhat_parameters": post.get("worst_rhat_parameters"),
        "sigma_mean": post.get("sigma_mean"),
        "notes": spec.notes,
        "recommendation": _variant_recommendation(post),
    }


def _variant_recommendation(post: dict[str, Any]) -> str:
    status = post.get("convergence_status")
    if status == "converged_diagnostic_only":
        return "report_only_pass — still not production promotion"
    if status == "weak_convergence":
        return "report_only_weak — may justify further probes on this panel only"
    return "report_only_fail — does not clear evidence bar"


def run_convergence_experiment_matrix(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    execute_fit: bool = True,
    specs: list[ConvergenceExperimentSpec] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs or default_experiment_specs():
        rows.append(
            run_convergence_experiment(
                spec,
                panel_path=panel_path,
                transform_config_path=transform_config_path,
                execute_fit=execute_fit,
            )
        )

    h5g_path = DEFAULT_H5G_ARTIFACT
    h5h_path = DEFAULT_H5H_ARTIFACT
    reference_rows = []
    if h5g_path.is_file():
        ref_g = _extract_run_summary(json.loads(h5g_path.read_text(encoding="utf-8")))
        reference_rows.append(
            {
                "variant_id": "H5I-REF-H5G-FAST",
                "changed_factor": "reference_artifact",
                "sampler_profile": "fast",
                "rhat_max": ref_g.get("rhat_max"),
                "divergence_count": ref_g.get("divergence_count"),
                "convergence_status": ref_g.get("convergence_status") or "failed_convergence",
                "notes": "From committed H5g shadow artifact (not re-run)",
                "recommendation": "report_only_fail",
            }
        )
    if h5h_path.is_file():
        ref_h = _extract_run_summary(json.loads(h5h_path.read_text(encoding="utf-8")))
        reference_rows.append(
            {
                "variant_id": "H5I-REF-H5H-EXTENDED",
                "changed_factor": "reference_artifact",
                "sampler_profile": "extended",
                "rhat_max": ref_h.get("rhat_max"),
                "divergence_count": ref_h.get("divergence_count"),
                "convergence_status": ref_h.get("convergence_status") or "failed_convergence",
                "notes": "From committed H5h shadow artifact (not re-run)",
                "recommendation": "report_only_fail",
            }
        )

    all_rows = reference_rows + rows
    return {
        "artifact_id": MATRIX_ARTIFACT_ID,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": PANEL_ID,
        "dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "experiments": all_rows,
        "matrix_notes": (
            "Research-only convergence probes on examples/sample_panel.csv only. "
            "Fast profile used for matrix comparability; H5h extended reference included from artifact."
        ),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def validate_convergence_artifact(artifact: dict[str, Any]) -> None:
    for key, val in research_production_flags().items():
        if artifact.get(key) is not val:
            raise H5ConvergenceDiagnosticError(f"artifact.{key} must be {val!r}")
    prod_keys = set(research_production_flags())
    for forbidden in FORBIDDEN_ARTIFACT_FIELDS:
        if forbidden in prod_keys:
            if artifact.get(forbidden) is True:
                raise H5ConvergenceDiagnosticError(f"forbidden production flag true: {forbidden!r}")
            continue
        if forbidden in artifact and artifact.get(forbidden) is not None:
            raise H5ConvergenceDiagnosticError(f"forbidden field: {forbidden!r}")


def write_investigation_artifacts(
    *,
    diagnostics_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
    run_experiments: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diag_path = Path(diagnostics_path or f"docs/05_validation/archives/{DIAGNOSTICS_ARTIFACT_ID}.json")
    matrix_out = Path(matrix_path or f"docs/05_validation/archives/{MATRIX_ARTIFACT_ID}.json")

    matrix = run_convergence_experiment_matrix(execute_fit=run_experiments)
    diag = build_convergence_diagnostics_artifact()
    diag["recommended_next_experiments"] = recommend_next_experiments(
        diag["suspected_failure_modes"],
        matrix.get("experiments") or [],
    )

    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diag, indent=2, default=str) + "\n", encoding="utf-8")
    matrix_out.write_text(json.dumps(matrix, indent=2, default=str) + "\n", encoding="utf-8")
    validate_convergence_artifact(diag)
    validate_convergence_artifact(matrix)
    return diag, matrix


def main() -> int:
    diag, matrix = write_investigation_artifacts(run_experiments=True)
    print(diag["artifact_id"], matrix["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
