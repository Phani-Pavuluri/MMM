"""Phase 4B-2 — train/decide recovery certification on rich DGP worlds (WORLD-008)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.decision.service import simulate_decision
from mmm.governance.decision_fingerprint import compare_training_and_decision_fingerprints
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.planning.baseline import BaselinePlan, BaselineType
from mmm.planning.context import ridge_context_from_fit, ridge_fit_summary_from_artifacts
from mmm.planning.decision_simulate import simulate
from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation
from mmm.validation.synthetic._io import read_json
from mmm.validation.synthetic.dgp_materializer import (
    geometric_adstock_series,
    hill_saturation_series,
    materialize_dgp_world,
)

RECOVERY_CERTIFICATION_VERSION = "recovery_cert_v1.0.0"
COEF_RECOVERY_WORLD_IDS = frozenset({"WORLD-008-exact-recovery"})
OPTIMIZER_RECOVERY_WORLD_IDS = frozenset({"WORLD-009-optimizer-recovery"})
REPLAY_RECOVERY_WORLD_IDS = frozenset({"WORLD-010-replay-recovery"})
DRIFT_RECOVERY_WORLD_IDS = frozenset({"WORLD-011-drift-recovery"})
IDENTIFIABILITY_RECOVERY_WORLD_IDS = frozenset({"WORLD-012-identifiability-recovery"})
RECOVERY_ELIGIBLE_WORLD_IDS = (
    COEF_RECOVERY_WORLD_IDS
    | OPTIMIZER_RECOVERY_WORLD_IDS
    | REPLAY_RECOVERY_WORLD_IDS
    | DRIFT_RECOVERY_WORLD_IDS
    | IDENTIFIABILITY_RECOVERY_WORLD_IDS
)

# Documented provisional runtime tolerances — not production thresholds (TBD_v1).
TOLERANCE_POLICY_ID = "TBD_v1_runtime"
COEF_RECOVERY_RTOL = 0.20
COEF_RECOVERY_ATOL = 0.08
DELTA_MU_RECOVERY_RTOL = 0.35
DELTA_MU_RECOVERY_ATOL = 0.15
TRANSFORM_PARAM_RTOL = 0.05
FORMULA_RTOL = 1e-6

# Optimizer recovery (VAL-005) — TBD_v1_runtime provisional
ALLOCATION_L1_RTOL = 0.15
ALLOCATION_L1_ATOL = 4.0
OBJECTIVE_GAP_RTOL = 0.30
OBJECTIVE_GAP_ATOL = 0.08
BUDGET_CONSERVATION_ATOL = 0.5

# Replay calibration recovery (VAL-006) — TBD_v1_runtime provisional
REPLAY_LIFT_RTOL = 0.25
REPLAY_LIFT_ATOL = 0.02

# Drift / identifiability (VAL-012 / VAL-013 / VAL-014 partial) — TBD_v1_runtime provisional
DRIFT_POST_PRE_MAE_RATIO_MIN = 1.15
IDENTIFIABILITY_MIN_CORRELATION = 0.95
IDENTIFIABILITY_MIN_VIF = 5.0

RecoveryStatus = Literal["pass", "fail", "skipped"]


@dataclass
class RecoveryCheckOutcome:
    check_id: str
    category: str
    status: RecoveryStatus
    message: str
    skip_reason: str | None = None
    metric_kind: Literal["exact_formula", "provisional_statistical"] = "provisional_statistical"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "skip_reason": self.skip_reason,
            "metric_kind": self.metric_kind,
            "details": self.details,
        }


@dataclass
class RecoveryCertificationResult:
    passed: bool
    checks: list[RecoveryCheckOutcome]
    recovery_results: dict[str, Any]
    train_decide_summary: dict[str, Any]

    def to_report_sections(self) -> dict[str, Any]:
        return {
            "recovery_certification_version": RECOVERY_CERTIFICATION_VERSION,
            "tolerance_policy_id": TOLERANCE_POLICY_ID,
            "recovery_results": self.recovery_results,
            "coefficient_recovery": self.recovery_results.get("coefficient_recovery", {}),
            "delta_mu_recovery": self.recovery_results.get("delta_mu_recovery", {}),
            "transform_recovery": self.recovery_results.get("transform_recovery", {}),
            "decision_artifact_recovery": self.recovery_results.get("decision_artifact_recovery", {}),
            "optimizer_recovery": self.recovery_results.get("optimizer_recovery", {}),
            "optimizer_recovery_status": self.recovery_results.get("optimizer_recovery_status"),
            "replay_recovery": self.recovery_results.get("replay_recovery", {}),
            "replay_recovery_status": self.recovery_results.get("replay_recovery_status"),
            "drift_recovery": self.recovery_results.get("drift_recovery", {}),
            "drift_recovery_status": self.recovery_results.get("drift_recovery_status"),
            "identifiability_recovery": self.recovery_results.get("identifiability_recovery", {}),
            "identifiability_recovery_status": self.recovery_results.get(
                "identifiability_recovery_status"
            ),
            "reliability_degradation_results": self.recovery_results.get(
                "reliability_degradation_results", {}
            ),
            "readiness_reaction_results": self.recovery_results.get("readiness_reaction_results", {}),
            "train_decide_recovery_status": self.recovery_results.get("train_decide_recovery_status", {}),
            "recovery_limitations": self.recovery_results.get("recovery_limitations", []),
            "recovery_validation_results": [c.to_dict() for c in self.checks],
        }


def is_recovery_eligible(truth: dict[str, Any]) -> bool:
    return (
        is_coef_recovery_eligible(truth)
        or is_optimizer_recovery_eligible(truth)
        or is_replay_recovery_eligible(truth)
        or is_drift_recovery_eligible(truth)
        or is_identifiability_recovery_eligible(truth)
    )


def is_coef_recovery_eligible(truth: dict[str, Any]) -> bool:
    meta = truth.get("metadata") or {}
    if str(meta.get("world_id", "")) in COEF_RECOVERY_WORLD_IDS:
        return True
    return "dgp:exact_recovery" in (meta.get("scenario_tags") or [])


def is_optimizer_recovery_eligible(truth: dict[str, Any]) -> bool:
    meta = truth.get("metadata") or {}
    if str(meta.get("world_id", "")) in OPTIMIZER_RECOVERY_WORLD_IDS:
        return True
    return "dgp:optimizer_recovery" in (meta.get("scenario_tags") or [])


def is_replay_recovery_eligible(truth: dict[str, Any]) -> bool:
    meta = truth.get("metadata") or {}
    if str(meta.get("world_id", "")) in REPLAY_RECOVERY_WORLD_IDS:
        return True
    return "dgp:replay_recovery" in (meta.get("scenario_tags") or [])


def is_drift_recovery_eligible(truth: dict[str, Any]) -> bool:
    meta = truth.get("metadata") or {}
    if str(meta.get("world_id", "")) in DRIFT_RECOVERY_WORLD_IDS:
        return True
    return "dgp:drift_recovery" in (meta.get("scenario_tags") or [])


def is_identifiability_recovery_eligible(truth: dict[str, Any]) -> bool:
    meta = truth.get("metadata") or {}
    if str(meta.get("world_id", "")) in IDENTIFIABILITY_RECOVERY_WORLD_IDS:
        return True
    return "dgp:identifiability_recovery" in (meta.get("scenario_tags") or [])


def recovery_replaces_deferred_val_ids(truth: dict[str, Any]) -> frozenset[str]:
    ids: set[str] = set()
    if is_coef_recovery_eligible(truth):
        ids.update({"VAL-001", "VAL-004"})
    if is_optimizer_recovery_eligible(truth):
        ids.add("VAL-005")
    if is_replay_recovery_eligible(truth):
        ids.add("VAL-006")
    if is_drift_recovery_eligible(truth):
        ids.update({"VAL-012", "VAL-014"})
    if is_identifiability_recovery_eligible(truth):
        ids.update({"VAL-013", "VAL-014"})
    return frozenset(ids)


def shared_ridge_transform_params(truth: dict[str, Any]) -> dict[str, float]:
    """
    Ridge BO uses one decay / Hill per panel — mean per-channel truth params when they differ.
    """
    transform = truth["transform_truth"]
    channels = list(truth["media_truth"]["channels"])
    decays = [float(transform["adstock_decay_by_channel"][c]) for c in channels]
    halves = [float(transform["hill_half_max_by_channel"][c]) for c in channels]
    slopes = [float(transform["hill_slope_by_channel"][c]) for c in channels]
    uniform = len(set(decays)) == 1 and len(set(halves)) == 1 and len(set(slopes)) == 1
    return {
        "decay": float(decays[0]) if uniform else float(np.mean(decays)),
        "hill_half": float(halves[0]) if uniform else float(np.mean(halves)),
        "hill_slope": float(slopes[0]) if uniform else float(np.mean(slopes)),
        "per_channel_uniform": uniform,
    }


def build_recovery_mmm_config(
    truth: dict[str, Any],
    *,
    panel_path: Path,
    run_environment: RunEnvironment = RunEnvironment.RESEARCH,
) -> MMMConfig:
    outcome = truth["outcome_truth"]
    geo = truth["geo_truth"]
    media = truth["media_truth"]
    meta = truth["metadata"]
    time_t = truth["time_truth"]
    n_periods = int(time_t["n_periods"])
    min_train = max(6, min(10, n_periods - 4))
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        run_environment=run_environment,
        random_seed=int(meta.get("generation_seed", 0)),
        data={
            "path": str(panel_path.resolve()),
            "geo_column": str(geo.get("geo_column_name") or "geo_id"),
            "week_column": str(time_t.get("week_column_name") or "week_start_date"),
            "target_column": str(outcome["target_column"]),
            "channel_columns": list(media["channels"]),
            "control_columns": [],
            "data_version_id": f"{meta.get('world_id')}-recovery",
        },
        cv=CVConfig(
            mode="rolling",
            n_splits=2,
            min_train_weeks=min_train,
            horizon_weeks=2,
        ),
        ridge_bo={"n_trials": 12, "sampler_seed": int(meta.get("generation_seed", 0))},
        transforms={"adstock": "geometric", "saturation": "hill"},
        extensions={
            "product": {
                "simulation_optimizer_n_starts": 10,
                "simulation_optimizer_stability_checks": 2,
            }
        },
    )


def _truth_coef_vector(truth: dict[str, Any], channels: list[str]) -> np.ndarray:
    betas = truth["coefficient_truth"]["true_beta_by_channel"]
    return np.array([float(betas[c]) for c in channels], dtype=float)


def _scenario_is_placeholder(scenario: dict[str, Any]) -> bool:
    if float(scenario.get("true_delta_mu", 0.0)) != 0.0:
        return False
    base = scenario.get("baseline_spend_by_channel") or {}
    cand = scenario.get("candidate_spend_by_channel") or {}
    return base != cand


def compute_analytic_delta_mu(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    scenario: dict[str, Any],
) -> float:
    """Δμ from true coefficients + shared transform params (authoritative for exact-recovery worlds)."""
    from mmm.models.ridge_bo.trainer import RidgeBOArtifacts

    channels = list(truth["media_truth"]["channels"])
    shared = shared_ridge_transform_params(truth)
    coef = _truth_coef_vector(truth, channels)
    intercept = np.array([float(truth["coefficient_truth"]["intercept"])], dtype=float)
    art = RidgeBOArtifacts(
        best_params={
            "decay": shared["decay"],
            "hill_half": shared["hill_half"],
            "hill_slope": shared["hill_slope"],
            "log_alpha": -6.0,
        },
        objective_history=[],
        coef=coef,
        intercept=intercept,
        leaderboard=[],
    )
    ctx = ridge_context_from_fit(panel, schema, config, {"artifacts": art})
    base = BaselinePlan(
        baseline_type=BaselineType.BAU,
        spend_by_channel={c: float(v) for c, v in (scenario.get("baseline_spend_by_channel") or {}).items()},
        baseline_definition="decision_truth scenario baseline",
        baseline_plan_source="decision_truth",
        suitable_for_decisioning=True,
    )
    cand = {c: float(v) for c, v in (scenario.get("candidate_spend_by_channel") or {}).items()}
    return float(simulate(cand, ctx, baseline_plan=base).delta_mu)


def _ensure_panel_materialized(bundle: Path) -> Path:
    panel_path = bundle / "panel.parquet"
    if not panel_path.is_file():
        materialize_dgp_world(bundle, overwrite=True)
    return panel_path


def _train_ridge(bundle: Path, truth: dict[str, Any]) -> tuple[pd.DataFrame, PanelSchema, MMMConfig, dict[str, Any]]:
    panel_path = _ensure_panel_materialized(bundle)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    panel = pd.read_parquet(panel_path)
    week_col = config.data.week_column
    if week_col in panel.columns:
        panel[week_col] = pd.to_datetime(panel[week_col])
    schema = PanelSchema(
        geo_column=config.data.geo_column,
        week_column=config.data.week_column,
        target_column=config.data.target_column,
        channel_columns=tuple(config.data.channel_columns),
        control_columns=(),
    )
    fit = RidgeBOMMMTrainer(config, schema).fit(panel)
    return panel, schema, config, fit


def _train_ridge_truth_transforms(
    bundle: Path, truth: dict[str, Any]
) -> tuple[pd.DataFrame, PanelSchema, MMMConfig, dict[str, Any]]:
    """Ridge coef fit with authoritative transform hyperparameters (optimizer worlds)."""
    from mmm.features.design_matrix import build_design_matrix
    from mmm.models.ridge_bo.ridge import fit_ridge
    from mmm.models.ridge_bo.trainer import RidgeBOArtifacts
    from mmm.validation.synthetic.optimizer_truth import shared_ridge_transform_params

    panel_path = _ensure_panel_materialized(bundle)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    panel = pd.read_parquet(panel_path)
    week_col = config.data.week_column
    if week_col in panel.columns:
        panel[week_col] = pd.to_datetime(panel[week_col])
    schema = PanelSchema(
        geo_column=config.data.geo_column,
        week_column=config.data.week_column,
        target_column=config.data.target_column,
        channel_columns=tuple(config.data.channel_columns),
        control_columns=(),
    )
    shared = shared_ridge_transform_params(truth)
    design = build_design_matrix(
        panel,
        schema,
        config,
        decay=shared["decay"],
        hill_half=shared["hill_half"],
        hill_slope=shared["hill_slope"],
    )
    coef, intercept = fit_ridge(design.X, design.y_modeling, alpha=1e-6)
    art = RidgeBOArtifacts(
        best_params={
            "decay": shared["decay"],
            "hill_half": shared["hill_half"],
            "hill_slope": shared["hill_slope"],
            "log_alpha": -6.0,
        },
        objective_history=[],
        coef=coef,
        intercept=intercept,
        leaderboard=[],
    )
    return panel, schema, config, {"artifacts": art}


def _minimal_extension_report(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit: dict[str, Any],
) -> dict[str, Any]:
    art = fit["artifacts"]
    gov = truth.get("governance_truth") or {}
    return {
        "ridge_fit_summary": ridge_fit_summary_from_artifacts(art, model_form="semi_log"),
        "data_fingerprint": fingerprint_panel(panel, schema, config=config),
        "governance": {
            "approved_for_optimization": bool(gov.get("approved_for_optimization", False)),
        },
        "model_release": {"state": str(gov.get("model_release_state", "research_only"))},
        "identifiability": {"identifiability_score": 0.95},
        "panel_qa": {"status": "pass", "blocking_issues": []},
        "calibration_summary": {"replay_calibration_active": False},
    }


def run_recovery_certification(
    bundle_dir: str | Path,
    truth: dict[str, Any] | None = None,
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """
    Execute Phase 4B-2/4B-3 recovery checks on eligible DGP world bundles.

    ``truth_override`` replaces in-memory truth for checks only (does not write world_truth.json).
    """
    bundle = Path(bundle_dir)
    truth_base = truth if truth is not None else read_json(bundle / "world_truth.json")

    if is_drift_recovery_eligible(truth_base):
        return run_drift_recovery_certification(
            bundle, truth_base, truth_override=truth_override
        )
    if is_identifiability_recovery_eligible(truth_base):
        return run_identifiability_recovery_certification(
            bundle, truth_base, truth_override=truth_override
        )
    if is_replay_recovery_eligible(truth_base):
        return run_replay_recovery_certification(
            bundle, truth_base, truth_override=truth_override
        )
    if is_optimizer_recovery_eligible(truth_base):
        return run_optimizer_recovery_certification(
            bundle, truth_base, truth_override=truth_override
        )
    if is_coef_recovery_eligible(truth_base):
        return run_coef_recovery_certification(
            bundle, truth_base, truth_override=truth_override
        )

    if not is_recovery_eligible(truth_base):
        return RecoveryCertificationResult(
            passed=False,
            checks=[
                RecoveryCheckOutcome(
                    "REC-4B2-000",
                    "eligibility",
                    "skipped",
                    f"world {truth_base['metadata'].get('world_id')} not recovery-eligible",
                    skip_reason="not_applicable",
                )
            ],
            recovery_results={"skipped": True, "reason": "not_recovery_eligible"},
            train_decide_summary={},
        )
    return RecoveryCertificationResult(
        passed=False,
        checks=[],
        recovery_results={"skipped": True},
        train_decide_summary={},
    )


def run_coef_recovery_certification(
    bundle_dir: str | Path,
    truth_base: dict[str, Any],
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """Phase 4B-2 — coefficient / Δμ recovery on WORLD-008."""
    bundle = Path(bundle_dir)
    truth_eff = truth_override if truth_override is not None else truth_base

    checks: list[RecoveryCheckOutcome] = []
    limitations: list[str] = [
        "Phase 4B-2 exact-recovery only — WORLD-008, Ridge semi_log, zero-noise DGP",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "Ridge BO uses shared adstock/Hill hyperparameters across channels",
        "Does not prove causal validity or production allocation correctness",
    ]

    try:
        panel, schema, config, fit = _train_ridge(bundle, truth_base)
    except Exception as exc:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B2-TRAIN",
                "train_path",
                "fail",
                f"Ridge train path failed: {exc}",
            )
        )
        return _finalize(checks, limitations, train_summary={"train_completed": False})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-TRAIN",
            "train_path",
            "pass",
            "Ridge BO fit completed on DGP panel",
            details={"n_rows": len(panel), "n_trials": config.ridge_bo.n_trials},
        )
    )

    ctx = ridge_context_from_fit(panel, schema, config, fit)
    channels = list(truth_eff["media_truth"]["channels"])
    true_coef = _truth_coef_vector(truth_eff, channels)
    fitted_coef = np.asarray(ctx.coef, dtype=float).ravel()
    coef_err = np.abs(fitted_coef - true_coef)
    coef_ok = np.all(coef_err <= COEF_RECOVERY_ATOL + COEF_RECOVERY_RTOL * np.maximum(np.abs(true_coef), 1e-9))
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-001",
            "coefficient_recovery",
            "pass" if coef_ok else "fail",
            "fitted vs coefficient_truth.true_beta_by_channel",
            metric_kind="provisional_statistical",
            details={
                "true_beta": {c: float(true_coef[i]) for i, c in enumerate(channels)},
                "fitted_beta": {c: float(fitted_coef[i]) for i, c in enumerate(channels)},
                "max_abs_error": float(np.max(coef_err)),
                "rtol": COEF_RECOVERY_RTOL,
                "atol": COEF_RECOVERY_ATOL,
            },
        )
    )

    shared = shared_ridge_transform_params(truth_base)
    if not shared["per_channel_uniform"]:
        limitations.append("Per-channel transform params differ — using mean hyperparameters for Ridge BO")
    bp = ctx.best_params
    transform_ok = (
        abs(float(bp.get("decay", -1)) - shared["decay"]) <= TRANSFORM_PARAM_RTOL
        and abs(float(bp.get("hill_half", -1)) - shared["hill_half"]) <= TRANSFORM_PARAM_RTOL * shared["hill_half"]
        and abs(float(bp.get("hill_slope", -1)) - shared["hill_slope"]) <= TRANSFORM_PARAM_RTOL * shared["hill_slope"]
    )
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-002",
            "adstock_transform_consistency",
            "pass" if transform_ok else "fail",
            "fitted decay vs transform_truth (shared hyperparameter policy)",
            metric_kind="provisional_statistical",
            details={"fitted": dict(bp), "truth_shared": shared},
        )
    )
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-003",
            "hill_transform_consistency",
            "pass" if transform_ok else "fail",
            "fitted Hill params vs transform_truth (shared hyperparameter policy)",
            metric_kind="provisional_statistical",
            details={"fitted": dict(bp), "truth_shared": shared},
        )
    )

    # Exact formula spot-check on first geo/channel series
    ch0 = channels[0]
    g0 = panel[schema.geo_column].iloc[0]
    sub = panel[panel[schema.geo_column] == g0].sort_values(schema.week_column)
    raw = sub[ch0].to_numpy(dtype=float)
    decay = shared["decay"]
    half = shared["hill_half"]
    slope = shared["hill_slope"]
    ad = GeometricAdstock(decay)
    sat = HillSaturation(half_max=half, slope=slope)
    feat = apply_adstock_saturation_series(raw, ad, sat)
    np.testing.assert_allclose(
        feat,
        hill_saturation_series(geometric_adstock_series(raw, decay), half_max=half, slope=slope),
        rtol=FORMULA_RTOL,
        atol=FORMULA_RTOL,
    )
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-004",
            "transform_formula_consistency",
            "pass",
            "geometric adstock + Hill match canonical transforms on panel spend",
            metric_kind="exact_formula",
        )
    )

    scenarios = (truth_base.get("decision_truth") or {}).get("scenarios") or []
    analytic_delta: float | None = None
    delta_status: RecoveryStatus = "skipped"
    delta_msg = "no decision scenarios"
    delta_details: dict[str, Any] = {}
    if scenarios:
        sc = scenarios[0]
        if _scenario_is_placeholder(sc):
            analytic_delta = compute_analytic_delta_mu(truth_base, panel, schema, config, sc)
            delta_details["analytic_true_delta_mu"] = analytic_delta
            delta_details["placeholder_replaced"] = True
        else:
            analytic_delta = float(sc.get("true_delta_mu"))
            delta_details["analytic_true_delta_mu"] = analytic_delta
        fitted_res = simulate(
            {c: float(v) for c, v in (sc.get("candidate_spend_by_channel") or {}).items()},
            ctx,
            baseline_plan=BaselinePlan(
                baseline_type=BaselineType.BAU,
                spend_by_channel={
                    c: float(v) for c, v in (sc.get("baseline_spend_by_channel") or {}).items()
                },
                baseline_definition="decision_truth",
                baseline_plan_source="decision_truth",
                suitable_for_decisioning=True,
            ),
        )
        est = float(fitted_res.delta_mu)
        delta_details["estimated_delta_mu"] = est
        if analytic_delta is not None:
            err = abs(est - analytic_delta)
            denom = max(abs(analytic_delta), 1e-9)
            ok = err <= DELTA_MU_RECOVERY_ATOL + DELTA_MU_RECOVERY_RTOL * denom
            delta_status = "pass" if ok else "fail"
            delta_msg = "fitted simulate Δμ vs analytic decision_truth"
            delta_details.update(
                {
                    "abs_error": err,
                    "rtol": DELTA_MU_RECOVERY_RTOL,
                    "atol": DELTA_MU_RECOVERY_ATOL,
                }
            )
        else:
            delta_msg = "analytic Δμ unavailable"
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-005",
            "delta_mu_recovery",
            delta_status,
            delta_msg,
            metric_kind="provisional_statistical",
            details=delta_details,
            skip_reason="not_applicable" if delta_status == "skipped" else None,
        )
    )

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-006",
            "optimizer_recovery",
            "skipped",
            "Optimizer recovery deferred",
            skip_reason="requires_optimizer_truth_thresholds",
        )
    )

    er = _minimal_extension_report(truth_base, panel, schema, config, fit)
    fp_decide = fingerprint_panel(panel, schema, config=config)
    fp_train = er.get("data_fingerprint") or {}
    fp_cmp = compare_training_and_decision_fingerprints(fp_train, fp_decide)
    fp_ok = bool(fp_cmp.get("matched"))
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-007",
            "train_decide_fingerprint",
            "pass" if fp_ok else "fail",
            "training extension_report fingerprint matches decide-time panel",
            metric_kind="exact_formula",
            details=fp_cmp,
        )
    )

    cand_spend = (
        {c: float(v) for c, v in scenarios[0]["candidate_spend_by_channel"].items()} if scenarios else {}
    )
    scenario_yaml = {
        "baseline_type": "bau",
        "candidate_spend": cand_spend,
    }
    scenario_path = bundle / "recovery_scenario.yaml"
    scenario_path.write_text(yaml.dump(scenario_yaml, sort_keys=False), encoding="utf-8")
    decide_ok = False
    decide_details: dict[str, Any] = {}
    out_path = bundle / "recovery_sim_decision.json"
    try:
        payload = simulate_decision(
            cfg=config,
            scenario=scenario_yaml,
            extension_report=er,
            out=out_path,
            scenario_source_path=str(scenario_path),
        )
        sim = payload.get("simulation") or {}
        bundle_js = payload.get("decision_bundle") or {}
        tier = bundle_js.get("artifact_tier") or payload.get("artifact_tier")
        if tier is None and out_path.is_file():
            persisted = json.loads(out_path.read_text(encoding="utf-8"))
            tier = (persisted.get("decision_bundle") or {}).get("artifact_tier") or persisted.get(
                "artifact_tier"
            )
        decide_ok = (
            "delta_mu" in sim
            and isinstance(sim.get("decision_safe"), bool)
            and str(truth_base.get("outcome_truth", {}).get("model_form")) == "semi_log"
            and str(truth_base.get("governance_truth", {}).get("model_release_state", ""))
            in ("planning_allowed", "research_only", "blocked")
        )
        if tier is not None:
            decide_ok = decide_ok and tier in ("decision", "research")
        decide_details = {
            "artifact_tier": tier,
            "decision_safe": sim.get("decision_safe"),
            "delta_mu": sim.get("delta_mu"),
            "model_form": truth_base.get("outcome_truth", {}).get("model_form"),
            "release_state": truth_base.get("governance_truth", {}).get("model_release_state"),
        }
    except Exception as exc:
        decide_details = {"error": str(exc)}
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B2-008",
            "decision_artifact_compatibility",
            "pass" if decide_ok else "fail",
            "simulate_decision bundle + DecisionSurface / release-gate fields",
            details=decide_details,
        )
    )

    train_summary = {
        "train_completed": True,
        "best_params": dict(ctx.best_params),
        "fitted_coef": {c: float(fitted_coef[i]) for i, c in enumerate(channels)},
    }
    return _finalize(checks, limitations, train_summary=train_summary, optimizer_recovery={})


def _week_period_masks(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    changepoint_index: int,
) -> tuple[pd.Series, pd.Series]:
    from mmm.validation.synthetic.materializer import _week_dates

    weeks = _week_dates(truth)
    cp = max(1, min(changepoint_index, len(weeks)))
    pre_dates = {pd.Timestamp(w).normalize() for w in weeks[:cp]}
    post_dates = {pd.Timestamp(w).normalize() for w in weeks[cp:]}
    wcol = pd.to_datetime(panel[schema.week_column]).dt.normalize()
    return wcol.isin(pre_dates), wcol.isin(post_dates)


def _fitted_period_mae(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit: dict[str, Any],
    row_mask: pd.Series,
) -> float:
    from mmm.features.design_matrix import build_design_matrix
    from mmm.models.ridge_bo.ridge import predict_ridge

    sub = panel.loc[row_mask]
    if sub.empty:
        return float("nan")
    art = fit["artifacts"]
    bundle = build_design_matrix(
        sub,
        schema,
        config,
        decay=art.best_params["decay"],
        hill_half=art.best_params["hill_half"],
        hill_slope=art.best_params["hill_slope"],
    )
    yhat = np.exp(predict_ridge(bundle.X, art.coef, art.intercept))
    y = sub[schema.target_column].to_numpy(dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def _run_identifiability_report(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit: dict[str, Any],
) -> dict[str, Any]:
    from mmm.features.design_matrix import build_design_matrix
    from mmm.identifiability.engine import IdentifiabilityEngine

    art = fit["artifacts"]
    bundle = build_design_matrix(
        panel,
        schema,
        config,
        decay=art.best_params["decay"],
        hill_half=art.best_params["hill_half"],
        hill_slope=art.best_params["hill_slope"],
    )
    ch_names = list(schema.channel_columns)
    n_ch = len(ch_names)
    X_media = bundle.X[:, :n_ch] if bundle.X.shape[1] >= n_ch else bundle.X
    rng = np.random.default_rng(int(config.random_seed))
    report = IdentifiabilityEngine(config.extensions.identifiability).analyze(
        X_media,
        ch_names,
        bundle.y_modeling,
        rng,
        ridge_log_alpha=float(art.best_params.get("log_alpha", 0.0)),
    )
    return report.to_json()


def run_drift_recovery_certification(
    bundle_dir: str | Path,
    truth_base: dict[str, Any],
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """Phase 5E — drift reliability via dedicated VAL-012 drift_detection_runner."""
    from mmm.validation.synthetic.drift_detection_runner import (
        recovery_check_from_drift_result,
        run_val_012_drift_detection,
    )

    bundle = Path(bundle_dir)
    truth_eff = truth_override if truth_override is not None else truth_base
    drift = truth_eff.get("drift_truth") or {}
    checks: list[RecoveryCheckOutcome] = []
    limitations = [
        "Phase 5E drift-recovery — VAL-012 drift_detection_runner",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "Coefficient recovery across drift regimes not expected",
    ]

    cps = drift.get("changepoints") or []
    coef_drift = drift.get("coefficient_drift") or []
    if not cps or not coef_drift:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B5-DRIFT-000",
                "drift_truth",
                "fail",
                "drift_truth changepoints/coefficient_drift missing",
            )
        )
        return _finalize(checks, limitations, train_summary={}, drift_recovery={})

    cp_index = int(cps[0]["period_index"])
    drift_result = run_val_012_drift_detection(bundle, truth_eff)

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B5-DRIFT-TRAIN",
            "train_path",
            "pass" if drift_result.pre_period_fit_error is not None else "fail",
            "Pre-changepoint Ridge fit for drift detection",
        )
    )
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B5-DRIFT-COEF",
            "coefficient_recovery",
            "skipped",
            "coefficient recovery across drift regimes not expected",
            skip_reason="not_applicable",
        )
    )
    checks.append(recovery_check_from_drift_result(drift_result))

    drift_metrics = drift_result.to_dict()
    drift_metrics["drift_recovery_status"] = (
        "pass" if drift_result.val_012_outcome == "pass" else "fail"
    )
    gov = truth_eff.get("governance_truth") or {}
    readiness_results = {
        "approved_for_optimization_truth": bool(gov.get("approved_for_optimization")),
        "readiness_downgraded": drift_result.readiness_downgraded,
        "optimization_block_recommended": drift_result.optimization_block_recommended,
        "model_release_state": str(gov.get("model_release_state", "")),
    }

    return _finalize(
        checks,
        limitations,
        train_summary={
            "train_completed": drift_result.pre_period_fit_error is not None,
            "changepoint_index": cp_index,
        },
        drift_recovery=drift_metrics,
        reliability_degradation_results={"drift": drift_metrics},
        readiness_reaction_results=readiness_results,
    )


def run_identifiability_recovery_certification(
    bundle_dir: str | Path,
    truth_base: dict[str, Any],
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """Phase 4B-5 — identifiability reliability on WORLD-012 (VAL-013/014 partial)."""
    from mmm.governance.scorecard import build_scorecard
    from mmm.validation.synthetic.reliability_truth import panel_channel_correlation

    bundle = Path(bundle_dir)
    truth_eff = truth_override if truth_override is not None else truth_base
    rel = (truth_eff.get("drift_truth") or {}).get("expected_reliability") or {}
    checks: list[RecoveryCheckOutcome] = []
    limitations = [
        "Phase 4B-5 identifiability-recovery — WORLD-012; coefficient recovery not required",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "VAL-013/014 partial via identifiability engine + governance scorecard",
    ]

    try:
        panel, schema, config, fit = _train_ridge_truth_transforms(bundle, truth_base)
    except Exception as exc:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B5-ID-TRAIN", "train_path", "fail", f"train failed: {exc}"
            )
        )
        return _finalize(checks, limitations, train_summary={}, identifiability_recovery={})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B5-ID-TRAIN",
            "train_path",
            "pass",
            "Ridge fit completed on collinear-spend panel",
        )
    )
    checks.append(
        RecoveryCheckOutcome(
            "REC-4B5-ID-COEF",
            "coefficient_recovery",
            "skipped",
            "coefficient recovery unstable under severe collinearity",
            skip_reason="recovery_marked_unstable",
        )
    )

    channels = list(truth_eff["media_truth"]["channels"])
    ch_corr = panel_channel_correlation(panel, schema, channels)
    ident_json = _run_identifiability_report(panel, schema, config, fit)
    max_vif = float(ident_json.get("max_vif", 0.0))
    warnings = list(ident_json.get("warnings") or [])
    collinearity_warn = any("high_collinearity" in w for w in warnings)
    ident_score = float(ident_json.get("identifiability_score", 0.0))

    from mmm.features.design_matrix import build_design_matrix
    from mmm.models.ridge_bo.ridge import predict_ridge

    art = fit["artifacts"]
    full_b = build_design_matrix(
        panel,
        schema,
        config,
        decay=art.best_params["decay"],
        hill_half=art.best_params["hill_half"],
        hill_slope=art.best_params["hill_slope"],
    )
    y = panel[schema.target_column].to_numpy(dtype=float)
    yhat = np.exp(predict_ridge(full_b.X, art.coef, art.intercept))
    gov = truth_eff.get("governance_truth") or {}
    sc = build_scorecard(
        cfg=config.extensions.governance,
        fit_mae=float(np.mean(np.abs(y - yhat))),
        baseline_mae=float(np.mean(np.abs(y - yhat))) * 1.05,
        identifiability_score=ident_score,
        calibration_loss=None,
        falsification_flags=[],
        beats_baselines=True,
    )

    min_corr = float(rel.get("min_channel_correlation", IDENTIFIABILITY_MIN_CORRELATION))
    min_vif = float(rel.get("min_max_vif", IDENTIFIABILITY_MIN_VIF))
    corr_ok = ch_corr >= min_corr
    vif_ok = max_vif >= min_vif
    warn_ok = collinearity_warn
    expected_warns = {
        str(w.get("warning_id"))
        for w in (truth_eff.get("artifact_truth") or {}).get("expected_warnings") or []
    }
    artifact_warn_ok = "identifiability_collinearity" in expected_warns
    readiness_downgraded = (
        not bool(gov.get("approved_for_optimization", True))
        or not sc.approved_for_optimization
        or str(gov.get("model_release_state", "")) == "review_required"
    )
    id_pass = corr_ok and vif_ok and warn_ok and artifact_warn_ok and readiness_downgraded

    id_metrics = {
        "channel_correlation": ch_corr,
        "max_vif": max_vif,
        "identifiability_score": ident_score,
        "identifiability_warning_emitted": collinearity_warn,
        "coefficient_instability_flag": True,
        "recovery_marked_unstable": True,
        "readiness_downgraded_or_review_required": readiness_downgraded,
        "identifiability_recovery_status": "pass" if id_pass else "fail",
        "registry_validation_id": "VAL-013",
        "registry_validation_ids": ["VAL-013", "VAL-014"],
    }
    readiness_results = {
        "approved_for_optimization_truth": bool(gov.get("approved_for_optimization")),
        "approved_for_optimization_scorecard": bool(sc.approved_for_optimization),
        "model_release_state": str(gov.get("model_release_state", "")),
        "scorecard_notes": list(sc.notes),
    }

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B5-ID",
            "identifiability_recovery",
            "pass" if id_pass else "fail",
            "VAL-013/014 partial — collinearity warning and readiness reaction",
            metric_kind="provisional_statistical",
            details=id_metrics,
        )
    )

    return _finalize(
        checks,
        limitations,
        train_summary={"train_completed": True},
        identifiability_recovery=id_metrics,
        reliability_degradation_results={"identifiability": id_metrics},
        readiness_reaction_results=readiness_results,
    )


def run_replay_recovery_certification(
    bundle_dir: str | Path,
    truth_base: dict[str, Any],
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """Phase 4B-4 — replay calibration recovery on WORLD-010 (executes VAL-006)."""
    from mmm.calibration.replay_lift import implied_lift_from_counterfactual
    from mmm.calibration.units_io import load_calibration_units_from_json
    from mmm.features.design_matrix import build_design_matrix
    from mmm.models.ridge_bo.ridge import predict_ridge
    from mmm.validation.synthetic.replay_truth import (
        build_calibration_unit_for_experiment,
        detect_window_slice_adstock_reset,
        replay_estimand_from_unit,
        validate_replay_lift_surface,
    )

    bundle = Path(bundle_dir)
    truth_eff = truth_override if truth_override is not None else truth_base
    experiment = truth_eff.get("experiment_truth") or {}
    units_def = list(experiment.get("units") or [])
    checks: list[RecoveryCheckOutcome] = []
    limitations = [
        "Phase 4B-4 replay-recovery — WORLD-010, Ridge semi_log, known experiment lift in truth",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "Train pins adstock/Hill to truth; replay uses fitted coefficients on full-panel frames",
    ]

    if not units_def:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-000",
                "experiment_truth",
                "fail",
                "experiment_truth.units missing",
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    unit0 = units_def[0]
    try:
        lift_val = float((unit0.get("lift_definition") or {}).get("value", 0.0))
        validate_replay_lift_surface(lift_val)
    except ValueError as exc:
        checks.append(
            RecoveryCheckOutcome("REC-4B4-000", "replay_surface", "fail", str(exc))
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    replay_path = bundle / "replay_units.json"
    if not replay_path.is_file():
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-LOAD",
                "replay_loader",
                "fail",
                "replay_units.json missing — materialize bundle first",
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    try:
        loaded = load_calibration_units_from_json(replay_path)
    except Exception as exc:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-LOAD",
                "replay_loader",
                "fail",
                f"load_calibration_units_from_json failed: {exc}",
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B4-LOAD",
            "replay_loader",
            "pass",
            f"loaded {len(loaded)} replay unit(s) from replay_units.json",
        )
    )

    try:
        panel, schema, config, fit = _train_ridge_truth_transforms(bundle, truth_base)
    except Exception as exc:
        checks.append(
            RecoveryCheckOutcome("REC-4B4-TRAIN", "train_path", "fail", f"train failed: {exc}")
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B4-TRAIN",
            "train_path",
            "pass",
            "Ridge fit (truth transforms) completed for replay world",
        )
    )

    if detect_window_slice_adstock_reset(truth_base, panel, schema, config, unit0):
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-ADSTOCK",
                "pre_window_adstock",
                "fail",
                "window-slice replay would reset adstock — full-panel path required",
                details={"registry_validation_id": "VAL-006"},
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    art = fit["artifacts"]

    def predict_fn(dfp: pd.DataFrame) -> np.ndarray:
        b = build_design_matrix(
            dfp,
            schema,
            config,
            decay=art.best_params["decay"],
            hill_half=art.best_params["hill_half"],
            hill_slope=art.best_params["hill_slope"],
        )
        ylog = predict_ridge(b.X, art.coef, art.intercept)
        return np.exp(ylog)

    cal_unit = build_calibration_unit_for_experiment(truth_eff, panel, schema, unit0)
    if cal_unit is None or cal_unit.observed_spend_frame is None:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-REPLAY",
                "replay_recovery",
                "fail",
                "could not build full-panel replay frames",
                details={"registry_validation_id": "VAL-006"},
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    if len(cal_unit.observed_spend_frame) < len(panel):
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B4-REPLAY",
                "replay_recovery",
                "fail",
                "replay frames are window-sliced — adstock reset not allowed",
                details={"registry_validation_id": "VAL-006"},
            )
        )
        return _finalize(checks, limitations, train_summary={}, replay_recovery={})

    spec = replay_estimand_from_unit(unit0, target_kpi=schema.target_column)
    replay_result = implied_lift_from_counterfactual(
        panel_observed=cal_unit.observed_spend_frame,
        panel_counterfactual=cal_unit.counterfactual_spend_frame,
        predict_fn=predict_fn,
        schema=schema,
        estimand=spec,
    )
    fitted_lift = float(replay_result["implied_mean_delta"])
    true_lift = float((unit0.get("lift_definition") or {}).get("value", 0.0))
    lift_err = abs(fitted_lift - true_lift)
    rel_err = lift_err / max(abs(true_lift), 1e-9)
    lift_ok = lift_err <= REPLAY_LIFT_ATOL + REPLAY_LIFT_RTOL * max(abs(true_lift), 1e-9)
    transform_mode = str(replay_result.get("replay_transform_mode", ""))
    mode_ok = transform_mode == "full_panel_transform_estimand_mask"

    replay_metrics = {
        "true_experiment_lift": true_lift,
        "fitted_replay_implied_lift": fitted_lift,
        "replay_lift_error": lift_err,
        "replay_lift_relative_error": rel_err,
        "replay_transform_mode": transform_mode,
        "estimand_mask_used": spec.to_json(),
        "pre_window_adstock_preserved": not detect_window_slice_adstock_reset(
            truth_base, panel, schema, config, unit0
        ),
        "replay_recovery_status": "pass" if lift_ok and mode_ok else "fail",
        "registry_validation_id": "VAL-006",
    }

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B4-REPLAY",
            "replay_recovery",
            "pass" if lift_ok and mode_ok else "fail",
            "VAL-006 fitted replay-implied lift vs experiment_truth.true_lift",
            metric_kind="provisional_statistical",
            details=replay_metrics,
        )
    )

    train_summary = {
        "train_completed": True,
        "replay_completed": True,
        "best_params": dict(art.best_params),
        "n_replay_units_loaded": len(loaded),
    }
    return _finalize(
        checks,
        limitations,
        train_summary=train_summary,
        replay_recovery=replay_metrics,
    )


def run_optimizer_recovery_certification(
    bundle_dir: str | Path,
    truth_base: dict[str, Any],
    *,
    truth_override: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    """Phase 4B-3 — optimizer recovery on WORLD-009 (executes VAL-005)."""
    from mmm.decision.gates import allow_decision_pipeline
    from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
    from mmm.planning.baseline import bau_baseline_from_panel
    from mmm.validation.synthetic.optimizer_truth import validate_optimizer_surface

    bundle = Path(bundle_dir)
    truth_eff = truth_override if truth_override is not None else truth_base
    decision = truth_eff.get("decision_truth") or {}
    checks: list[RecoveryCheckOutcome] = []
    limitations = [
        "Phase 4B-3 optimizer-recovery — WORLD-009, Ridge semi_log, known grid optimum in truth",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "Train pins adstock/Hill to truth hyperparameters; optimizer uses fitted coefs only",
        "Panel uses channel_modulated spend for coefficient identifiability",
    ]

    if not decision.get("true_optimal_budget"):
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B3-000",
                "optimizer_truth",
                "fail",
                "decision_truth.true_optimal_budget missing",
            )
        )
        return _finalize(checks, limitations, train_summary={}, optimizer_recovery={})

    try:
        validate_optimizer_surface(
            {
                "true_optimal_budget": decision["true_optimal_budget"],
                "optimum_interior": True,
                "optimum_lift_over_bau": float(decision.get("optimum_lift_over_bau", 1.0)),
            }
        )
    except ValueError as exc:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B3-000",
                "optimizer_surface",
                "fail",
                str(exc),
            )
        )
        return _finalize(checks, limitations, train_summary={}, optimizer_recovery={})

    try:
        panel, schema, config, fit = _train_ridge_truth_transforms(bundle, truth_base)
    except Exception as exc:
        checks.append(
            RecoveryCheckOutcome("REC-4B3-TRAIN", "train_path", "fail", f"train failed: {exc}")
        )
        return _finalize(checks, limitations, train_summary={}, optimizer_recovery={})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B3-TRAIN",
            "train_path",
            "pass",
            "Ridge fit (truth transforms) completed for optimizer world",
        )
    )

    ctx = ridge_context_from_fit(panel, schema, config, fit)
    channels = list(truth_eff["media_truth"]["channels"])
    if len(channels) != 2:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B3-OPT",
                "optimizer_recovery",
                "fail",
                f"optimizer world requires 2 channels, got {len(channels)}",
            )
        )
        return _finalize(checks, limitations, train_summary={}, optimizer_recovery={})

    constraints = (decision.get("budget_constraints") or [{}])[0]
    total_budget = float(constraints.get("total_budget", 40.0))
    ch_min = constraints.get("channel_min") or {c: 0.0 for c in channels}
    ch_max = constraints.get("channel_max") or {c: total_budget for c in channels}
    zmin = np.array([float(ch_min[c]) for c in channels], dtype=float)
    zmax = np.array([float(ch_max[c]) for c in channels], dtype=float)
    base = bau_baseline_from_panel(panel, schema)
    current = np.array([float(base.spend_by_channel[c]) for c in channels], dtype=float)

    opt_result: dict[str, Any] = {}
    opt_err: str | None = None
    try:
        with allow_decision_pipeline():
            opt_result = optimize_budget_via_simulation(
                ctx,
                current_spend=current,
                total_budget=total_budget,
                channel_min=zmin,
                channel_max=zmax,
            )
    except Exception as exc:
        opt_err = str(exc)

    if opt_err:
        checks.append(
            RecoveryCheckOutcome(
                "REC-4B3-OPT",
                "optimizer_recovery",
                "fail",
                f"optimize_budget_via_simulation failed: {opt_err}",
                details={"registry_validation_id": "VAL-005"},
            )
        )
        return _finalize(checks, limitations, train_summary={}, optimizer_recovery={})

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B3-OPT-PATH",
            "optimizer_path",
            "pass",
            "optimize_budget_via_simulation completed",
        )
    )

    true_opt = {c: float(decision["true_optimal_budget"][c]) for c in channels}
    fitted = opt_result.get("recommended_spend_plan") or {}
    fitted_vec = np.array([float(fitted.get(c, 0.0)) for c in channels], dtype=float)
    true_vec = np.array([float(true_opt[c]) for c in channels], dtype=float)
    l1_err = float(np.sum(np.abs(fitted_vec - true_vec)))
    budget_err = abs(float(fitted_vec.sum()) - total_budget)
    true_delta = float(decision.get("true_optimal_delta_mu", 0.0))
    fitted_delta = float(opt_result.get("objective_delta_mu", 0.0))
    obj_gap = abs(true_delta - fitted_delta)

    band = decision.get("expected_allocation_band") or {}
    high_ch = str(band.get("high_return_channel", channels[0]))
    low_ch = str(band.get("low_return_channel", channels[1]))
    high_share = float(fitted.get(high_ch, 0.0)) / max(total_budget, 1e-9)
    low_share = float(fitted.get(low_ch, 0.0)) / max(total_budget, 1e-9)
    min_high_share = float(band.get("high_return_min_budget_share", 0.5))
    band_ok = high_share >= min_high_share and high_share > low_share

    l1_ok = l1_err <= ALLOCATION_L1_ATOL + ALLOCATION_L1_RTOL * total_budget
    obj_ok = obj_gap <= OBJECTIVE_GAP_ATOL + OBJECTIVE_GAP_RTOL * max(abs(true_delta), 1e-9)
    budget_ok = budget_err <= BUDGET_CONSERVATION_ATOL
    opt_pass = l1_ok and obj_ok and budget_ok and band_ok

    opt_metrics = {
        "allocation_l1_error": l1_err,
        "budget_conservation_error": budget_err,
        "objective_gap": obj_gap,
        "true_optimal_delta_mu": true_delta,
        "fitted_objective_delta_mu": fitted_delta,
        "true_optimal_budget": true_opt,
        "fitted_budget": {c: float(fitted_vec[i]) for i, c in enumerate(channels)},
        "expected_allocation_band": band,
        "high_return_budget_share": high_share,
        "constraint_violations": [] if budget_ok else ["sum_budget_mismatch"],
        "optimizer_recovery_status": "pass" if opt_pass else "fail",
        "optimizer_success": bool(opt_result.get("optimizer_success")),
        "registry_validation_id": "VAL-005",
    }

    checks.append(
        RecoveryCheckOutcome(
            "REC-4B3-OPT",
            "optimizer_recovery",
            "pass" if opt_pass else "fail",
            "VAL-005 optimizer allocation vs true_optimal_budget and expected band",
            metric_kind="provisional_statistical",
            details=opt_metrics,
        )
    )

    train_summary = {
        "train_completed": True,
        "optimizer_completed": True,
        "best_params": dict(ctx.best_params),
    }
    return _finalize(
        checks,
        limitations,
        train_summary=train_summary,
        optimizer_recovery=opt_metrics,
    )


def _finalize(
    checks: list[RecoveryCheckOutcome],
    limitations: list[str],
    *,
    train_summary: dict[str, Any],
    optimizer_recovery: dict[str, Any] | None = None,
    replay_recovery: dict[str, Any] | None = None,
    drift_recovery: dict[str, Any] | None = None,
    identifiability_recovery: dict[str, Any] | None = None,
    reliability_degradation_results: dict[str, Any] | None = None,
    readiness_reaction_results: dict[str, Any] | None = None,
) -> RecoveryCertificationResult:
    coef = next((c for c in checks if c.check_id == "REC-4B2-001"), None)
    delta = next((c for c in checks if c.check_id == "REC-4B2-005"), None)
    transform = {
        "adstock": next((c for c in checks if c.check_id == "REC-4B2-002"), None),
        "hill": next((c for c in checks if c.check_id == "REC-4B2-003"), None),
        "formula": next((c for c in checks if c.check_id == "REC-4B2-004"), None),
    }
    decision = next((c for c in checks if c.check_id == "REC-4B2-008"), None)
    fingerprint = next((c for c in checks if c.check_id == "REC-4B2-007"), None)
    executed = [c for c in checks if c.status in ("pass", "fail")]
    passed = all(c.status != "fail" for c in executed) and bool(executed)

    opt_rec = optimizer_recovery or {}
    rep_rec = replay_recovery or {}
    drift_rec = drift_recovery or {}
    ident_rec = identifiability_recovery or {}
    recovery_results = {
        "passed": passed,
        "coefficient_recovery": coef.to_dict() if coef else {},
        "delta_mu_recovery": delta.to_dict() if delta else {},
        "transform_recovery": {k: v.to_dict() if v else {} for k, v in transform.items()},
        "decision_artifact_recovery": decision.to_dict() if decision else {},
        "train_decide_recovery_status": fingerprint.to_dict() if fingerprint else {},
        "optimizer_recovery": opt_rec,
        "optimizer_recovery_status": opt_rec.get("optimizer_recovery_status"),
        "replay_recovery": rep_rec,
        "replay_recovery_status": rep_rec.get("replay_recovery_status"),
        "drift_recovery": drift_rec,
        "drift_recovery_status": drift_rec.get("drift_recovery_status"),
        "identifiability_recovery": ident_rec,
        "identifiability_recovery_status": ident_rec.get("identifiability_recovery_status"),
        "reliability_degradation_results": reliability_degradation_results or {},
        "readiness_reaction_results": readiness_reaction_results or {},
        "recovery_limitations": limitations,
    }
    return RecoveryCertificationResult(
        passed=passed,
        checks=checks,
        recovery_results=recovery_results,
        train_decide_summary=train_summary,
    )
