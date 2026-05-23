"""Ridge + Bayesian Optimization trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.evidence_replay import (
    build_evidence_weighted_replay_summary,
    prepare_evidence_replay,
    uses_evidence_registry_replay,
    uses_weighted_evidence_replay,
    validate_evidence_registry_replay_config,
)
from mmm.calibration.replay_bo_objective import evaluate_replay_calibration_for_trial
from mmm.calibration.replay_frames import normalize_replay_units_to_full_panel
from mmm.calibration.replay_prod_gate import assert_replay_production_ready
from mmm.calibration.replay_refit_mode import replay_refit_enters_objective, validate_replay_refit_mode
from mmm.calibration.replay_units_resolve import resolve_replay_unit_sets
from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.panel_qa import assert_panel_qa_allows_training
from mmm.data.schema import PanelSchema, validate_panel
from mmm.features.design_matrix import apply_masks_for_fit, build_design_matrix
from mmm.hierarchy.diagnostics import hierarchy_enabled
from mmm.hierarchy.hierarchy_extension import load_and_validate_hierarchy
from mmm.hierarchy.penalty import hierarchical_penalty
from mmm.models.base import RidgeBOMMMBase
from mmm.models.ridge_bo.objective import build_composite, intercept_only_predictive_baseline
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
from mmm.performance.cache import _df_fingerprint
from mmm.validation.cv import auto_cv_mode


@dataclass
class RidgeBOArtifacts:
    best_params: dict[str, float]
    objective_history: list[dict[str, Any]]
    coef: np.ndarray
    intercept: np.ndarray
    leaderboard: list[dict[str, Any]]


class RidgeBOMMMTrainer(RidgeBOMMMBase):
    """Time-series CV + Optuna over adstock/saturation/ridge alpha."""

    def __init__(self, config: MMMConfig, schema: PanelSchema) -> None:
        from mmm.contracts.seed_resolution import resolve_seed_contract

        resolve_seed_contract(config)
        if config.framework != Framework.RIDGE_BO:
            raise ValueError("RidgeBOMMMTrainer requires framework=ridge_bo")
        self.config = config
        self.schema = schema
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._best_params: dict[str, float] = {}
        self._artifacts: RidgeBOArtifacts | None = None
        self._replay_units: list = []
        self._replay_units_holdout: list = []
        self._legacy_replay_warnings: list[str] = []
        self._replay_split_meta: dict[str, Any] = {}
        self._evidence_replay_prepared = None
        self._hierarchy_pairs: list = []
        self._hierarchy_penalty_warnings: list[str] = []
        validate_evidence_registry_replay_config(config)
        if hierarchy_enabled(config) and not config.hierarchy.hierarchy_definition_path:
            raise ValueError("hierarchy.enabled requires hierarchy.hierarchy_definition_path")
        if uses_weighted_evidence_replay(config):
            pass
        elif config.calibration.use_replay_calibration:
            train_u, hold_u, split_meta = resolve_replay_unit_sets(config, schema)
            self._replay_units = train_u
            self._replay_units_holdout = hold_u
            self._replay_split_meta = split_meta

    @property
    def _replay_units_train(self) -> list:
        """Alias for train replay units (main holdout-split API)."""
        return self._replay_units

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        df = validate_panel(
            df,
            self.schema,
            integrity_qa=self.config.run_environment == RunEnvironment.PROD,
            calendar_strict=self.config.run_environment == RunEnvironment.PROD,
        )
        df = sort_panel_for_modeling(df, self.schema)
        assert_panel_qa_allows_training(df, self.schema, self.config)
        if self._replay_units:
            self._replay_units, self._legacy_replay_warnings = normalize_replay_units_to_full_panel(
                df, self.schema, self._replay_units
            )
        if self._replay_units_holdout:
            self._replay_units_holdout, hold_warn = normalize_replay_units_to_full_panel(
                df, self.schema, self._replay_units_holdout
            )
            self._legacy_replay_warnings.extend(hold_warn)
        cv = auto_cv_mode(df, self.schema, self.config.cv)
        splits = cv.split(df, self.schema)
        if not splits:
            raise RuntimeError("CV produced no splits; adjust min_train_weeks/horizon")

        has_replay = self.config.calibration.use_replay_calibration and (
            uses_weighted_evidence_replay(self.config) or bool(self._replay_units)
        )
        if (
            self.config.run_environment == RunEnvironment.PROD
            and self.config.calibration.enabled
            and not (self.config.calibration.use_replay_calibration and has_replay)
        ):
            raise ValueError(
                "Production Ridge BO: calibration.enabled requires calibration.use_replay_calibration "
                "with replay_units_path (legacy) or evidence_registry weighted replay configured."
            )

        objective_history: list[dict[str, Any]] = []
        leaderboard: list[dict[str, Any]] = []
        design_bundle_cache: dict[tuple[Any, ...], Any] = {}
        replay_loss_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
        if hierarchy_enabled(self.config):
            _def, _rep, self._hierarchy_pairs, self._hierarchy_penalty_warnings = load_and_validate_hierarchy(
                self.config, self.schema, df
            )

        if uses_weighted_evidence_replay(self.config):
            self._evidence_replay_prepared = prepare_evidence_replay(self.config, df, self.schema)
            if not self._evidence_replay_prepared.used:
                raise ValueError(
                    "evidence_registry weighted replay: no compatible evidence units after filtering; "
                    f"rejected={self._evidence_replay_prepared.rejected!r}"
                )
            if uses_evidence_registry_replay(self.config):
                summary = build_evidence_weighted_replay_summary(
                    self._evidence_replay_prepared,
                    {"replay_mode_used": "evidence_registry"},
                )
                assert_replay_production_ready(
                    self.config,
                    [e.unit for e in self._evidence_replay_prepared.used],
                    schema=self.schema,
                    evidence_summary=summary,
                )

        def evaluate_trial(params: dict[str, float]) -> tuple[float, dict[str, Any]]:
            t_wall0 = time.perf_counter()
            decay = params["decay"]
            hill_half = params["hill_half"]
            hill_slope = params["hill_slope"]
            log_alpha = params["log_alpha"]
            alpha = float(10**log_alpha)
            coef_rows: list[np.ndarray] = []
            intercept_rows: list[np.ndarray] = []
            y_true_folds: list[np.ndarray] = []
            y_pred_folds: list[np.ndarray] = []
            dkey = (
                _df_fingerprint(df, self.schema),
                round(float(decay), 9),
                round(float(hill_half), 9),
                round(float(hill_slope), 9),
            )
            if dkey not in design_bundle_cache:
                design_bundle_cache[dkey] = build_design_matrix(
                    df,
                    self.schema,
                    self.config,
                    decay=decay,
                    hill_half=hill_half,
                    hill_slope=hill_slope,
                )
            bundle = design_bundle_cache[dkey]
            for train_mask, val_mask in splits:
                if len(train_mask) != len(bundle.df_aligned):
                    raise RuntimeError("CV masks length mismatch; ensure panel is sorted by geo, week")
                X_tr, y_tr_t = apply_masks_for_fit(bundle, train_mask)
                coef, intercept = fit_ridge(X_tr, y_tr_t, alpha)
                coef_rows.append(coef)
                intercept_rows.append(intercept)
                X_va = bundle.X[val_mask]
                y_va_t = bundle.y_modeling[val_mask]
                yhat_va = predict_ridge(X_va, coef, intercept)
                y_true_folds.append(y_va_t)
                y_pred_folds.append(yhat_va)
            coef_mat = np.vstack(coef_rows) if coef_rows else np.zeros((0, 0))
            cal_detail: dict[str, Any] | None = None
            cal = 0.0
            parts: dict[str, Any] = {}
            refit_mode = validate_replay_refit_mode(self.config.calibration.replay_refit_mode)
            replay_active = (
                self.config.calibration.use_replay_calibration
                and (self._replay_units or self._evidence_replay_prepared is not None)
                and coef_rows
                and intercept_rows
            )
            if replay_active:
                rkey = (dkey, round(float(alpha), 9), refit_mode)
                if rkey not in replay_loss_cache:
                    out = evaluate_replay_calibration_for_trial(
                        panel_df=df,
                        schema=self.schema,
                        config=self.config,
                        bundle=bundle,
                        splits=splits,
                        coef_rows=coef_rows,
                        intercept_rows=intercept_rows,
                        replay_units=self._replay_units,
                        evidence_prepared=self._evidence_replay_prepared,
                        legacy_warnings=self._legacy_replay_warnings,
                        replay_split_meta=self._replay_split_meta,
                        refit_mode=refit_mode,
                        decay=decay,
                        hill_half=hill_half,
                        hill_slope=hill_slope,
                        alpha=alpha,
                    )
                    if out is None:
                        replay_loss_cache[rkey] = (0.0, {})
                    else:
                        obj_loss, merged, _ = out
                        replay_loss_cache[rkey] = (float(obj_loss), merged)
                rloss, rmeta = replay_loss_cache[rkey]
                cal = float(rloss) if replay_refit_enters_objective(refit_mode, use_replay_calibration=True) else 0.0
                parts["replay"] = rmeta
                parts["mean_lift_se"] = rmeta.get("mean_lift_se", 1.0)
                for key in (
                    "calibration_refit_mode",
                    "replay_refit_mode",
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
                    "replay_mode_used",
                    "predictive_score_source",
                    "calibration_score_source",
                    "weighted_replay_loss",
                    "calibration_refit_n_rows",
                    "train_vs_holdout_replay_loss",
                    "legacy_replay_upgrade_warnings",
                    "fold_replay_losses",
                    "fold_replay_units_used",
                    "fold_replay_units_skipped",
                    "replay_fold_alignment_warnings",
                ):
                    if key in rmeta:
                        parts[key] = rmeta[key]
            if parts:
                parts["loss"] = float(cal)
                cal_detail = parts
            hier_pen = 0.0
            if self._hierarchy_pairs and coef_rows:
                n_media = len(self.schema.channel_columns)
                coef_last = coef_mat[-1, :n_media]
                hier_pen, hier_meta = hierarchical_penalty(
                    coef_last,
                    self._hierarchy_pairs,
                    regularization_strength=self.config.hierarchy.regularization_strength,
                )
                if cal_detail is None:
                    cal_detail = {}
                cal_detail["hierarchical_penalty"] = float(hier_pen)
                cal_detail["hierarchy_penalty_meta"] = hier_meta
                cal_detail["loss"] = float(cal_detail.get("loss", 0.0)) + float(hier_pen)
                cal_detail["hierarchy_enabled"] = True
                if self._hierarchy_penalty_warnings:
                    cal_detail["hierarchy_penalty_warnings"] = list(self._hierarchy_penalty_warnings)
            elif hierarchy_enabled(self.config) and not self._hierarchy_pairs:
                if cal_detail is None:
                    cal_detail = {}
                cal_detail["hierarchy_enabled"] = True
                cal_detail["hierarchical_penalty"] = 0.0
            baseline_pred = intercept_only_predictive_baseline(
                y_true_folds,
                self.config.objective.primary_metric,
            )
            total, raw, norm, norm_report = build_composite(
                y_true_folds=y_true_folds,
                y_pred_folds=y_pred_folds,
                metric=self.config.objective.primary_metric,
                coef_mat=coef_mat[:, : len(self.schema.channel_columns)],
                decay=decay,
                hill_half=hill_half,
                hill_slope=hill_slope,
                log_alpha=log_alpha,
                calibration_details=cal_detail,
                cfg=self.config.objective,
                baseline_predictive=baseline_pred,
                include_weight_sensitivity=True,
            )
            detail = {
                "total": total,
                "raw": raw.as_dict(),
                "normalized": norm.as_dict(),
                "objective_normalization": norm_report,
                "normalization_profile": self.config.objective.normalization_profile.value,
                "params": params,
                "evaluate_wall_time_ms": float((time.perf_counter() - t_wall0) * 1000.0),
            }
            if isinstance(cal_detail, dict):
                for key in (
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
                    "train_vs_holdout_replay_loss",
                    "predictive_score_source",
                    "calibration_score_source",
                    "replay_mode_used",
                    "replay_transform_mode",
                    "legacy_replay_upgrade_warnings",
                    "legacy_replay_warnings",
                ):
                    if key in cal_detail:
                        detail[key] = cal_detail[key]
            objective_history.append(detail)
            leaderboard.append(detail)
            return total, detail

        best_score = float("inf")
        best_detail: dict[str, Any] | None = None
        best_params = {
            "decay": 0.5,
            "hill_half": 1.0,
            "hill_slope": 2.0,
            "log_alpha": 0.0,
        }

        def run_grid() -> None:
            nonlocal best_score, best_detail, best_params
            rng = np.random.default_rng(self.config.ridge_bo.sampler_seed)
            for _ in range(self.config.ridge_bo.n_trials):
                params = {
                    "decay": float(rng.uniform(0.1, 0.9)),
                    "hill_half": float(rng.uniform(0.3, 3.0)),
                    "hill_slope": float(rng.uniform(0.8, 4.0)),
                    "log_alpha": float(rng.uniform(-3, 3)),
                }
                score, detail = evaluate_trial(params)
                if score < best_score:
                    best_score = score
                    best_detail = detail
                    best_params = params

        used_optuna = False
        try:
            import optuna

            sampler = optuna.samplers.TPESampler(seed=self.config.ridge_bo.sampler_seed)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "decay": trial.suggest_float("decay", 0.05, 0.95),
                    "hill_half": trial.suggest_float("hill_half", 0.2, 5.0),
                    "hill_slope": trial.suggest_float("hill_slope", 0.5, 6.0),
                    "log_alpha": trial.suggest_float("log_alpha", -4.0, 3.0),
                }
                score, detail = evaluate_trial(params)
                trial.set_user_attr("decomposition", detail)
                return score

            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                objective,
                n_trials=self.config.ridge_bo.n_trials,
                timeout=self.config.ridge_bo.timeout_sec,
                show_progress_bar=False,
            )
            used_optuna = True
            best_params = study.best_trial.params
            best_score = float(study.best_value)
            best_detail = study.best_trial.user_attrs.get("decomposition")
        except ImportError:
            run_grid()
            if not leaderboard:
                raise RuntimeError("No successful trials in grid search") from None
            if best_score == float("inf"):
                best_params = min(leaderboard, key=lambda d: d["total"])["params"]

        # Final refit on full data
        decay = best_params["decay"]
        hill_half = best_params["hill_half"]
        hill_slope = best_params["hill_slope"]
        log_alpha = best_params["log_alpha"]
        alpha = float(10**log_alpha)
        full_bundle = build_design_matrix(
            df,
            self.schema,
            self.config,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        coef, intercept = fit_ridge(full_bundle.X, full_bundle.y_modeling, alpha)
        self._coef = coef
        self._intercept = intercept
        self._best_params = best_params
        self._artifacts = RidgeBOArtifacts(
            best_params=best_params,
            objective_history=objective_history if used_optuna else leaderboard,
            coef=coef,
            intercept=intercept,
            leaderboard=sorted(leaderboard, key=lambda d: d["total"])[:20],
        )
        hist = objective_history if used_optuna else leaderboard
        times = [float(h.get("evaluate_wall_time_ms", 0.0)) for h in hist if isinstance(h, dict)]
        telem = {
            "n_hyperparameter_evaluations": len(times),
            "total_eval_wall_time_ms": float(sum(times)),
            "mean_eval_wall_time_ms": float(np.mean(times)) if times else 0.0,
            "design_matrix_cache_unique_hyperparameter_keys": len(design_bundle_cache),
            "replay_calibration_loss_cache_entries": len(replay_loss_cache),
        }
        return {
            "best_score": best_score,
            "best_detail": best_detail,
            "used_optuna": used_optuna,
            "artifacts": self._artifacts,
            "ridge_bo_telemetry": telem,
            "replay_split_meta": dict(self._replay_split_meta),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._coef is None or self._intercept is None:
            raise RuntimeError("Call fit() first")
        df = validate_panel(
            df,
            self.schema,
            integrity_qa=self.config.run_environment == RunEnvironment.PROD,
            calendar_strict=self.config.run_environment == RunEnvironment.PROD,
        )
        df = sort_panel_for_modeling(df, self.schema)
        bundle = build_design_matrix(
            df,
            self.schema,
            self.config,
            decay=self._best_params["decay"],
            hill_half=self._best_params["hill_half"],
            hill_slope=self._best_params["hill_slope"],
        )
        yhat_log = predict_ridge(bundle.X, self._coef, self._intercept)
        # Both forms model log(y); SEMI_LOG uses level media, LOG_LOG uses log(media) in design_matrix.
        if self.config.model_form == ModelForm.LOG_LOG:
            return np.exp(yhat_log)
        return np.exp(yhat_log)
