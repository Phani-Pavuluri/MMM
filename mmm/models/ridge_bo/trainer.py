"""Ridge + Bayesian Optimization trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.calibration.replay_prod_gate import assert_replay_production_ready
from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.panel_qa import assert_panel_qa_allows_training
from mmm.data.schema import PanelSchema, validate_panel
from mmm.features.design_matrix import apply_masks_for_fit, build_design_matrix
from mmm.models.base import RidgeBOMMMBase
from mmm.models.ridge_bo.objective import build_composite
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
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
        if config.framework != Framework.RIDGE_BO:
            raise ValueError("RidgeBOMMMTrainer requires framework=ridge_bo")
        self.config = config
        self.schema = schema
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._best_params: dict[str, float] = {}
        self._artifacts: RidgeBOArtifacts | None = None
        self._replay_units: list = []
        if config.calibration.use_replay_calibration and config.calibration.replay_units_path:
            self._replay_units = load_calibration_units_from_json(Path(config.calibration.replay_units_path))
            assert_replay_production_ready(config, self._replay_units, schema=self.schema)

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        df = validate_panel(
            df,
            self.schema,
            integrity_qa=self.config.run_environment == RunEnvironment.PROD,
        )
        df = sort_panel_for_modeling(df, self.schema)
        assert_panel_qa_allows_training(df, self.schema, self.config)
        cv = auto_cv_mode(df, self.schema, self.config.cv)
        splits = cv.split(df, self.schema)
        if not splits:
            raise RuntimeError("CV produced no splits; adjust min_train_weeks/horizon")

        if (
            self.config.run_environment == RunEnvironment.PROD
            and self.config.calibration.enabled
            and not (self.config.calibration.use_replay_calibration and self._replay_units)
        ):
            raise ValueError(
                "Production Ridge BO: calibration.enabled requires calibration.use_replay_calibration "
                "and calibration.replay_units_path with explicit replay_estimand on each replay unit."
            )

        objective_history: list[dict[str, Any]] = []
        leaderboard: list[dict[str, Any]] = []

        def evaluate_trial(params: dict[str, float]) -> tuple[float, dict[str, Any]]:
            decay = params["decay"]
            hill_half = params["hill_half"]
            hill_slope = params["hill_slope"]
            log_alpha = params["log_alpha"]
            alpha = float(10**log_alpha)
            coef_rows: list[np.ndarray] = []
            intercept_rows: list[np.ndarray] = []
            y_true_folds: list[np.ndarray] = []
            y_pred_folds: list[np.ndarray] = []
            bundle = build_design_matrix(
                df,
                self.schema,
                self.config,
                decay=decay,
                hill_half=hill_half,
                hill_slope=hill_slope,
            )
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
            if self._replay_units and coef_rows and intercept_rows:
                coef_l = coef_rows[-1]
                int_l = intercept_rows[-1]

                def predict_level(dfp: pd.DataFrame) -> np.ndarray:
                    b = build_design_matrix(
                        dfp,
                        self.schema,
                        self.config,
                        decay=decay,
                        hill_half=hill_half,
                        hill_slope=hill_slope,
                    )
                    ylog = predict_ridge(b.X, coef_l, int_l)
                    return np.exp(ylog)

                rloss, rmeta = aggregate_replay_calibration_loss(
                    self._replay_units,
                    predict_level,
                    schema=self.schema,
                    target_col=self.schema.target_column,
                    config=self.config,
                )
                cal = float(rloss)
                parts["replay"] = rmeta
                parts["mean_lift_se"] = rmeta.get("mean_lift_se", 1.0)
            if parts:
                parts["loss"] = float(cal)
                cal_detail = parts
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
            )
            detail = {
                "total": total,
                "raw": raw.as_dict(),
                "normalized": norm.as_dict(),
                "objective_normalization": norm_report,
                "normalization_profile": self.config.objective.normalization_profile.value,
                "params": params,
            }
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
        return {
            "best_score": best_score,
            "best_detail": best_detail,
            "used_optuna": used_optuna,
            "artifacts": self._artifacts,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._coef is None or self._intercept is None:
            raise RuntimeError("Call fit() first")
        df = validate_panel(
            df,
            self.schema,
            integrity_qa=self.config.run_environment == RunEnvironment.PROD,
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
        if self.config.model_form == ModelForm.SEMI_LOG:
            return np.exp(yhat_log)
        return np.exp(yhat_log)
