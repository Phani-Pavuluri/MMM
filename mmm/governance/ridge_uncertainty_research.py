"""Ridge uncertainty research path — coverage diagnostics only; not production decisioning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema, PanelValidationError
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

PRODUCTION_INTERVALS_ALLOWED = False


def _empirical_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if actual.size == 0:
        return float("nan")
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def investigate_bootstrap_intervals(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    n_bootstrap: int = 12,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Row-resample bootstrap for in-sample prediction intervals (research only).

    Does not change production ``decision_uncertainty`` contract.
    """
    rng = np.random.default_rng(seed)
    n = len(panel)
    if n < 20:
        return {"status": "insufficient_rows", "n_rows": n}
    holdout_n = max(4, int(n * holdout_frac))
    idx = np.arange(n)
    rng.shuffle(idx)
    test_idx = idx[:holdout_n]
    train_idx = idx[holdout_n:]

    train_df = panel.iloc[train_idx].reset_index(drop=True)
    test_df = panel.iloc[test_idx].reset_index(drop=True)
    cfg = config.model_copy(deep=True)
    cfg.ridge_bo = cfg.ridge_bo.model_copy(update={"n_trials": max(2, min(4, cfg.ridge_bo.n_trials))})

    trainer = RidgeBOMMMTrainer(cfg, schema)
    trainer.fit(train_df)
    yhat_base = trainer.predict(test_df)
    y_true = test_df[schema.target_column].to_numpy(dtype=float)

    preds = np.zeros((n_bootstrap, holdout_n), dtype=float)
    n_ok = 0
    for b in range(n_bootstrap):
        boot_idx = rng.choice(len(train_df), size=len(train_df), replace=True)
        boot_df = train_df.iloc[boot_idx].reset_index(drop=True)
        boot_df = boot_df.drop_duplicates(subset=[schema.geo_column, schema.week_column], keep="first")
        if len(boot_df) < max(10, len(train_df) // 3):
            continue
        try:
            boot_trainer = RidgeBOMMMTrainer(cfg, schema)
            boot_trainer.fit(boot_df)
            preds[b] = boot_trainer.predict(test_df)
            n_ok += 1
        except PanelValidationError:
            continue
    if n_ok < 3:
        return {
            "status": "bootstrap_failed",
            "n_successful": n_ok,
            "production_intervals_allowed": PRODUCTION_INTERVALS_ALLOWED,
        }

    lower = np.percentile(preds, 10, axis=0)
    upper = np.percentile(preds, 90, axis=0)
    coverage = _empirical_coverage(y_true, lower, upper)
    return {
        "status": "ok",
        "n_bootstrap": n_bootstrap,
        "holdout_n": holdout_n,
        "empirical_coverage_p80": coverage,
        "mean_interval_width": float(np.mean(upper - lower)),
        "baseline_mae": float(np.mean(np.abs(y_true - yhat_base))),
        "production_intervals_allowed": PRODUCTION_INTERVALS_ALLOWED,
    }


def investigate_conformal_intervals(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    alpha: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Split-conformal style residual quantile intervals (research only)."""
    rng = np.random.default_rng(seed)
    n = len(panel)
    if n < 24:
        return {"status": "insufficient_rows", "n_rows": n}
    cal_n = n // 2
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    cal_df = panel.iloc[cal_idx].reset_index(drop=True)
    test_df = panel.iloc[test_idx].reset_index(drop=True)
    cfg = config.model_copy(deep=True)
    cfg.ridge_bo = cfg.ridge_bo.model_copy(update={"n_trials": max(2, min(4, cfg.ridge_bo.n_trials))})
    cal_trainer = RidgeBOMMMTrainer(cfg, schema)
    cal_trainer.fit(cal_df)
    y_cal = cal_df[schema.target_column].to_numpy(dtype=float)
    yhat_cal = cal_trainer.predict(cal_df)
    resid = np.abs(y_cal - yhat_cal)
    q = float(np.quantile(resid, 1.0 - alpha / 2.0))
    y_test = test_df[schema.target_column].to_numpy(dtype=float)
    yhat_test = cal_trainer.predict(test_df)
    lower = yhat_test - q
    upper = yhat_test + q
    coverage = _empirical_coverage(y_test, lower, upper)
    return {
        "status": "ok",
        "alpha": alpha,
        "residual_quantile": q,
        "empirical_coverage": coverage,
        "production_intervals_allowed": PRODUCTION_INTERVALS_ALLOWED,
    }


def synthetic_dgp_validation(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Sanity-check interval machinery on known synthetic panel."""
    boot = investigate_bootstrap_intervals(panel, schema, config, n_bootstrap=8, seed=seed)
    conf = investigate_conformal_intervals(panel, schema, config, seed=seed + 1)
    sufficient = (
        boot.get("status") == "ok"
        and conf.get("status") == "ok"
        and boot.get("empirical_coverage_p80", 0) >= 0.5
    )
    return {
        "bootstrap": boot,
        "conformal": conf,
        "synthetic_panel_rows": len(panel),
        "evidence_sufficient_for_production_intervals": False,
        "conclusion": (
            "Research intervals may be computed for diagnostics; production decision surfaces "
            "remain point-estimate only until formal coverage gates are defined."
            if not sufficient
            else "Coverage on synthetic DGP is plausible for research; production intervals remain blocked."
        ),
    }


def build_ridge_uncertainty_research_report(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Full research artifact for extension_report / documentation."""
    dgp = synthetic_dgp_validation(panel, schema, config, seed=seed)
    return {
        "production_intervals_allowed": PRODUCTION_INTERVALS_ALLOWED,
        "uncertainty_available_for_decisioning": False,
        "methods": {
            "bootstrap_intervals": dgp["bootstrap"],
            "conformal_intervals": dgp["conformal"],
        },
        "synthetic_dgp_validation": dgp,
        "failure_modes": [
            "High carryover/adstock breaks exchangeability for row bootstrap.",
            "Conformal residual stationarity may fail under structural breaks.",
            "Small panels yield unstable coverage estimates.",
        ],
        "policy_note": "Research output only — never attached to production decision bundles as calibrated CIs.",
    }
