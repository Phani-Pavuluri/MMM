"""Curve vs full-panel Δμ alignment diagnostics (governance regression)."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.decomposition.curve_bundle import curve_bundle_to_artifact
from mmm.decomposition.curve_stress import stress_test_curve
from mmm.decomposition.curves import build_curve_for_channel
from mmm.decomposition.response_diagnostics import diagnose_response_curve
from mmm.evaluation.curve_decision_alignment import (
    CURVE_DECISION_DOC,
    evaluate_curve_decision_alignment,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _fit_panel(*, decay: float, n_channels: int = 1, n_geos: int = 2):
    channels = tuple(f"ch{i}" for i in range(n_channels))
    betas = tuple(0.5 for _ in channels)
    spec = SyntheticGeoPanelSpec(
        n_geos=n_geos,
        n_weeks=60,
        channels=channels,
        betas=betas,
        decay=decay,
        noise=0.05,
    )
    df, schema = generate_geo_panel(spec, seed=4)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=20, horizon_weeks=4),
        ridge_bo={"n_trials": 4},
    )
    resolve_seed_contract(cfg)
    fit = RidgeBOMMMTrainer(cfg, schema).fit(df)
    return df, schema, cfg, fit


def test_curves_and_panel_agree_under_simple_conditions() -> None:
    df, schema, cfg, fit = _fit_panel(decay=0.01, n_channels=1, n_geos=1)
    art = fit["artifacts"]
    ch = schema.channel_columns[0]
    q95 = float(df[ch].quantile(0.95))
    grid = np.linspace(1.0, max(q95, 2.0), 40)
    curve = build_curve_for_channel(
        grid,
        decay=float(art.best_params["decay"]),
        hill_half=float(art.best_params["hill_half"]),
        hill_slope=float(art.best_params["hill_slope"]),
        beta=float(np.asarray(art.coef).ravel()[0]),
        model_form=cfg.model_form.value,
    )
    diag = diagnose_response_curve(curve)
    stress = stress_test_curve(curve)
    bundles = [
        curve_bundle_to_artifact(
            channel=ch,
            curve=curve,
            diagnostics=diag,
            stress=stress,
            horizon_weeks=52,
            model_form=cfg.model_form.value,
            economics_contract={},
            y_level_scale=1.0,
            target_column=schema.target_column,
        )
    ]
    rep = evaluate_curve_decision_alignment(
        panel=df, schema=schema, config=cfg, fit_out=fit, curve_bundles=bundles
    )
    assert rep["policy_note"] == CURVE_DECISION_DOC
    assert np.isfinite(rep["full_panel_delta_mu"])
    assert np.isfinite(rep["curve_delta_response"])
    # Univariate curve vs full-panel path should be closer under near-zero adstock than high carryover.
    assert rep["relative_abs_difference"] < 2.0


def _alignment_report(decay: float, n_geos: int) -> dict:
    df, schema, cfg, fit = _fit_panel(decay=decay, n_channels=1, n_geos=n_geos)
    art = fit["artifacts"]
    ch = schema.channel_columns[0]
    grid = np.linspace(1.0, float(df[ch].max()) * 1.2, 40)
    curve = build_curve_for_channel(
        grid,
        decay=float(art.best_params["decay"]),
        hill_half=float(art.best_params["hill_half"]),
        hill_slope=float(art.best_params["hill_slope"]),
        beta=float(np.asarray(art.coef).ravel()[0]),
        model_form=cfg.model_form.value,
    )
    diag = diagnose_response_curve(curve)
    stress = stress_test_curve(curve)
    bundles = [
        curve_bundle_to_artifact(
            channel=ch,
            curve=curve,
            diagnostics=diag,
            stress=stress,
            horizon_weeks=52,
            model_form=cfg.model_form.value,
            economics_contract={},
            y_level_scale=1.0,
            target_column=schema.target_column,
        )
    ]
    return evaluate_curve_decision_alignment(
        panel=df, schema=schema, config=cfg, fit_out=fit, curve_bundles=bundles
    )


def test_curves_diverge_more_under_adstock_than_simple() -> None:
    simple = _alignment_report(decay=0.01, n_geos=1)
    carry = _alignment_report(decay=0.85, n_geos=2)
    assert carry["relative_abs_difference"] >= simple["relative_abs_difference"]
    assert carry["warning_level"] in ("warning", "critical", "info")
