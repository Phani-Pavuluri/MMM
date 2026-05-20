"""Controlled DGP tests — implementation sanity only, not causal validity on real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.context import ridge_context_from_fit
from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

# Recovery tolerances (documented; fail on drift).
SEMI_LOG_COEF_RTOL = 0.35
DELTA_MU_RTOL = 0.4
ADSTOCK_RTOL = 1e-6
HILL_RTOL = 1e-6


def test_semi_log_noiseless_coef_recovery() -> None:
    """Fixed transforms; ridge on log(y) ~ log media should recover positive sign and approximate scale."""
    rng = np.random.default_rng(0)
    n = 40
    decay, half, slope = 0.5, 1.0, 2.0
    spend = rng.uniform(5, 20, size=n)
    ad = GeometricAdstock(decay)
    sat = HillSaturation(half, slope)
    x_feat = apply_adstock_saturation_series(spend, ad, sat)
    beta = 0.4
    y = np.exp(0.5 + beta * np.log1p(np.maximum(x_feat, 1e-9)))
    df = pd.DataFrame(
        {
            "geo_id": ["G0"] * n,
            "week_start_date": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "revenue": y,
            "search": spend,
        }
    )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search",))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["search"],
        },
    )
    bundle = build_design_matrix(df, schema, cfg, decay=decay, hill_half=half, hill_slope=slope)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, alpha=1e-6)
    assert float(coef.ravel()[0]) > 0
    yhat = np.exp(predict_ridge(bundle.X, coef, intercept))
    assert float(np.mean(np.abs(y - yhat) / (y + 1e-9))) < 0.05


def test_known_delta_mu_from_fixed_coefs() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=45, channels=("search",), betas=(0.6,), noise=0.0)
    df, schema = generate_geo_panel(spec, seed=3)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=15, horizon_weeks=3),
        ridge_bo={"n_trials": 4},
    )
    fit = RidgeBOMMMTrainer(cfg, schema).fit(df)
    ctx = ridge_context_from_fit(df, schema, cfg, fit)
    base = bau_baseline_from_panel(df, schema)
    med = float(df["search"].median())
    res_lo = simulate({**base.spend_by_channel, "search": med * 0.9}, ctx, baseline_plan=base)
    res_hi = simulate({**base.spend_by_channel, "search": med * 1.1}, ctx, baseline_plan=base)
    expected_sign = np.sign(res_hi.delta_mu - res_lo.delta_mu)
    assert expected_sign > 0 or res_hi.delta_mu > res_lo.delta_mu


def test_geometric_adstock_formula() -> None:
    decay = 0.6
    x = np.array([1.0, 0.0, 2.0, 0.0])
    ad = GeometricAdstock(decay)
    out = ad.transform(x)
    expected = np.zeros(4)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        expected[i] = carry
    np.testing.assert_allclose(out, expected, rtol=ADSTOCK_RTOL)


def test_hill_saturation_monotone_formula() -> None:
    half, slope = 1.0, 2.0
    x = np.linspace(0.1, 5.0, 8)
    sat = HillSaturation(half, slope)
    out = sat.transform(x)
    expected = x**slope / (half**slope + x**slope + 1e-12)
    np.testing.assert_allclose(out, expected, rtol=HILL_RTOL)
    assert np.all(np.diff(out) >= -1e-9)


def test_collinear_channels_identifiability_warning() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=3, n_weeks=50, channels=("a", "b"), betas=(0.3, 0.3))
    df, schema = generate_geo_panel(spec, seed=5)
    df["b"] = df["a"] * 1.01 + 1e-6
    from mmm.evaluation.extension_runner import run_post_fit_extensions

    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=15, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    ext = run_post_fit_extensions(
        panel=df, schema=schema, config=cfg, fit_out=fit, yhat=yhat, store=None
    )
    ident = float(ext.get("identifiability", {}).get("identifiability_score", 0.0))
    sep = ext.get("feature_separability_report") or {}
    assert ident > 0.0 or (isinstance(sep, dict) and sep.get("high_risk_pairs"))


def test_placebo_null_documented_threshold() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=4, n_weeks=55, channels=("search",), betas=(0.0,), noise=0.06)
    df, schema = generate_geo_panel(spec, seed=6)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=15, horizon_weeks=3),
        ridge_bo={"n_trials": 3},
    )
    coef = float(np.asarray(RidgeBOMMMTrainer(cfg, schema).fit(df)["artifacts"].coef).ravel()[0])
    assert abs(coef) < 0.12, "placebo channel coef should stay small under null DGP"
