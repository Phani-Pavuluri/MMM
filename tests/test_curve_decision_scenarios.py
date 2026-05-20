"""Expanded curve vs Δμ alignment scenario matrix (diagnostic only)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.decomposition.curve_bundle import curve_bundle_to_artifact
from mmm.decomposition.curve_stress import stress_test_curve
from mmm.decomposition.curves import build_curve_for_channel
from mmm.decomposition.response_diagnostics import diagnose_response_curve
from mmm.evaluation.curve_decision_alignment import (
    categorize_alignment,
    evaluate_alignment_scenarios,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _fit_and_bundles(
    *,
    decay: float,
    n_channels: int = 2,
    n_geos: int = 2,
):
    channels = tuple(f"ch{i}" for i in range(n_channels))
    spec = SyntheticGeoPanelSpec(
        n_geos=n_geos,
        n_weeks=60,
        channels=channels,
        betas=tuple(0.4 + 0.1 * i for i in range(n_channels)),
        decay=decay,
        noise=0.05,
    )
    df, schema = generate_geo_panel(spec, seed=7)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
            "control_columns": list(schema.control_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=20, horizon_weeks=4),
        ridge_bo={"n_trials": 4},
    )
    resolve_seed_contract(cfg)
    fit = RidgeBOMMMTrainer(cfg, schema).fit(df)
    art = fit["artifacts"]
    bundles = []
    for i, ch in enumerate(schema.channel_columns):
        grid = np.linspace(1.0, float(df[ch].max()) * 1.2, 40)
        curve = build_curve_for_channel(
            grid,
            decay=float(art.best_params["decay"]),
            hill_half=float(art.best_params["hill_half"]),
            hill_slope=float(art.best_params["hill_slope"]),
            beta=float(np.asarray(art.coef).ravel()[i]),
            model_form=cfg.model_form.value,
        )
        bundles.append(
            curve_bundle_to_artifact(
                channel=ch,
                curve=curve,
                diagnostics=diagnose_response_curve(curve),
                stress=stress_test_curve(curve),
                horizon_weeks=52,
                model_form=cfg.model_form.value,
                economics_contract={},
                y_level_scale=1.0,
                target_column=schema.target_column,
            )
        )
    return df, schema, cfg, fit, bundles


@pytest.mark.parametrize(
    "decay,n_geos,label",
    [
        (0.01, 1, "simple"),
        (0.85, 3, "high_carryover"),
        (0.55, 2, "mixed_geo"),
    ],
)
def test_scenario_matrix_runs_and_labels_divergence(decay: float, n_geos: int, label: str) -> None:
    df, schema, cfg, fit, bundles = _fit_and_bundles(decay=decay, n_geos=n_geos, n_channels=2)
    scenarios = [
        {"label": label, "channel": schema.channel_columns[0], "spend_delta_frac": 0.05},
        {"label": f"{label}_ch1", "channel": schema.channel_columns[1], "spend_delta_frac": 0.1},
    ]
    matrix = evaluate_alignment_scenarios(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out=fit,
        curve_bundles=bundles,
        scenarios=scenarios,
    )
    assert matrix["diagnostic_only"] is True
    assert len(matrix["scenarios"]) == 2
    for row in matrix["scenarios"]:
        assert row["divergence_category"] in (
            "expected_divergence",
            "suspicious_divergence",
            "aligned",
        )
        cat = categorize_alignment(row["report"])
        assert cat == row["divergence_category"]


def test_high_carryover_more_divergence_than_simple() -> None:
    _, _, _, _, bundles_simple = _fit_and_bundles(decay=0.01, n_geos=1, n_channels=1)
    df_s, schema_s, cfg_s, fit_s, _ = _fit_and_bundles(decay=0.01, n_geos=1, n_channels=1)
    simple = evaluate_alignment_scenarios(
        panel=df_s,
        schema=schema_s,
        config=cfg_s,
        fit_out=fit_s,
        curve_bundles=bundles_simple,
    )
    df_h, schema_h, cfg_h, fit_h, bundles_h = _fit_and_bundles(decay=0.9, n_geos=3, n_channels=2)
    heavy = evaluate_alignment_scenarios(
        panel=df_h,
        schema=schema_h,
        config=cfg_h,
        fit_out=fit_h,
        curve_bundles=bundles_h,
    )
    rel_s = simple["scenarios"][0]["report"]["relative_abs_difference"]
    rel_h = heavy["scenarios"][0]["report"]["relative_abs_difference"]
    assert rel_h >= rel_s
