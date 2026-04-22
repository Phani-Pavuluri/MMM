from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.decomposition.curve_bundle import curve_bundle_to_artifact
from mmm.decomposition.curve_stress import stress_test_curve
from mmm.decomposition.curves import build_curve_for_channel
from mmm.decomposition.response_diagnostics import diagnose_response_curve
from mmm.economics.canonical import build_economics_contract


def build_curve_diagnostics_bundle(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
) -> dict[str, Any]:
    art_params = None
    art_coef0 = 0.1
    if config.framework == Framework.RIDGE_BO and fit_out.get("artifacts") is not None:
        art_params = fit_out["artifacts"].best_params
        art_coef0 = float(fit_out["artifacts"].coef[0])
    decay = art_params["decay"] if art_params else float(config.transforms.adstock_params.get("decay", 0.5))
    hh = art_params["hill_half"] if art_params else float(config.transforms.saturation_params.get("half_max", 1.0))
    hs = art_params["hill_slope"] if art_params else float(config.transforms.saturation_params.get("slope", 2.0))
    hz = int(config.extensions.curves.steady_state_horizon_weeks)
    coefs: list[float] = []
    if config.framework == Framework.RIDGE_BO and fit_out.get("artifacts") is not None:
        coef_arr = np.asarray(fit_out["artifacts"].coef, dtype=float).ravel()
        n_ch = len(schema.channel_columns)
        coefs = [float(coef_arr[j]) if j < coef_arr.size else art_coef0 for j in range(n_ch)]
    else:
        coefs = [art_coef0] * len(schema.channel_columns)

    curve_bundles: list[dict[str, Any]] = []
    diag_first = None
    stress_first = None
    safe_all = True
    y_level_scale = float(np.mean(np.maximum(panel[schema.target_column].to_numpy(dtype=float), 1e-12)))
    economics_contract = build_economics_contract(config)

    for idx, ch in enumerate(schema.channel_columns):
        q95 = float(panel[ch].quantile(0.95))
        grid = np.linspace(1.0, max(q95, 2.0), 40)
        beta_ch = coefs[idx] if idx < len(coefs) else art_coef0
        curve = build_curve_for_channel(
            grid,
            decay=decay,
            hill_half=hh,
            hill_slope=hs,
            beta=beta_ch,
            model_form=config.model_form.value,
            horizon_weeks=hz,
        )
        diag = diagnose_response_curve(curve)
        stress = stress_test_curve(curve)
        safe_ch = bool(diag.safe_for_optimization and not stress.numerically_unstable_for_sqp)
        safe_all = safe_all and safe_ch
        if idx == 0:
            diag_first = diag
            stress_first = stress
        curve_bundles.append(
            curve_bundle_to_artifact(
                channel=ch,
                curve=curve,
                diagnostics=diag,
                stress=stress,
                horizon_weeks=hz,
                model_form=config.model_form.value,
                economics_contract=economics_contract,
                y_level_scale=y_level_scale,
                target_column=schema.target_column,
            )
        )

    assert diag_first is not None and stress_first is not None
    return {
        "response_diagnostics": diag_first.to_json(),
        "curve_stress": stress_first.to_json(),
        "safe_for_optimization": safe_all,
        "curve_bundle": curve_bundles[0],
        "curve_bundles": curve_bundles,
    }
