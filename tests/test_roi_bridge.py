import numpy as np

from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.decomposition.curve_bundle import curve_bundle_to_artifact
from mmm.decomposition.curve_stress import stress_test_curve
from mmm.decomposition.curves import build_curve_for_channel
from mmm.decomposition.response_diagnostics import diagnose_response_curve
from mmm.decomposition.roi_bridge import marginal_revenue_and_mroas_level_proxy
from mmm.economics.canonical import build_economics_contract
from mmm.reporting.roi_sections import curve_bundles_to_roi_summary


def test_marginal_revenue_scales_with_y():
    m = np.array([0.01, 0.02])
    out = marginal_revenue_and_mroas_level_proxy(
        marginal_roi_modeling=m, model_form="semi_log", y_level_scale=100.0
    )
    assert np.allclose(np.asarray(out["mroas_level_proxy"]), m * 100.0)


def test_curve_bundle_artifact_includes_mroas_when_y_scale_set():
    grid = np.array([1.0, 5.0, 10.0])
    curve = build_curve_for_channel(
        grid, decay=0.5, hill_half=1.0, hill_slope=2.0, beta=0.5, model_form="semi_log"
    )
    diag = diagnose_response_curve(curve)
    stress = stress_test_curve(curve)
    ec = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(
                path=None,
                channel_columns=["c1"],
                target_column="revenue",
            ),
        )
    )
    art = curve_bundle_to_artifact(
        channel="c1",
        curve=curve,
        diagnostics=diag,
        stress=stress,
        horizon_weeks=52,
        model_form="semi_log",
        economics_contract=ec,
        y_level_scale=50.0,
        target_column="revenue",
    )
    assert "mroas_level_proxy" in art
    assert "mroas_level_consistent" in art
    assert len(art["mroas_level_proxy"]) == len(grid)
    assert len(art["mroas_level_consistent"]) == len(grid)
    assert art["roi_bridge"]["y_level_scale"] == 50.0
    assert "economics" in art
    assert art["economics"]["version"] == "mmm_economics_v1"
    assert art.get("economics_contract", {}).get("contract_version")


def test_roi_summary_reads_curve_bundles():
    bundles = [
        {
            "channel": "a",
            "spend_grid": [1.0, 10.0],
            "mroas_level_proxy": [0.5, 1.0],
            "roi_bridge": {"y_level_scale": 10.0},
        }
    ]
    s = curve_bundles_to_roi_summary(bundles)
    assert s["channels"][0]["channel"] == "a"
    # mid index for len 2 is 1 → second grid value
    assert s["channels"][0]["mroas_level_proxy_mid_grid"] == 1.0
