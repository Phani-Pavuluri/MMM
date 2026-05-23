"""Exact synthetic certification — CI calls the same CHECK_REGISTRY as runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm.config.extensions import ExtensionSuiteConfig, FeatureSeparabilityConfig
from mmm.config.schema import CVConfig, Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.features.design_matrix import build_design_matrix
from mmm.governance.synthetic_certification import EXACT_CHECK_NAMES, run_exact_check, run_synthetic_certification_suite
from mmm.models.ridge_bo.trainer import fit_ridge, predict_ridge


@pytest.mark.parametrize("check_name", EXACT_CHECK_NAMES)
def test_exact_check_via_shared_registry(check_name: str) -> None:
    run_exact_check(check_name)


def test_exact_suite_matches_registry() -> None:
    rep = run_synthetic_certification_suite(mode="exact")
    assert rep["certification_level"] == "exact"
    assert rep["certification_status"] == "pass"
    assert rep["n_pass"] == len(EXACT_CHECK_NAMES)
    assert {c["name"] for c in rep["checks"]} == set(EXACT_CHECK_NAMES)


def test_collinear_channels_identifiability_warning() -> None:
    """Integration diagnostic — not part of runtime EXACT_CHECK_NAMES."""
    n = 30
    tv = np.linspace(10, 20, n)
    search = tv * 1.01 + 0.001
    panel = pd.DataFrame(
        {
            "geo_id": ["G0"] * n,
            "week_start_date": range(n),
            "revenue": 100 + 0.1 * tv,
            "tv": tv,
            "search": search,
        }
    )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv", "search"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        extensions=ExtensionSuiteConfig(
            feature_separability=FeatureSeparabilityConfig(enabled=True, auto_group_prefix=True)
        ),
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv", "search"],
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=3),
        ridge_bo={"n_trials": 1},
    )
    bundle = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, 1.0)
    yhat = predict_ridge(bundle.X, coef, intercept)
    er = run_post_fit_extensions(
        panel=panel,
        schema=schema,
        config=cfg,
        fit_out={
            "artifacts": type(
                "A",
                (),
                {
                    "coef": coef,
                    "intercept": intercept,
                    "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
                },
            )(),
            "best_detail": {},
        },
        yhat=np.exp(yhat),
        store=None,
    )
    id_js = er.get("identifiability", {})
    sep = er.get("feature_separability_report", {})
    assert float(id_js.get("identifiability_score", 1.0)) < 0.99 or sep
