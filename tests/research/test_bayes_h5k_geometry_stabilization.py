"""Tests for H5k geometry stabilization (non-centered / hierarchy ablations)."""

from __future__ import annotations

import pytest

from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import BayesSandboxGuardError
from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    HIERARCHY_FULL_GEO_CHANNEL,
    LIKELIHOOD_CURRENT_DEFAULT,
    PARAMETERIZATION_CENTERED,
    PARAMETERIZATION_NON_CENTERED,
    H5GeometryConfigError,
    resolve_geometry_config,
    validate_geometry_config,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    classify_convergence_status,
    evidence_promotion_allowed,
)
from mmm.research.bayes_h3_sandbox.h5k_geometry_stabilization_runner import (
    build_geometry_stabilization_artifact,
    default_stabilization_specs,
    validate_geometry_stabilization_artifact,
)
from mmm.research.bayes_h3_sandbox.model import fit_h5_sandbox_hierarchical


def test_legacy_default_geometry_unchanged_without_explicit_config() -> None:
    geom = resolve_geometry_config({})
    assert geom["legacy_default"] is True
    assert geom["explicit"] is False
    assert geom["parameterization"] == PARAMETERIZATION_NON_CENTERED
    assert geom["hierarchy_policy"] == HIERARCHY_FULL_GEO_CHANNEL
    assert geom["likelihood_scale_policy"] == LIKELIHOOD_CURRENT_DEFAULT


def test_unsupported_parameterization_fails_closed() -> None:
    with pytest.raises(H5GeometryConfigError, match="parameterization"):
        validate_geometry_config(
            {
                "parameterization": "fully_nonlinear",
                "likelihood_scale_policy": "current_default",
                "hierarchy_policy": "full_geo_channel_hierarchy",
            }
        )


def test_geometry_config_rejected_without_h5_gate() -> None:
    import pandas as pd

    from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
    from mmm.data.schema import PanelSchema

    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "path": None,
            "geo_column": "geo",
            "week_column": "week",
            "target_column": "y",
            "channel_columns": ["c1"],
            "control_columns": [],
        },
        bayesian={"backend": BayesianBackend.PYMC, "draws": 10, "tune": 10, "chains": 2},
    )
    schema = PanelSchema("geo", "week", "y", ("c1",))
    df = pd.DataFrame({"geo": ["a"], "week": ["2020-01-01"], "y": [1.0], "c1": [1.0]})
    with pytest.raises(BayesSandboxGuardError, match="h5_geometry_config"):
        run_sandbox_fit(
            cfg,
            schema,
            df,
            sandbox_model_overrides={
                "h5_geometry_config": {
                    "parameterization": PARAMETERIZATION_NON_CENTERED,
                    "likelihood_scale_policy": "current_default",
                    "hierarchy_policy": "full_geo_channel_hierarchy",
                }
            },
            enable_h5_sandbox=False,
        )


def test_artifact_schema_valid_without_fit() -> None:
    artifact = build_geometry_stabilization_artifact(execute_fit=False)
    validate_geometry_stabilization_artifact(artifact)
    assert artifact["approved_for_prod"] is False
    assert artifact["prod_decisioning_allowed"] is False
    assert artifact["hard_gate"] is False
    assert artifact["production_promotion"] is False
    assert artifact["any_variant_converged_diagnostic_only"] is False
    assert len(artifact["variants"]) == len(default_stabilization_specs())


def test_production_flags_false_on_all_variants() -> None:
    artifact = build_geometry_stabilization_artifact(execute_fit=False)
    for row in artifact["variants"]:
        assert row["approved_for_prod"] is False
        assert row["evidence_promotion_allowed"] is False


def test_no_optimizer_fields_on_artifact() -> None:
    artifact = build_geometry_stabilization_artifact(execute_fit=False)
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert artifact.get(forbidden) is None


def test_evidence_promotion_only_when_converged_diagnostic_only() -> None:
    assert evidence_promotion_allowed("converged_diagnostic_only") is True
    assert evidence_promotion_allowed("weak_convergence") is False
    assert evidence_promotion_allowed("failed_convergence") is False
    assert (
        evidence_promotion_allowed(
            classify_convergence_status(rhat_max=1.02, divergence_count=4)
        )
        is False
    )


def test_stabilization_specs_cover_required_variants() -> None:
    ids = {s.variant_id for s in default_stabilization_specs()}
    assert "H5K-A-H5J-BEST-CENTERED-REPLAY" in ids
    assert "H5K-B-NON-CENTERED-DROP-COLLINEAR-EXTENDED" in ids
    assert "H5K-C-NON-CENTERED-TARGET-ACCEPT-099" in ids
    assert "H5K-D-NON-CENTERED-SINGLE-SEARCH" in ids
    assert "H5K-E-POOLED-CHANNEL-ABLATION" in ids
    assert "H5K-F-FIXED-TAU-ABLATION" in ids


def test_variants_record_geometry_fields_without_fit() -> None:
    artifact = build_geometry_stabilization_artifact(execute_fit=False)
    row = artifact["variants"][1]
    assert row["parameterization"] == PARAMETERIZATION_NON_CENTERED
    assert row["geometry_config"]["parameterization"] == PARAMETERIZATION_NON_CENTERED
    assert row["hierarchy_policy"] == HIERARCHY_FULL_GEO_CHANNEL


@pytest.mark.slow
def test_tiny_non_centered_synthetic_smoke() -> None:
    pymc = pytest.importorskip("pymc")
    del pymc
    import numpy as np
    import pandas as pd

    from mmm.config.schema import MMMConfig, ModelForm
    from mmm.data.schema import PanelSchema

    rng = np.random.default_rng(0)
    rows = []
    for g in ("a", "b"):
        for t in range(8):
            rows.append(
                {
                    "geo": g,
                    "week": f"2020-01-{t+1:02d}",
                    "y": float(rng.normal(10, 1)),
                    "c1": float(rng.normal(5, 1)),
                }
            )
    df = pd.DataFrame(rows)
    schema = PanelSchema(
        geo_column="geo",
        time_column="week",
        target_column="y",
        channel_columns=["c1"],
    )
    cfg = MMMConfig(model_form=ModelForm.SEMI_LOG)
    cfg.bayesian.draws = 50
    cfg.bayesian.tune = 50
    cfg.bayesian.chains = 2
    out = fit_h5_sandbox_hierarchical(
        cfg,
        schema,
        df,
        sandbox_model_overrides={
            "h5_geometry_config": {
                "parameterization": PARAMETERIZATION_NON_CENTERED,
                "likelihood_scale_policy": "current_default",
                "hierarchy_policy": "full_geo_channel_hierarchy",
            }
        },
    )
    assert out["h5_geometry_diagnostics"]["parameterization"] == PARAMETERIZATION_NON_CENTERED
    assert out["convergence_diagnostics"]["rhat_max"] == out["convergence_diagnostics"]["rhat_max"]


@pytest.mark.slow
def test_tiny_centered_synthetic_smoke() -> None:
    pytest.importorskip("pymc")
    import numpy as np
    import pandas as pd

    from mmm.config.schema import MMMConfig, ModelForm
    from mmm.data.schema import PanelSchema

    rng = np.random.default_rng(1)
    rows = []
    for g in ("a", "b"):
        for t in range(8):
            rows.append(
                {
                    "geo": g,
                    "week": f"2020-01-{t+1:02d}",
                    "y": float(rng.normal(10, 1)),
                    "c1": float(rng.normal(5, 1)),
                }
            )
    df = pd.DataFrame(rows)
    schema = PanelSchema(
        geo_column="geo",
        time_column="week",
        target_column="y",
        channel_columns=["c1"],
    )
    cfg = MMMConfig(model_form=ModelForm.SEMI_LOG)
    cfg.bayesian.draws = 50
    cfg.bayesian.tune = 50
    cfg.bayesian.chains = 2
    out = fit_h5_sandbox_hierarchical(
        cfg,
        schema,
        df,
        sandbox_model_overrides={
            "h5_geometry_config": {
                "parameterization": PARAMETERIZATION_CENTERED,
                "likelihood_scale_policy": "current_default",
                "hierarchy_policy": "full_geo_channel_hierarchy",
            }
        },
    )
    assert out["h5_geometry_diagnostics"]["parameterization"] == PARAMETERIZATION_CENTERED
