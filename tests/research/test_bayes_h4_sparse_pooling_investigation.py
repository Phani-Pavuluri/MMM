"""INV-H4-001 sparse pooling investigation tests (research only)."""

from __future__ import annotations

import pytest

from mmm.research.bayes_h3_sandbox.model import MODEL_KIND
from mmm.research.bayes_h3_sandbox.recovery_runner import (
    build_h4_recovery_report,
    compute_recovery_metrics,
    validate_posterior_index_mapping,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    WORLD_BAYES_H4_SPARSE_GEO,
    get_recovery_world,
    get_sparse_pooling_diagnostic_world,
)
from mmm.research.bayes_h3_sandbox.sparse_shrinkage_metrics import (
    compute_sparse_shrinkage_decomposition,
    posterior_beta_means_by_geo,
    shrinkage_ratio_vs_center,
)


def _mock_artifact(
    spec,
    *,
    beta_by_geo: dict[str, dict[str, float]],
    mu_post: dict[str, float] | None = None,
    geo_order: list[str] | None = None,
) -> dict:
    geo_order = geo_order or list(spec.geo_order)
    mu_post = mu_post or dict(spec.true_mu_c)
    hier_beta = {
        str(i): {ch: beta_by_geo[geo][ch] for ch in spec.channels} for i, geo in enumerate(geo_order)
    }
    return {
        "model_kind": MODEL_KIND,
        "posterior_summary": {
            "mu_channel_mean": mu_post,
            "tau_channel_mean": dict(spec.true_tau_c),
        },
        "hierarchy_evidence_diagnostics": {
            "beta_geo_index_order": geo_order,
            "channel_index_order": list(spec.channels),
            "beta_geo_channel_mean": hier_beta,
        },
        "pooling_diagnostics": {
            "mu_channel_mean": mu_post,
            "tau_channel_mean": dict(spec.true_tau_c),
        },
        "convergence_diagnostics": {"rhat_max": 1.01},
        "outputs_are_diagnostic_only": True,
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_decision_surface": False,
        "bayes_h3_sandbox": True,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "decision_grade": False,
    }


def test_shrinkage_ratio_lt_one_when_posterior_moves_toward_posterior_mu() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    geo = "dma_sparse"
    ch = "tv"
    true_b = spec.true_beta_gc[geo][ch]
    post_mu = 0.42
    post_b = post_mu + 0.1 * (true_b - post_mu)
    ratio = shrinkage_ratio_vs_center(true_b, post_b, post_mu)
    assert ratio is not None
    assert ratio < 1.0


def test_shrinkage_ratio_gt_one_when_posterior_moves_away_from_center() -> None:
    true_b = 0.85
    center = 0.30
    post_b = 1.10
    ratio = shrinkage_ratio_vs_center(true_b, post_b, center)
    assert ratio is not None
    assert ratio > 1.0


def test_sparse_decomposition_fields_present() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    post_mu = {"tv": 0.35, "search": 0.18}
    artifact = _mock_artifact(
        spec,
        beta_by_geo={
            "dma_dense_a": dict(spec.true_beta_gc["dma_dense_a"]),
            "dma_dense_b": dict(spec.true_beta_gc["dma_dense_b"]),
            "dma_sparse": {"tv": 0.50, "search": 0.12},
        },
        mu_post=post_mu,
    )
    decomp = compute_sparse_shrinkage_decomposition(artifact, spec)
    assert decomp["by_geo_channel"]
    row = next(e for e in decomp["by_geo_channel"] if e["geo"] == "dma_sparse" and e["channel"] == "tv")
    assert row["true_beta_gc"] == spec.true_beta_gc["dma_sparse"]["tv"]
    assert row["true_mu_c"] == spec.true_mu_c["tv"]
    assert row["posterior_mu_c_mean"] == post_mu["tv"]
    assert row["posterior_beta_gc_mean"] == 0.50
    assert row["distance_true_sparse_to_true_mu"] is not None
    assert row["distance_posterior_sparse_to_posterior_mu"] is not None
    assert row["shrinkage_ratio_vs_posterior_mu"] is not None
    assert decomp["shrinkage_ratio_sparse"] is not None


def test_recovery_metrics_primary_vs_legacy_shrinkage() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    artifact = _mock_artifact(
        spec,
        beta_by_geo={g: dict(spec.true_beta_gc[g]) for g in spec.geo_order},
        mu_post={"tv": 0.30, "search": 0.18},
    )
    artifact["hierarchy_evidence_diagnostics"]["beta_geo_channel_mean"]["2"]["tv"] = 0.85
    metrics = compute_recovery_metrics(artifact, spec)
    assert metrics["sparse_shrinkage_decomposition"] is not None
    assert metrics["shrinkage_ratio_sparse"] is not None
    assert metrics["shrinkage_ratio_sparse_vs_true_mu"] is not None


def test_posterior_index_mapping_deterministic() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    artifact = _mock_artifact(spec, beta_by_geo={g: dict(spec.true_beta_gc[g]) for g in spec.geo_order})
    a = validate_posterior_index_mapping(artifact, spec)
    b = validate_posterior_index_mapping(artifact, spec)
    assert a == b
    assert a["beta_geo_index_order"] == ["dma_dense_a", "dma_dense_b", "dma_sparse"]
    assert a["channel_index_order"] == ["tv", "search"]
    beta = posterior_beta_means_by_geo(artifact, spec)
    assert set(beta["dma_sparse"]) == {"tv", "search"}


def test_posterior_index_mismatch_detected() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    artifact = _mock_artifact(
        spec,
        beta_by_geo={g: dict(spec.true_beta_gc[g]) for g in spec.geo_order},
        geo_order=["dma_sparse", "dma_dense_a", "dma_dense_b"],
    )
    mapping = validate_posterior_index_mapping(artifact, spec)
    assert mapping["spec_geo_order"] != mapping["beta_geo_index_order"]
    metrics = compute_recovery_metrics(artifact, spec)
    assert metrics["posterior_indexing"]["index_order_matches_spec"] is False


def test_h4_report_preserves_research_only_flags() -> None:
    spec = get_recovery_world(WORLD_BAYES_H4_SPARSE_GEO)
    artifact = _mock_artifact(spec, beta_by_geo={g: dict(spec.true_beta_gc[g]) for g in spec.geo_order})
    report = build_h4_recovery_report(spec, artifact)
    rec = report["h4_recovery"]
    assert rec["approved_for_prod"] is False
    assert rec["prod_decisioning_allowed"] is False
    assert report.get("decision_surface") is None


def test_diagnostic_variants_materialize() -> None:
    for name in (
        "sparse_no_outlier",
        "sparse_outlier_moderate",
        "sparse_stronger_tau_prior",
        "sparse_more_weeks",
    ):
        spec = get_sparse_pooling_diagnostic_world(name)
        assert spec.sparse_geos == ("dma_sparse",)
        assert spec.world_id.startswith("WORLD-BAYES-H4-SPARSE-DIAG")


@pytest.mark.pymc
@pytest.mark.slow
def test_live_sparse_world_shrinkage_decomposition() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world

    report = run_h4_recovery_world(WORLD_BAYES_H4_SPARSE_GEO, fast_mcmc=True)
    rec = report["h4_recovery"]
    decomp = rec.get("sparse_shrinkage_decomposition")
    assert decomp is not None
    assert decomp.get("by_geo_channel")
    assert rec.get("shrinkage_ratio_sparse") is not None
    assert rec.get("shrinkage_ratio_sparse_vs_true_mu") is not None
    assert rec["approved_for_prod"] is False
