"""INV-H4-001 sparse pooling behavior investigation runner (research only)."""

from __future__ import annotations

from typing import Any

from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    SPARSE_POOLING_DIAGNOSTIC_WORLD_IDS,
    WORLD_BAYES_H4_SPARSE_GEO,
    get_sparse_pooling_diagnostic_world,
)

INVESTIGATION_ID = "INV-H4-001"


def _recovery_row(report: dict[str, Any]) -> dict[str, Any]:
    rec = report.get("h4_recovery") or {}
    decomp = rec.get("sparse_shrinkage_decomposition") or {}
    return {
        "world_id": rec.get("world_id"),
        "variant": (rec.get("expected_diagnostic_behavior") or {}).get("variant"),
        "shrinkage_ratio_sparse": rec.get("shrinkage_ratio_sparse"),
        "shrinkage_ratio_sparse_vs_true_mu": rec.get("shrinkage_ratio_sparse_vs_true_mu"),
        "sparse_shrinkage_decomposition": decomp,
        "beta_gc_mae": rec.get("beta_gc_mae"),
        "mu_c_mae": rec.get("mu_c_mae"),
        "posterior_indexing": rec.get("posterior_indexing"),
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "research_only": True,
    }


def run_sparse_pooling_investigation(
    *,
    include_baseline_sparse: bool = True,
    fast_mcmc: bool = True,
    nuts_seed: int | None = None,
    panel_seed: int | None = None,
) -> dict[str, Any]:
    """Run baseline sparse world + diagnostic variants (research only)."""
    world_ids: list[str] = []
    if include_baseline_sparse:
        world_ids.append(WORLD_BAYES_H4_SPARSE_GEO)
    world_ids.extend(SPARSE_POOLING_DIAGNOSTIC_WORLD_IDS)

    rows: list[dict[str, Any]] = []
    for wid in world_ids:
        report = run_h4_recovery_world(
            wid,
            fast_mcmc=fast_mcmc,
            nuts_seed=nuts_seed,
            panel_seed=panel_seed,
        )
        rows.append(_recovery_row(report))

    return {
        "investigation_id": INVESTIGATION_ID,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "world_runs": rows,
    }


def run_diagnostic_variant(
    variant: str,
    *,
    fast_mcmc: bool = True,
    nuts_seed: int | None = None,
    panel_seed: int | None = None,
) -> dict[str, Any]:
    """Run one INV-H4-001 diagnostic variant by alias or world id."""
    spec = get_sparse_pooling_diagnostic_world(variant)
    report = run_h4_recovery_world(
        spec.world_id,
        fast_mcmc=fast_mcmc,
        nuts_seed=nuts_seed,
        panel_seed=panel_seed,
    )
    return _recovery_row(report)
