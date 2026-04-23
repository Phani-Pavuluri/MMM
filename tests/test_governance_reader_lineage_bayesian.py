"""Reader tier matrix, prod Bayesian decision block, decision-tier lineage completeness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.artifacts.reader import load_artifact
from mmm.config.extensions import ExtensionSuiteConfig, GovernanceConfig, OptimizationGateConfig
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.decision.service import optimize_budget_decision
from mmm.governance.policy import PolicyError
from mmm.governance.semantics import Surface
from mmm.governance.validation import validate_decision_tier_lineage


def _bundle_core(*, tier: str) -> dict:
    return {
        "artifact_tier": tier,
        "tier": tier,
        "bundle_version": "mmm_decision_bundle_v2",
        "git_sha": "a" * 40,
        "package_version": "0.0.0",
        "dependency_digest": "d" * 64,
        "dependency_lock_digest": "d" * 64,
        "config_sha": "c" * 64,
        "config_hash": "c" * 64,
        "config_fingerprint_sha256": "c" * 64,
        "panel_fingerprint": {"sha256_panel_keycols_sorted_csv": "p" * 64, "n_rows": 1},
        "dataset_snapshot_id": "snap-1",
        "model_release_id": "mr-test-1",
        "runtime_policy_hash": "r" * 16,
        "artifact_schema_version": "mmm_artifact_bundle_v3",
    }


def _write(tmp: Path, name: str, payload: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    ("tier", "surface", "expect_ok"),
    [
        ("diagnostic", Surface.DIAGNOSTIC, True),
        ("research", Surface.DIAGNOSTIC, True),
        ("decision", Surface.DIAGNOSTIC, True),
        ("diagnostic", Surface.RESEARCH, False),
        ("research", Surface.RESEARCH, True),
        ("decision", Surface.RESEARCH, True),
        ("diagnostic", Surface.DECISION, False),
        ("research", Surface.DECISION, False),
        ("decision", Surface.DECISION, True),
    ],
)
def test_load_artifact_tier_matrix(
    tmp_path: Path, tier: str, surface: Surface, *, expect_ok: bool
) -> None:
    path = _write(tmp_path, f"a_{tier}_{surface.value}.json", _bundle_core(tier=tier))
    if expect_ok:
        data = load_artifact(path, surface=surface)
        assert isinstance(data, dict)
    else:
        with pytest.raises(PolicyError, match="surface="):
            load_artifact(path, surface=surface)


def test_optimize_budget_decision_blocks_bayesian_in_prod(tmp_path: Path) -> None:
    out = tmp_path / "opt.json"
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        framework=Framework.BAYESIAN,
        data={
            "path": str(tmp_path / "panel.csv"),
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["c1"],
            "control_columns": [],
        },
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod"},
        bayesian={"posterior_predictive_draws": 100},
        extensions=ExtensionSuiteConfig(
            governance=GovernanceConfig(bayesian_max_mean_abs_ppc_gap=0.5),
            optimization_gates=OptimizationGateConfig(enabled=True),
        ),
    )
    (tmp_path / "panel.csv").write_text("g,w,y,c1\nG1,1,1,1\n", encoding="utf-8")
    er = {
        "ridge_fit_summary": {"coef": [0.1]},
        "governance": {"approved_for_optimization": True},
        "response_diagnostics": {"safe_for_optimization": True},
        "identifiability": {"identifiability_score": 0.5},
        "panel_qa": {"max_severity": "info", "issues": []},
        "model_release": {"state": "planning_allowed", "reasons": [], "triggers": {}},
        "experiment_matching": {"ok": True},
    }
    with pytest.raises(PolicyError, match="Bayesian|bayesian|allow_bayesian"):
        optimize_budget_decision(cfg=cfg, extension_report=er, out=out)


def test_bayesian_dev_framework_parses_for_non_prod_surface() -> None:
    """Non-prod Bayesian config is allowed at parse time (prod decision path still blocks elsewhere)."""
    MMMConfig(
        run_environment=RunEnvironment.DEV,
        framework=Framework.BAYESIAN,
        data={"channel_columns": ["c1"], "control_columns": []},
        bayesian={"posterior_predictive_draws": 50},
    )


@pytest.mark.parametrize(
    "pop_key",
    [
        "git_sha",
        "package_version",
        "dataset_snapshot_id",
        "dependency_lock_digest",
        "runtime_policy_hash",
        "config_hash",
        "model_release_id",
        "artifact_schema_version",
    ],
)
def test_validate_decision_tier_lineage_prod_requires_fields(tmp_path: Path, pop_key: str) -> None:
    b = _bundle_core(tier="decision")
    b.pop(pop_key)
    with pytest.raises(PolicyError, match="incomplete lineage|Missing or empty"):
        validate_decision_tier_lineage(b, run_environment=RunEnvironment.PROD)
