"""Programmatic synthetic DGP certification — parity with CI exact tests."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.config.transform_policy import TRANSFORM_POLICY_VERSION, build_transform_policy_manifest
from mmm.data.schema import PanelSchema
from mmm.decision.gates import allow_decision_pipeline
from mmm.features.design_matrix import build_design_matrix
from mmm.governance.decision_ridge_summary import validate_ridge_fit_summary_for_prod_decide
from mmm.governance.policy import PolicyError
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate
from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.saturation.hill import HillSaturation
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

REPORT_VERSION = "mmm_synthetic_certification_v2"

CertificationLevel = Literal["smoke", "exact", "incomplete"]

EXACT_CHECK_NAMES: tuple[str, ...] = (
    "semi_log_delta_mu_exact",
    "geometric_adstock_carryover",
    "hill_saturation_analytic",
    "geometric_adstock_design_matrix",
    "hill_saturation_monotone_design_matrix",
    "two_channel_optimizer_direction",
    "transform_policy_consistency",
)

SMOKE_CHECK_NAMES: tuple[str, ...] = (
    "semi_log_delta_mu_exact",
    "geometric_adstock_carryover",
    "hill_saturation_analytic",
)


def _semi_log_panel_cfg() -> tuple[pd.DataFrame, PanelSchema, MMMConfig]:
    rows = []
    for g in ("G0", "G1"):
        for w in range(8):
            rows.append({"geo_id": g, "week_start_date": w, "revenue": 100.0, "tv": 10.0, "search": 5.0})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv", "search"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv", "search"],
        },
    )
    return panel, schema, cfg


def _check_semi_log_delta_mu_exact() -> None:
    panel, schema, cfg = _semi_log_panel_cfg()
    coef = np.array([0.02, 0.05])
    intercept = np.array([np.log(100.0)])
    base = bau_baseline_from_panel(panel, schema)
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=coef,
        intercept=intercept,
        best_params={"decay": 0.01, "hill_half": 1e6, "hill_slope": 1.0},
    )
    res_lo = simulate({"tv": 10.0, "search": 5.0}, ctx, baseline_plan=base)
    res_hi = simulate({"tv": 20.0, "search": 5.0}, ctx, baseline_plan=base)
    expected = res_hi.delta_mu - res_lo.delta_mu
    res = simulate({"tv": 20.0, "search": 5.0}, ctx, baseline_plan=base)
    if res.delta_mu <= 0.0 or abs(res.delta_mu - expected) > 1e-9:
        raise AssertionError(f"semi_log delta_mu inconsistent: got {res.delta_mu}, expected {expected}")


def _check_geometric_adstock_carryover() -> None:
    decay = 0.5
    ad = GeometricAdstock(decay)
    out = ad.transform(np.array([100.0, 0.0, 0.0, 0.0]))
    if abs(float(out[1]) - decay * 100.0) > 1e-12:
        raise AssertionError(f"adstock week-1 carryover expected {decay * 100}, got {out[1]}")


def _check_hill_saturation_analytic() -> None:
    half, slope, x = 10.0, 2.0, 5.0
    sat = HillSaturation(half_max=half, slope=slope)
    expected = x**slope / (half**slope + x**slope + 1e-12)
    got = float(sat.transform(np.array([x]))[0])
    if abs(got - expected) > 1e-12:
        raise AssertionError(f"hill transform mismatch: {got} vs {expected}")


def _check_geometric_adstock_design_matrix() -> None:
    spend = [100.0, 0, 0, 0, 10.0]
    rows = [{"geo_id": "G0", "week_start_date": i, "revenue": 10.0, "tv": s} for i, s in enumerate(spend)]
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv"],
        },
    )
    b0 = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1e6, hill_slope=1.0)
    b1 = build_design_matrix(panel.assign(tv=0.0), schema, cfg, decay=0.5, hill_half=1e6, hill_slope=1.0)
    if not float(b0.X[1, 0]) > float(b1.X[1, 0]):
        raise AssertionError("design matrix adstock carryover not preserved")


def _check_hill_monotone_design_matrix() -> None:
    spends = np.linspace(1.0, 50.0, 10)
    xs: list[float] = []
    for s in spends:
        panel = pd.DataFrame([{"geo_id": "G0", "week_start_date": 0, "revenue": 10.0, "tv": float(s)}])
        schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
        cfg = MMMConfig(
            framework=Framework.RIDGE_BO,
            data={
                "geo_column": "geo_id",
                "week_column": "week_start_date",
                "target_column": "revenue",
                "channel_columns": ["tv"],
            },
        )
        b = build_design_matrix(panel, schema, cfg, decay=0.01, hill_half=10.0, hill_slope=2.0)
        xs.append(float(b.X[0, 0]))
    if not np.all(np.diff(xs) >= -1e-9):
        raise AssertionError("hill saturation not monotone in design matrix path")


def _check_two_channel_optimizer_direction() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=24, channels=("low", "high"), betas=(0.05, 0.4))
    panel, schema = generate_geo_panel(spec, seed=99)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv={"mode": "rolling", "n_splits": 2, "min_train_weeks": 10, "horizon_weeks": 3},
        budget={"enabled": True, "total_budget": 200.0},
        extensions={"product": {"simulation_optimizer_n_starts": 10, "simulation_optimizer_stability_checks": 2}},
    )
    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

    tr_out = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=tr_out["artifacts"].coef,
        intercept=tr_out["artifacts"].intercept,
        best_params=tr_out["artifacts"].best_params,
    )
    base = bau_baseline_from_panel(panel, schema)
    n_ch = len(schema.channel_columns)
    with allow_decision_pipeline():
        opt = optimize_budget_via_simulation(
            ctx,
            baseline_plan=base,
            current_spend=np.array([50.0, 50.0]),
            total_budget=200.0,
            channel_min=np.zeros(n_ch),
            channel_max=np.full(n_ch, 200.0),
        )
    alloc = opt.get("allocation") or opt.get("recommended_allocation") or opt.get("spend_by_channel") or {}
    if not isinstance(alloc, dict):
        raise AssertionError("optimizer returned no allocation dict")
    high_spend = float(alloc.get("high", 0.0))
    low_spend = float(alloc.get("low", 0.0))
    if high_spend < low_spend - 1e-6:
        raise AssertionError(f"optimizer did not prefer high-beta channel: high={high_spend}, low={low_spend}")


def _check_transform_policy_consistency() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={"channel_columns": ["tv"], "data_version_id": "synthetic-cert"},
        cv={"mode": "rolling", "n_splits": 2, "min_train_weeks": 8, "horizon_weeks": 2},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
        transforms={"adstock": "geometric", "saturation": "hill"},
    )
    manifest = build_transform_policy_manifest(cfg)
    if manifest.get("adstock") != "geometric" or manifest.get("saturation") != "hill":
        raise AssertionError("transform policy manifest must be geometric + hill for prod Ridge contract")
    if manifest.get("policy_version") != TRANSFORM_POLICY_VERSION:
        raise AssertionError("transform policy version mismatch")
    er_ok = {
        "ridge_fit_summary": {
            "coef": [0.1],
            "intercept": [0.0],
            "model_form": "semi_log",
            "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
        },
        "transform_policy": manifest,
        "data_fingerprint": {"sha256_combined": "b" * 64},
    }
    validate_ridge_fit_summary_for_prod_decide(cfg, er_ok)
    er_bad = {**er_ok, "transform_policy": {**manifest, "saturation": "log"}}
    try:
        validate_ridge_fit_summary_for_prod_decide(cfg, er_bad)
    except PolicyError:
        return
    raise AssertionError("transform_policy mismatch must fail prod decide validation")


CHECK_REGISTRY: dict[str, Any] = {
    "semi_log_delta_mu_exact": _check_semi_log_delta_mu_exact,
    "geometric_adstock_carryover": _check_geometric_adstock_carryover,
    "hill_saturation_analytic": _check_hill_saturation_analytic,
    "geometric_adstock_design_matrix": _check_geometric_adstock_design_matrix,
    "hill_saturation_monotone_design_matrix": _check_hill_monotone_design_matrix,
    "two_channel_optimizer_direction": _check_two_channel_optimizer_direction,
    "transform_policy_consistency": _check_transform_policy_consistency,
}

# Backward-compatible alias
_CHECK_REGISTRY = CHECK_REGISTRY


def run_exact_check(name: str) -> None:
    """Run one exact-tier check by name (raises on failure). Used by CI and runtime suite."""
    if name not in EXACT_CHECK_NAMES:
        raise KeyError(f"unknown exact check {name!r}; expected one of {EXACT_CHECK_NAMES!r}")
    CHECK_REGISTRY[name]()


def _resolve_certification_level(*, checks: list[dict[str, Any]], check_names: tuple[str, ...]) -> CertificationLevel:
    name_set = set(check_names)
    subset = [c for c in checks if c["name"] in name_set]
    if not subset:
        return "incomplete"
    if all(c["status"] == "pass" for c in subset):
        if name_set == set(EXACT_CHECK_NAMES):
            return "exact"
        return "smoke"
    return "incomplete"


def run_synthetic_certification_suite(*, mode: Literal["exact", "smoke"] = "exact") -> dict[str, Any]:
    """
    Run synthetic DGP checks (single source of truth for CI and runtime extension reports).

    ``certification_level`` is ``exact`` when all exact-tier checks pass, ``smoke`` when only the
    smoke subset passes, and ``incomplete`` otherwise.
    """
    check_names = EXACT_CHECK_NAMES if mode == "exact" else SMOKE_CHECK_NAMES
    checks: list[dict[str, Any]] = []
    for name in check_names:
        fn = CHECK_REGISTRY[name]
        try:
            fn()
            checks.append({"name": name, "status": "pass", "tier": "exact" if name in EXACT_CHECK_NAMES else "smoke"})
        except Exception as exc:  # noqa: BLE001 — certification captures failures
            checks.append(
                {
                    "name": name,
                    "status": "fail",
                    "tier": "exact" if name in EXACT_CHECK_NAMES else "smoke",
                    "error": str(exc),
                }
            )
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    level = _resolve_certification_level(checks=checks, check_names=check_names)
    exact_pass = all(c["status"] == "pass" for c in checks if c["name"] in EXACT_CHECK_NAMES)
    status = "pass" if level == "exact" and exact_pass else "fail"
    return {
        "report_version": REPORT_VERSION,
        "certification_status": status,
        "certification_level": level,
        "mode": mode,
        "n_pass": n_pass,
        "n_checks": len(checks),
        "checks": checks,
        "exact_check_names": list(EXACT_CHECK_NAMES),
        "governance_warnings": [
            "Synthetic certification proves internal numerical consistency only — not causal validity.",
            "CI and runtime both call CHECK_REGISTRY / run_exact_check (single implementation).",
        ],
    }
