"""Scenario engine: what-if spend paths against fitted curve bundles (planning surface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
import yaml
from scipy.interpolate import interp1d

from mmm.economics.canonical import economics_contract_for_curve_bundles

UncertaintyMode = Literal["point", "bootstrap", "posterior"]
CurveExtrapolationPolicy = Literal["clamp", "allow_linear_extrapolation_research_only"]


def _interp_response(
    bundle: dict[str, Any],
    spend: float,
    *,
    extrapolation: CurveExtrapolationPolicy = "clamp",
) -> tuple[float, bool]:
    """
    Linear response on the spend grid.

    Returns (value, out_of_grid) where ``out_of_grid`` is True when ``spend`` lies outside the
    observed grid support (values are endpoint-clamped unless research extrapolation is requested).
    """
    g = np.asarray(bundle["spend_grid"], dtype=float)
    r = np.asarray(bundle["response_on_modeling_scale"], dtype=float)
    if g.size < 2:
        raise ValueError("curve bundle needs at least 2 grid points")
    order = np.argsort(g)
    g_s = g[order].astype(float)
    r_s = r[order].astype(float)
    g_min, g_max = float(g_s[0]), float(g_s[-1])
    out_of_grid = bool(spend < g_min or spend > g_max)
    if extrapolation == "allow_linear_extrapolation_research_only":
        f = interp1d(g_s, r_s, kind="linear", fill_value="extrapolate", bounds_error=False)
        return float(f(float(spend))), out_of_grid
    spend_c = float(np.clip(spend, g_min, g_max))
    f = interp1d(g_s, r_s, kind="linear", bounds_error=False, fill_value=(float(r_s[0]), float(r_s[-1])))
    return float(f(spend_c)), out_of_grid


@dataclass
class SpendScenario:
    """Single-period or stepped spend vectors by channel name."""

    baseline_spend: dict[str, float]
    proposed_spend: dict[str, float] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    y_level_scale: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpendScenario:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("scenario YAML must be a mapping")
        steps = list(raw.get("steps") or [])
        base_raw = raw.get("baseline_spend")
        if isinstance(base_raw, dict) and base_raw:
            base = {str(k): float(v) for k, v in base_raw.items()}
        elif steps and isinstance(steps[0].get("spend"), dict):
            base = {str(k): float(v) for k, v in steps[0]["spend"].items()}
        else:
            base = {}
        prop_raw = raw.get("proposed_spend")
        prop = {str(k): float(v) for k, v in prop_raw.items()} if isinstance(prop_raw, dict) else None
        ys = raw.get("y_level_scale")
        y_level = float(ys) if ys is not None else None
        if not base and not steps:
            raise ValueError("scenario YAML needs baseline_spend and/or steps")
        return cls(baseline_spend=base, proposed_spend=prop, steps=steps, y_level_scale=y_level)

    def resolved_proposed(self) -> dict[str, float]:
        if self.proposed_spend is not None:
            return dict(self.proposed_spend)
        if self.steps:
            last = self.steps[-1]
            sp = last.get("spend")
            if isinstance(sp, dict):
                return {str(k): float(v) for k, v in sp.items()}
        raise ValueError("Set proposed_spend or steps[-1].spend in scenario YAML")


def run_curve_bundle_scenario(
    curve_bundles: list[dict[str, Any]],
    scenario: SpendScenario,
    *,
    curve_extrapolation_policy: CurveExtrapolationPolicy = "clamp",
) -> dict[str, Any]:
    """
    Compare baseline vs proposed spend using each channel's ``response_on_modeling_scale`` curve.

    Returns per-channel modeling-scale response levels and deltas, plus an optional **level**
    incremental proxy ``≈ y * Δ(log contribution)`` when ``y_level_scale`` is set (small-delta).

    ``curve_extrapolation_policy`` defaults to ``clamp`` (no silent extrapolation); out-of-grid spends
    are flagged per channel. ``allow_linear_extrapolation_research_only`` is explicit and non-prod-oriented.
    """
    by_ch = {str(b["channel"]): b for b in curve_bundles if b.get("channel")}
    per: list[dict[str, Any]] = []
    dlog_sum = 0.0
    any_oob = False
    for ch, b in by_ch.items():
        sb = float(scenario.baseline_spend.get(ch, np.nan))
        sp = float(scenario.resolved_proposed().get(ch, sb))
        if np.isnan(sb):
            continue
        rb, oob_b = _interp_response(b, sb, extrapolation=curve_extrapolation_policy)
        rp, oob_p = _interp_response(b, sp, extrapolation=curve_extrapolation_policy)
        any_oob = any_oob or oob_b or oob_p
        dlog = rp - rb
        dlog_sum += dlog
        row: dict[str, Any] = {
            "channel": ch,
            "baseline_spend": sb,
            "proposed_spend": sp,
            "response_modeling_baseline": rb,
            "response_modeling_proposed": rp,
            "delta_response_modeling": dlog,
            "spend_out_of_curve_grid": bool(oob_b or oob_p),
        }
        if scenario.y_level_scale and scenario.y_level_scale > 0:
            y = float(scenario.y_level_scale)
            row["incremental_kpi_level_small_delta_proxy"] = float(y * dlog)
        per.append(row)
    out: dict[str, Any] = {
        "kind": "curve_bundle_spend_shift",
        "delta_response_modeling_sum": float(dlog_sum),
        "per_channel": per,
        "curve_extrapolation_policy": curve_extrapolation_policy,
        "any_spend_out_of_curve_grid": bool(any_oob),
    }
    if scenario.y_level_scale and scenario.y_level_scale > 0:
        out["incremental_kpi_level_sum_small_delta_proxy"] = float(scenario.y_level_scale * dlog_sum)
    if scenario.steps:
        out["n_steps"] = len(scenario.steps)
    ec = economics_contract_for_curve_bundles(curve_bundles, strict=False)
    if ec is not None:
        out["economics_contract"] = ec
    return out


def run_stepped_scenario(
    curve_bundles: list[dict[str, Any]],
    *,
    steps: list[dict[str, Any]],
    y_level_scale: float | None,
    curve_extrapolation_policy: CurveExtrapolationPolicy = "clamp",
) -> dict[str, Any]:
    """Evaluate a list of ``steps`` each with ``spend: {channel: level}``; returns trajectory."""
    by_ch = {str(b["channel"]): b for b in curve_bundles if b.get("channel")}
    traj: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    any_oob = False
    for i, st in enumerate(steps):
        sp = st.get("spend")
        if not isinstance(sp, dict):
            raise ValueError(f"steps[{i}] must have spend: {{channel: value}}")
        spend = {str(k): float(v) for k, v in sp.items()}
        row: dict[str, Any] = {
            "step": i,
            "week": st.get("week", i),
            "spend": spend,
            "response_modeling": {},
            "spend_out_of_curve_grid": {},
        }
        for ch, sv in spend.items():
            if ch not in by_ch:
                continue
            val, oob = _interp_response(by_ch[ch], sv, extrapolation=curve_extrapolation_policy)
            any_oob = any_oob or oob
            row["response_modeling"][ch] = val
            row["spend_out_of_curve_grid"][ch] = oob
        if prev is not None and y_level_scale and y_level_scale > 0:
            dlog = 0.0
            for ch in by_ch:
                if ch in spend and ch in prev:
                    v1, _ = _interp_response(by_ch[ch], spend[ch], extrapolation=curve_extrapolation_policy)
                    v0, _ = _interp_response(by_ch[ch], prev[ch], extrapolation=curve_extrapolation_policy)
                    dlog += v1 - v0
            row["incremental_kpi_level_small_delta_proxy_vs_prev"] = float(y_level_scale * dlog)
        prev = spend
        traj.append(row)
    out: dict[str, Any] = {
        "kind": "curve_bundle_stepped",
        "trajectory": traj,
        "curve_extrapolation_policy": curve_extrapolation_policy,
        "any_spend_out_of_curve_grid": bool(any_oob),
    }
    ec = economics_contract_for_curve_bundles(curve_bundles, strict=False)
    if ec is not None:
        out["economics_contract"] = ec
    return out


@dataclass
class SpendPlan:
    """
    Counterfactual spend path for ``simulate()`` — aggregate and optional per-geo paths.

    ``aggregate_steps`` entries are ``{"week": int, "spend": {channel: level}}`` (same shape as
    :func:`run_stepped_scenario` ``steps``).
    """

    horizon_weeks: int
    aggregate_steps: list[dict[str, Any]]
    y_level_scale: float | None = None
    by_geo: dict[str, list[dict[str, Any]]] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpendPlan:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("plan YAML must be a mapping")
        hz = int(raw.get("horizon_weeks", 0) or 0)
        steps = list(raw.get("aggregate_steps") or [])
        ys = raw.get("y_level_scale")
        yl = float(ys) if ys is not None else None
        bg = raw.get("by_geo")
        geo = dict(bg) if isinstance(bg, dict) else None
        if hz < 1 and not steps:
            raise ValueError("Set horizon_weeks and/or aggregate_steps")
        if hz < 1:
            hz = len(steps)
        return cls(horizon_weeks=hz, aggregate_steps=steps, y_level_scale=yl, by_geo=geo)


def simulate_curve_diagnostic(
    plan: SpendPlan,
    curve_bundles: list[dict[str, Any]],
    *,
    uncertainty_mode: UncertaintyMode = "point",
    bootstrap_bundle_sets: list[list[dict[str, Any]]] | None = None,
    curve_extrapolation_policy: CurveExtrapolationPolicy = "clamp",
) -> dict[str, Any]:
    """
    **Diagnostic only:** curve / response-surface interpolation over ``SpendPlan`` steps.

    Not the canonical Δμ path; for budget decisions use :func:`mmm.planning.decision_simulate.simulate`
    with the training panel and ``ridge_fit_summary`` (CLI: ``mmm simulate``).

    - ``point``: deterministic curves.
    - ``bootstrap``: pass ``bootstrap_bundle_sets`` (one list of curve bundles per draw); returns draws.
    - ``posterior``: reserved; same as bootstrap with posterior draws when wired to MCMC output.
    """
    if uncertainty_mode == "bootstrap":
        if not bootstrap_bundle_sets:
            raise ValueError("uncertainty_mode='bootstrap' requires bootstrap_bundle_sets")
        draws: list[dict[str, Any]] = []
        for bset in bootstrap_bundle_sets:
            step_out = run_stepped_scenario(
                bset,
                steps=plan.aggregate_steps,
                y_level_scale=plan.y_level_scale,
                curve_extrapolation_policy=curve_extrapolation_policy,
            )
            ec_draw = economics_contract_for_curve_bundles(bset, strict=False)
            if ec_draw is not None:
                step_out = {**step_out, "economics_contract": ec_draw}
            draws.append(step_out)
        boot_out: dict[str, Any] = {
            "kind": "simulate_bootstrap",
            "horizon_weeks": plan.horizon_weeks,
            "n_draws": len(draws),
            "draws": draws,
            "simulation_contract": {
                "entrypoint": "simulate_curve_diagnostic",
                "role": "diagnostic_curve_interpolation",
                "curve_extrapolation_policy": curve_extrapolation_policy,
            },
        }
        ec_top = economics_contract_for_curve_bundles(curve_bundles, strict=False)
        if ec_top is not None:
            boot_out["economics_contract"] = ec_top
        return boot_out
    out: dict[str, Any] = {
        "kind": "simulate_point",
        "horizon_weeks": plan.horizon_weeks,
        "aggregate": run_stepped_scenario(
            curve_bundles,
            steps=plan.aggregate_steps,
            y_level_scale=plan.y_level_scale,
            curve_extrapolation_policy=curve_extrapolation_policy,
        ),
        "simulation_contract": {
            "entrypoint": "simulate_curve_diagnostic",
            "role": "diagnostic_curve_interpolation",
            "curve_extrapolation_policy": curve_extrapolation_policy,
        },
    }
    if plan.by_geo:
        out["by_geo"] = {}
        for geo, gsteps in plan.by_geo.items():
            out["by_geo"][geo] = run_stepped_scenario(
                curve_bundles,
                steps=gsteps,
                y_level_scale=plan.y_level_scale,
                curve_extrapolation_policy=curve_extrapolation_policy,
            )
    ec = economics_contract_for_curve_bundles(curve_bundles, strict=False)
    if ec is not None:
        out["economics_contract"] = ec
    return out


def simulate(
    plan: SpendPlan,
    curve_bundles: list[dict[str, Any]],
    *,
    uncertainty_mode: UncertaintyMode = "point",
    bootstrap_bundle_sets: list[list[dict[str, Any]]] | None = None,
    curve_extrapolation_policy: CurveExtrapolationPolicy = "clamp",
) -> dict[str, Any]:
    """Deprecated alias for :func:`simulate_curve_diagnostic` — do not use for decision workflows."""
    warnings.warn(
        "mmm.simulation.engine.simulate is deprecated; use simulate_curve_diagnostic for curve diagnostics "
        "or mmm.planning.decision_simulate.simulate for full-panel Δμ decisions.",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate_curve_diagnostic(
        plan,
        curve_bundles,
        uncertainty_mode=uncertainty_mode,
        bootstrap_bundle_sets=bootstrap_bundle_sets,
        curve_extrapolation_policy=curve_extrapolation_policy,
    )
