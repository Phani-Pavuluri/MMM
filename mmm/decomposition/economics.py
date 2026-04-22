"""
Canonical economics definitions (Tier 1).

**Disclosure:** Dollar-level budget optimization using only partial steady-state curves is **not**
finance-grade unless full μ (all channels + base) is aligned; use ``disclosure`` text in artifacts.
"""

from __future__ import annotations

from typing import Any, Literal

ECONOMICS_VERSION = "mmm_economics_v1"

ModelFormEconomics = Literal["semi_log", "log_log"]


def economics_definitions_block(
    *,
    model_form: str,
    target_kpi_column: str,
) -> dict[str, Any]:
    mf = model_form.lower().strip()
    base = {
        "version": ECONOMICS_VERSION,
        "target_kpi_column": target_kpi_column,
        "incremental_revenue": (
            "Change in expected KPI (revenue) vs a stated baseline spend path, holding other modeled "
            "inputs fixed where the artifact allows."
        ),
        "roi": (
            "incremental_revenue / incremental_spend, **only** when both are in the same currency and "
            "incremental_spend is explicitly defined for the same counterfactual."
        ),
        "mroas": "d(incremental_revenue)/d(spend) for an infinitesimal spend change on the stated channel.",
        "disclosure": (
            "Do **not** claim dollar-accurate global optimization from isolated partial curves without "
            "full-model calibration; use consistent_partial_curve fields for anchored single-channel "
            "interpretation or full posterior predictive spend simulations."
        ),
    }
    if mf == "semi_log":
        base["semi_log_mapping"] = (
            "Gaussian mean on log scale: μ = α + Σ β_j x_j(S) + controls; KPI Y ≈ exp(μ). "
            "Single-channel partial curve r(S)=β_k x_k(S); anchored level curve Y(S)=exp(μ_rest+r(S)) "
            "with μ_rest=log(Y_anchor)-r(S_anchor). "
            "Linear proxy: ΔY ≈ Y_ref · Δr for small Δr."
        )
    elif mf == "log_log":
        base["log_log_mapping"] = (
            "Media enters as log(x_media); curve r(S)=β·log(x(S)) after adstock/saturation in x. "
            "Elasticity of μ w.r.t. spend is mediated through dx/dS; curve marginal dμ/dS is used. "
            "Level KPI still Y≈exp(μ); same anchored construction applies to partial r(S)."
        )
    else:
        base["note"] = f"model_form {model_form!r} not specialized in economics block"
    return base


def attach_economics_to_curve_artifact(artifact: dict[str, Any], *, target_column: str) -> dict[str, Any]:
    mf = str(artifact.get("model_form", "semi_log"))
    out = {**artifact, "economics": economics_definitions_block(model_form=mf, target_kpi_column=target_column)}
    return out
