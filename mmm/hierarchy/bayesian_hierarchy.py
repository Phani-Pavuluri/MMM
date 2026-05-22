"""Bayesian hierarchical channel coefficients (research-only): child ~ Normal(parent, sigma_group)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, ModelForm, PoolingMode
from mmm.data.schema import PanelSchema
from mmm.governance.model_form_policy import LOG_LOG_HIERARCHY_POLICY_MESSAGE
from mmm.governance.policy import PolicyError
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, HierarchyValidationReport, load_hierarchy_definition
from mmm.hierarchy.penalty import HierarchyCoefPair, prepare_hierarchy_for_ridge

BAYESIAN_HIERARCHY_GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Bayesian hierarchy is research-only and does not enable prod decisioning.",
    "Hierarchical borrowing stabilizes estimates but does not create causal evidence.",
    "Parent posterior mass does not imply child causal validity.",
    "Hierarchy is regularization / partial pooling, not identification.",
)


@dataclass
class BayesianHierarchyPrepareResult:
    definition: HierarchyDefinition | None = None
    validation: HierarchyValidationReport | None = None
    pairs: list[HierarchyCoefPair] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def uses_bayesian_hierarchy(config: MMMConfig) -> bool:
    return bool(config.framework == Framework.BAYESIAN and config.bayesian.use_hierarchy)


def validate_bayesian_hierarchy_config(config: MMMConfig) -> None:
    if not uses_bayesian_hierarchy(config):
        return
    if config.framework != Framework.BAYESIAN:
        raise ValueError("bayesian.use_hierarchy requires framework=bayesian")
    b = config.bayesian
    if not b.hierarchy_research_only:
        raise ValueError(
            "bayesian.hierarchy_research_only must remain true (Bayesian hierarchy cannot enable prod decisioning)"
        )
    if not config.hierarchy.hierarchy_definition_path:
        raise ValueError(
            "bayesian.use_hierarchy requires hierarchy.hierarchy_definition_path "
            "(explicit HierarchyDefinition JSON; never inferred from data)"
        )
    if config.model_form == ModelForm.LOG_LOG:
        raise PolicyError(LOG_LOG_HIERARCHY_POLICY_MESSAGE)
    if config.pooling == PoolingMode.NONE:
        raise ValueError(
            "bayesian.use_hierarchy requires pooling=full or pooling=partial "
            "(per-geo independent coefficients are not supported)"
        )


def prepare_bayesian_hierarchy(
    config: MMMConfig,
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> BayesianHierarchyPrepareResult:
    validate_bayesian_hierarchy_config(config)
    if not uses_bayesian_hierarchy(config):
        return BayesianHierarchyPrepareResult()
    path = config.hierarchy.hierarchy_definition_path
    assert path is not None
    definition = load_hierarchy_definition(path)
    geos = {str(x) for x in panel[schema.geo_column].unique()}
    pairs, validation, warnings = prepare_hierarchy_for_ridge(
        definition,
        list(schema.channel_columns),
        panel_geos=geos,
        min_children_per_parent=config.hierarchy.min_children_per_parent,
        allow_cross_branch_pooling=config.hierarchy.allow_cross_branch_pooling,
    )
    if not validation.valid:
        raise ValueError(f"Bayesian hierarchy validation failed: {validation.model_dump()}")
    if not pairs:
        raise ValueError(
            "Bayesian hierarchy: no media coefficient pairs resolved "
            "(use channel/campaign hierarchy or metadata.ridge_effect_pairs for geography)"
        )
    return BayesianHierarchyPrepareResult(
        definition=definition,
        validation=validation,
        pairs=pairs,
        warnings=list(warnings),
    )


def register_bayesian_hierarchical_media_coefs(
    pm: Any,
    *,
    n_media: int,
    media_prior: str,
    media_sigma: float,
    pairs: list[HierarchyCoefPair],
    group_sigma_prior: float,
    deterministic_name: str = "beta_mu_media",
) -> tuple[Any, list[str]]:
    """
    Media coefficients with ``local ~ Normal(parent, hier_sigma_group)`` for mapped children.

    Unmapped media indices keep the standard global/media prior.
    """
    if not pairs:
        if media_prior == "normal_symmetric":
            coef = pm.Normal(deterministic_name, mu=0.0, sigma=media_sigma, shape=n_media)
        else:
            coef = pm.HalfNormal(deterministic_name, sigma=media_sigma, shape=n_media)
        return coef, [deterministic_name]

    child_indices = {p.child_index for p in pairs}
    var_names: list[str] = ["hier_sigma_group"]
    sigma_group = pm.HalfNormal("hier_sigma_group", sigma=float(group_sigma_prior))

    coef_by_idx: dict[int, Any] = {}
    for i in range(n_media):
        if i in child_indices:
            continue
        vname = f"{deterministic_name}_root_{i}"
        if media_prior == "normal_symmetric":
            coef_by_idx[i] = pm.Normal(vname, mu=0.0, sigma=media_sigma)
        else:
            coef_by_idx[i] = pm.HalfNormal(vname, sigma=media_sigma)
        var_names.append(vname)

    for p in pairs:
        parent_var = coef_by_idx[p.parent_index]
        cname = f"{deterministic_name}_child_{p.child_name}"
        coef_by_idx[p.child_index] = pm.Normal(cname, mu=parent_var, sigma=sigma_group)
        var_names.append(cname)

    stacked = pm.math.stack([coef_by_idx[i] for i in range(n_media)])
    coef = pm.Deterministic(deterministic_name, stacked)
    var_names.append(deterministic_name)
    return coef, var_names


def build_bayesian_hierarchy_report(
    config: MMMConfig,
    prepared: BayesianHierarchyPrepareResult,
    idata: Any,
    *,
    pymc_var_names: list[str] | None = None,
) -> dict[str, Any]:
    """Post-fit ``bayesian_hierarchy_report`` (research-only)."""
    enabled = uses_bayesian_hierarchy(config)
    warnings = list(prepared.warnings) + list(BAYESIAN_HIERARCHY_GOVERNANCE_WARNINGS)
    base: dict[str, Any] = {
        "enabled": enabled,
        "research_only": bool(config.bayesian.hierarchy_research_only),
        "prod_decisioning_allowed": False,
        "warnings": warnings,
        "governance_warnings": list(BAYESIAN_HIERARCHY_GOVERNANCE_WARNINGS),
    }
    if not enabled or prepared.definition is None or not prepared.pairs:
        base["skipped"] = True
        return base

    defn = prepared.definition
    val = prepared.validation
    base.update(
        {
            "hierarchy_id": defn.hierarchy_id,
            "hierarchy_type": defn.hierarchy_type,
            "parent_child_mapping": dict(defn.node_mapping),
            "validation": val.model_dump() if val is not None else {},
            "n_hierarchy_pairs": len(prepared.pairs),
            "pooling_mode": config.pooling.value,
            "model_form": config.model_form.value,
        }
    )
    if pymc_var_names:
        base["pymc_hierarchy_var_names"] = list(pymc_var_names)

    if idata is None or not hasattr(idata, "posterior"):
        return base

    post = idata.posterior
    media_var = "beta_mu_media" if config.pooling == PoolingMode.PARTIAL else "beta_media"
    if media_var not in post:
        base["posterior_note"] = f"missing {media_var} in idata.posterior"
        return base

    beta_raw = np.asarray(post[media_var].stack(sample=("chain", "draw")))
    beta = beta_raw.reshape(-1, beta_raw.shape[-1])

    sigma_g = None
    if "hier_sigma_group" in post:
        sigma_g = np.asarray(post["hier_sigma_group"].stack(sample=("chain", "draw"))).ravel()

    shrink_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for p in prepared.pairs:
        child_s = beta[:, p.child_index]
        parent_s = beta[:, p.parent_index]
        diff = child_s - parent_s
        shrink_rows.append(
            {
                "child": p.child_name,
                "parent": p.parent_name,
                "posterior_child_mean": float(np.mean(child_s)),
                "posterior_parent_mean": float(np.mean(parent_s)),
                "posterior_shrinkage_delta": float(np.mean(child_s) - np.mean(parent_s)),
                "posterior_shrinkage_ratio": float(
                    np.mean(diff) / (np.mean(np.abs(parent_s)) + 1e-9)
                ),
                "posterior_child_parent_diff_std": float(np.std(diff)),
            }
        )
        sg = float(np.mean(sigma_g)) if sigma_g is not None and len(sigma_g) else 0.5
        overlap_rows.append(
            {
                "child": p.child_name,
                "parent": p.parent_name,
                "prior_posterior_overlap_score": float(np.mean(np.abs(diff) <= sg)),
                "note": "Fraction of posterior draws with |child-parent| <= posterior mean hier_sigma_group",
            }
        )

    base["posterior_shrinkage_summary"] = shrink_rows
    base["prior_posterior_overlap"] = overlap_rows
    if sigma_g is not None and len(sigma_g):
        base["group_variance_summary"] = {
            "hier_sigma_group_mean": float(np.mean(sigma_g)),
            "hier_sigma_group_std": float(np.std(sigma_g)),
            "hier_sigma_group_p10": float(np.quantile(sigma_g, 0.1)),
            "hier_sigma_group_p90": float(np.quantile(sigma_g, 0.9)),
        }
    else:
        base["group_variance_summary"] = {"note": "hier_sigma_group not found in posterior"}
    return base
