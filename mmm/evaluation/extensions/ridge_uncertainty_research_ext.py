"""Ridge uncertainty research extension (disabled by default)."""

from __future__ import annotations

from mmm.config.schema import Framework
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension
from mmm.governance.ridge_uncertainty_research import build_ridge_uncertainty_research_report


def _run_ridge_uncertainty_research(ctx: ExtensionContext) -> None:
    if ctx.config.framework != Framework.RIDGE_BO:
        return
    seed = int(ctx.config.random_seed or 0)
    ctx.out["ridge_uncertainty_research"] = build_ridge_uncertainty_research_report(
        ctx.panel_s,
        ctx.schema,
        ctx.config,
        seed=seed,
    )


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="ridge_uncertainty_research",
            artifact_key="ridge_uncertainty_research",
            priority=245,
            dependencies=("post_fit_validation",),
            config_key="extensions.ridge_uncertainty_research.enabled",
            run=_run_ridge_uncertainty_research,
            report_keys=("ridge_uncertainty_research",),
        )
    )
