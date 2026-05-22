"""Performance audit extension."""

from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension
from mmm.evaluation.performance_audit import build_performance_report


def _run_performance_audit(ctx: ExtensionContext) -> None:
    ctx.out["performance_report"] = build_performance_report(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
    )


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="performance_audit",
            artifact_key="performance_report",
            priority=265,
            dependencies=("post_fit_validation",),
            config_key="extensions.performance_audit.enabled",
            run=_run_performance_audit,
            report_keys=("performance_report",),
        )
    )
