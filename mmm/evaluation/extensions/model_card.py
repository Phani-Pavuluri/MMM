"""Model card extension registration."""

from mmm.evaluation.extensions.handlers.reporting_governance import _run_model_card
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="model_card",
            artifact_key="model_card_md",
            priority=270,
            dependencies=("decision_bundle",),
            run=_run_model_card,
            report_keys=("model_card_md",),
        )
    )
