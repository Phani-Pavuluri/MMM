"""Post-fit extension registry and built-in plugins."""

from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import (
    ExtensionDependencyError,
    ExtensionRegistrationError,
    ExtensionRegistry,
    ExtensionSpec,
    get_extension_registry,
    get_registered_extensions,
    register_extension,
    reset_extension_registry_for_tests,
)

_PLUGINS_LOADED = False


def ensure_extensions_registered(*, force: bool = False) -> None:
    """Import plugin modules and register specs (deterministic, no discovery)."""
    global _PLUGINS_LOADED
    if force:
        _PLUGINS_LOADED = False
    if _PLUGINS_LOADED:
        return
    from mmm.evaluation.extensions import (
        bootstrap_ext,
        curve_decision_alignment,
        drift_monitor,
        feature_separability,
        governance,
        model_card,
        panel_qa,
        standard_extensions,
    )
    from mmm.evaluation.extensions import experiment_scheduler as experiment_scheduler_plugin

    bootstrap_ext.register()
    panel_qa.register()
    standard_extensions.register_standard_extensions()
    governance.register()
    feature_separability.register()
    experiment_scheduler_plugin.register()
    drift_monitor.register()
    curve_decision_alignment.register()
    model_card.register()
    _PLUGINS_LOADED = True


def register_builtin_extensions(registry: ExtensionRegistry) -> None:
    """Backward-compatible hook: populate *registry* from plugin definitions."""
    ensure_extensions_registered(force=True)
    for spec in get_registered_extensions():
        if spec.name not in registry.names():
            registry.register(spec)


__all__ = [
    "ExtensionContext",
    "ExtensionDependencyError",
    "ExtensionRegistrationError",
    "ExtensionRegistry",
    "ExtensionSpec",
    "ensure_extensions_registered",
    "get_extension_registry",
    "get_registered_extensions",
    "register_builtin_extensions",
    "register_extension",
    "reset_extension_registry_for_tests",
]
