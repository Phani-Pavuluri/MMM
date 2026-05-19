"""Deterministic extension registry for post-fit diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from mmm.config.schema import MMMConfig
from mmm.evaluation.extensions.context import ExtensionContext

ArtifactTier = Literal["research", "diagnostic", "decision", "internal"]


class ExtensionRegistrationError(ValueError):
    """Raised when registry invariants are violated."""


class ExtensionDependencyError(ValueError):
    """Raised when dependencies cannot be satisfied."""


def _resolve_config_flag(config: MMMConfig, dotted_path: str) -> bool:
    """Resolve ``extensions.panel_qa.enabled`` style paths on ``MMMConfig``."""
    if not dotted_path:
        return True
    root: Any = config
    for part in dotted_path.split("."):
        if part == "extensions":
            root = config.extensions
            continue
        root = getattr(root, part)
    return bool(root)


@dataclass(frozen=True)
class ExtensionSpec:
    """
    Registered post-fit extension.

    ``run`` mutates ``ctx.out`` (and optional pipeline fields on ``ctx``).
    When ``config_key`` is set and resolves to ``False``, the extension is skipped
    (not executed; not marked executed for dependency purposes).
    """

    name: str
    artifact_key: str
    run: Callable[[ExtensionContext], None]
    artifact_tier: ArtifactTier = "research"
    dependencies: tuple[str, ...] = ()
    priority: int = 0
    config_key: str | None = None
    report_keys: tuple[str, ...] = field(default_factory=tuple)
    side_effect_only: bool = False
    optional_dependency: bool = False
    allow_duplicate_artifact_key: bool = False

    @property
    def output_keys(self) -> tuple[str, ...]:
        """Backward-compatible alias for report keys written to ``ctx.out``."""
        return self.report_keys


class ExtensionRegistry:
    """Ordered, duplicate-free extension registry."""

    def __init__(self) -> None:
        self._specs: dict[str, ExtensionSpec] = {}
        self._artifact_keys: dict[str, str] = {}

    def register(self, spec: ExtensionSpec) -> None:
        if spec.name in self._specs:
            raise ExtensionRegistrationError(f"extension already registered: {spec.name!r}")
        if (
            spec.artifact_key in self._artifact_keys
            and not spec.allow_duplicate_artifact_key
        ):
            existing = self._artifact_keys[spec.artifact_key]
            raise ExtensionRegistrationError(
                f"artifact_key {spec.artifact_key!r} already registered by {existing!r}"
            )
        self._specs[spec.name] = spec
        if not spec.allow_duplicate_artifact_key:
            self._artifact_keys[spec.artifact_key] = spec.name

    def get(self, name: str) -> ExtensionSpec:
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs_sorted(self) -> tuple[ExtensionSpec, ...]:
        return tuple(sorted(self._specs.values(), key=lambda s: (s.priority, s.name)))

    def validate(self) -> None:
        """Validate dependency graph at plan construction time."""
        known = set(self._specs)
        for spec in self._specs.values():
            missing = [d for d in spec.dependencies if d not in known]
            if missing:
                raise ExtensionDependencyError(
                    f"extension {spec.name!r} has unknown dependencies: {missing}"
                )

    def run_all(self, ctx: ExtensionContext) -> None:
        self.validate()
        executed: set[str] = set()
        for spec in self.specs_sorted():
            for dep in spec.dependencies:
                if dep not in executed:
                    raise ExtensionDependencyError(
                        f"extension {spec.name!r} requires {dep!r} but it has not run "
                        f"(disabled, skipped, or ordered after dependent)"
                    )
            if spec.config_key is not None and not _resolve_config_flag(ctx.config, spec.config_key):
                continue
            spec.run(ctx)
            executed.add(spec.name)

    def clear(self) -> None:
        self._specs.clear()
        self._artifact_keys.clear()


_GLOBAL_REGISTRY: ExtensionRegistry | None = None
_PLUGINS_ENSURED = False


def _global_registry() -> ExtensionRegistry:
    """Return the process-global registry, creating it if needed (no plugin load)."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ExtensionRegistry()
    return _GLOBAL_REGISTRY


def register_extension(spec: ExtensionSpec) -> None:
    """Register one extension on the process-global registry."""
    _global_registry().register(spec)


def get_registered_extensions() -> tuple[ExtensionSpec, ...]:
    """Return registered specs in deterministic execution order."""
    return get_extension_registry().specs_sorted()


def get_extension_registry() -> ExtensionRegistry:
    global _PLUGINS_ENSURED
    reg = _global_registry()
    if not _PLUGINS_ENSURED:
        from mmm.evaluation.extensions import ensure_extensions_registered

        ensure_extensions_registered()
    return reg


def reset_extension_registry_for_tests() -> ExtensionRegistry:
    """Replace global registry and re-register plugins (tests only)."""
    global _GLOBAL_REGISTRY, _PLUGINS_ENSURED
    _GLOBAL_REGISTRY = ExtensionRegistry()
    _PLUGINS_ENSURED = False
    from mmm.evaluation.extensions import ensure_extensions_registered

    ensure_extensions_registered(force=True)
    return _global_registry()
