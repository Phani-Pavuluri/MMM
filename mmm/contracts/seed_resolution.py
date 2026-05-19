"""Unified seed contract — resolve master and child seeds without changing model math."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from mmm.config.schema import MMMConfig

SEED_CONTRACT_VERSION = "seed_resolution_v1"

InheritanceSource = Literal["explicit", "inherited_from_master"]

# Child keys that reuse master_seed directly when unset (training / MCMC entry points).
_DIRECT_INHERIT_KEYS = frozenset(
    {"ridge_bo.sampler_seed", "cv.geo_blocked_seed", "bayesian.nuts_seed"}
)


def derive_child_seed(master_seed: int, label: str) -> int:
    """Deterministic derived child seed from master (for diagnostics / extensions)."""
    digest = hashlib.sha256(f"{int(master_seed)}|{label}|{SEED_CONTRACT_VERSION}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def resolve_seed_contract(config: MMMConfig) -> dict[str, Any]:
    """
    Build ``seed_resolution`` artifact and apply resolved values to ``config`` in place.

    Does not alter optimizer or planning mathematics — only centralizes RNG entry seeds.
    """
    master = int(config.random_seed)
    inheritance: dict[str, InheritanceSource] = {}
    resolved: dict[str, int] = {}

    def _resolve(key: str, explicit: int | None) -> int:
        if explicit is not None:
            resolved[key] = int(explicit)
            inheritance[key] = "explicit"
            return resolved[key]
        v = master if key in _DIRECT_INHERIT_KEYS else derive_child_seed(master, key)
        resolved[key] = v
        inheritance[key] = "inherited_from_master"
        return v

    config.ridge_bo.sampler_seed = _resolve(
        "ridge_bo.sampler_seed",
        int(config.ridge_bo.sampler_seed) if config.ridge_bo.sampler_seed is not None else None,
    )
    config.cv.geo_blocked_seed = _resolve(
        "cv.geo_blocked_seed",
        int(config.cv.geo_blocked_seed) if config.cv.geo_blocked_seed is not None else None,
    )
    config.bootstrap_seed = _resolve(
        "bootstrap_seed",
        int(config.bootstrap_seed) if config.bootstrap_seed is not None else None,
    )
    config.extension_seed = _resolve(
        "extension_seed",
        int(config.extension_seed) if config.extension_seed is not None else None,
    )
    config.experiment_scheduler_seed = _resolve(
        "experiment_scheduler_seed",
        int(config.experiment_scheduler_seed) if config.experiment_scheduler_seed is not None else None,
    )
    config.simulation_seed = _resolve(
        "simulation_seed",
        int(config.simulation_seed) if config.simulation_seed is not None else None,
    )

    if config.framework.value == "bayesian":
        config.bayesian.nuts_seed = _resolve(
            "bayesian.nuts_seed",
            int(config.bayesian.nuts_seed) if config.bayesian.nuts_seed is not None else None,
        )

    return {
        "policy_version": SEED_CONTRACT_VERSION,
        "master_seed": master,
        "resolved_child_seeds": dict(resolved),
        "inheritance_source": dict(inheritance),
    }
