"""ScenarioBuilder MVP — deterministic world_truth from scenario specifications (truth only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from mmm.validation.synthetic.generators import compose_archetype_truth, write_world_truth

SCENARIO_BUILDER_VERSION = "scenario_builder_v1.0.0"

Family = Literal["baseline", "replay"]
NoiseLevel = Literal["low", "medium", "high"]
CorrelationLevel = Literal["low", "medium", "severe"]
Seasonality = Literal["none", "mild", "strong"]
ExperimentQuality = Literal["none", "weak", "medium", "high"]
Missingness = Literal["none", "mild"]

_VALID_FAMILIES = frozenset({"baseline", "replay"})
_VALID_NOISE = frozenset({"low", "medium", "high"})
_VALID_CORRELATION = frozenset({"low", "medium", "severe"})
_VALID_SEASONALITY = frozenset({"none", "mild", "strong"})
_VALID_EXPERIMENT_QUALITY = frozenset({"none", "weak", "medium", "high"})
_VALID_MISSINGNESS = frozenset({"none", "mild"})


@dataclass(frozen=True)
class ScenarioSpec:
    world_id: str
    family: str
    seed: int
    n_geos: int
    n_periods: int
    channels: tuple[str, ...]
    noise_level: str
    correlation_level: str
    seasonality: str
    drift: bool
    experiment_quality: str
    privacy_loss: bool
    missingness: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioSpec:
        return cls(
            world_id=str(data["world_id"]),
            family=str(data["family"]),
            seed=int(data["seed"]),
            n_geos=int(data["n_geos"]),
            n_periods=int(data["n_periods"]),
            channels=tuple(str(c) for c in data["channels"]),
            noise_level=str(data["noise_level"]),
            correlation_level=str(data["correlation_level"]),
            seasonality=str(data["seasonality"]),
            drift=bool(data["drift"]),
            experiment_quality=str(data["experiment_quality"]),
            privacy_loss=bool(data["privacy_loss"]),
            missingness=str(data["missingness"]),
        )


def validate_scenario_spec(spec: ScenarioSpec) -> None:
    if spec.family not in _VALID_FAMILIES:
        raise ValueError(f"invalid family {spec.family!r}")
    if spec.noise_level not in _VALID_NOISE:
        raise ValueError(f"invalid noise_level {spec.noise_level!r}")
    if spec.correlation_level not in _VALID_CORRELATION:
        raise ValueError(f"invalid correlation_level {spec.correlation_level!r}")
    if spec.seasonality not in _VALID_SEASONALITY:
        raise ValueError(f"invalid seasonality {spec.seasonality!r}")
    if spec.experiment_quality not in _VALID_EXPERIMENT_QUALITY:
        raise ValueError(f"invalid experiment_quality {spec.experiment_quality!r}")
    if spec.missingness not in _VALID_MISSINGNESS:
        raise ValueError(f"invalid missingness {spec.missingness!r}")
    if spec.n_geos < 1:
        raise ValueError("n_geos must be >= 1")
    if spec.n_periods < 4:
        raise ValueError("n_periods must be >= 4")
    if not spec.channels:
        raise ValueError("channels must be non-empty")
    if spec.family == "replay" and spec.experiment_quality == "none" and spec.drift:
        pass  # allowed: replay bundle without units


def build_world_truth(spec: ScenarioSpec) -> dict[str, Any]:
    """Build ``world_truth.json`` from a scenario spec (no derived artifacts)."""
    validate_scenario_spec(spec)
    tags = [
        f"family:{spec.family}",
        f"builder:{SCENARIO_BUILDER_VERSION}",
    ]
    if spec.drift:
        tags.append("drift:on")
    if spec.privacy_loss:
        tags.append("privacy:on")

    truth = compose_archetype_truth(
        family=spec.family,
        seed=spec.seed,
        world_id=spec.world_id,
        n_geos=spec.n_geos,
        n_periods=spec.n_periods,
        channels=list(spec.channels),
        noise_level=spec.noise_level,
        correlation_level=spec.correlation_level,
        seasonality=spec.seasonality,
        drift=spec.drift,
        experiment_quality=spec.experiment_quality,
        privacy_loss=spec.privacy_loss,
        missingness=spec.missingness,
        scenario_tags=tags,
        description=f"ScenarioBuilder {spec.family} ({spec.world_id}, seed={spec.seed})",
    )
    return truth


def write_scenario_world(bundle_dir: str | Path, spec: ScenarioSpec) -> Path:
    """Write ``world_truth.json`` only; materializer owns panel/replay/checksums."""
    validate_scenario_spec(spec)
    return write_world_truth(bundle_dir, build_world_truth(spec))


# Committed smoke scenario definitions
WORLD_005_LOW_NOISE = ScenarioSpec(
    world_id="WORLD-005-scenario-low-noise",
    family="baseline",
    seed=5005,
    n_geos=3,
    n_periods=12,
    channels=("search", "social"),
    noise_level="low",
    correlation_level="low",
    seasonality="none",
    drift=False,
    experiment_quality="none",
    privacy_loss=False,
    missingness="none",
)

WORLD_006_HIGH_COLLINEARITY = ScenarioSpec(
    world_id="WORLD-006-scenario-high-collinearity",
    family="baseline",
    seed=5006,
    n_geos=2,
    n_periods=12,
    channels=("search", "social"),
    noise_level="medium",
    correlation_level="severe",
    seasonality="none",
    drift=False,
    experiment_quality="none",
    privacy_loss=False,
    missingness="none",
)

WORLD_007_REPLAY_DRIFT = ScenarioSpec(
    world_id="WORLD-007-scenario-replay-drift",
    family="replay",
    seed=5007,
    n_geos=2,
    n_periods=14,
    channels=("search",),
    noise_level="medium",
    correlation_level="low",
    seasonality="mild",
    drift=True,
    experiment_quality="medium",
    privacy_loss=False,
    missingness="none",
)

COMMITTED_SCENARIO_WORLDS: tuple[ScenarioSpec, ...] = (
    WORLD_005_LOW_NOISE,
    WORLD_006_HIGH_COLLINEARITY,
    WORLD_007_REPLAY_DRIFT,
)
