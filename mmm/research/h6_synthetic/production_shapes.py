"""H6a — production-shaped synthetic DMA world generator (research only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import (
    BayesianBackend,
    CVConfig,
    Framework,
    MMMConfig,
    ModelForm,
    PoolingMode,
    RunEnvironment,
)
from mmm.data.schema import PanelSchema
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    SAMPLER_FAST,
    RecoveryWorldSpec,
)
from mmm.research.h6_synthetic.vertical_controls import (
    VerticalControlProfile,
    control_truth_for_profile,
    get_vertical_profile,
)

H6_SEED = 6600

WORLD_H6_PILOT_RETAIL_FULL = "WORLD-H6-PILOT-RETAIL-FULL-CONTROLS"
WORLD_H6_PILOT_RETAIL_OMITTED = "WORLD-H6-PILOT-RETAIL-OMITTED-CONTROLS"
WORLD_H6_PILOT_RETAIL_MEDIA_CORR = "WORLD-H6-PILOT-RETAIL-MEDIA-CORRELATED-CONTROLS"
WORLD_H6_PILOT_CPG_FULL = "WORLD-H6-PILOT-CPG-FULL-CONTROLS"
WORLD_H6_PILOT_AUTO_OMITTED = "WORLD-H6-PILOT-AUTO-OMITTED-CONTROLS"

H6_PILOT_WORLD_IDS: tuple[str, ...] = (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    WORLD_H6_PILOT_RETAIL_MEDIA_CORR,
    WORLD_H6_PILOT_CPG_FULL,
    WORLD_H6_PILOT_AUTO_OMITTED,
)

StressVariant = Literal[
    "full_controls",
    "omitted_controls",
    "mis_specified_controls",
    "media_correlated_controls",
]
ScaleProfile = Literal["pilot", "production"]

_PILOT_GEOS = 20
_PILOT_WEEKS = 52
_PILOT_CHANNELS = (
    "tv",
    "search",
    "social",
    "display",
    "ctv",
    "radio",
    "local_flyer",
)

_PROD_GEOS = 200
_PROD_WEEKS = 130
_PROD_CHANNELS = (
    "tv",
    "search",
    "social",
    "display",
    "ctv",
    "audio",
    "radio",
    "local_flyer",
)

_COLLINEARITY_BLOCKS: dict[str, tuple[str, ...]] = {
    "digital_block": ("display", "ctv", "audio"),
}


def _collinearity_blocks_for_channels(channels: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    blocks: dict[str, tuple[str, ...]] = {}
    for name, block in _COLLINEARITY_BLOCKS.items():
        present = tuple(c for c in block if c in channels)
        if len(present) >= 2:
            blocks[name] = present
    return blocks
_NATIONAL_CHANNELS = ("tv",)
_SPARSE_CHANNELS = ("radio", "local_flyer")


@dataclass(frozen=True)
class ProductionShapedWorldSpec:
    """Known-truth production-shaped synthetic DMA world (H6)."""

    world_id: str
    scale: ScaleProfile
    vertical_id: str
    stress_variant: StressVariant
    n_geos: int
    n_weeks: int
    channels: tuple[str, ...]
    collinearity_blocks: dict[str, tuple[str, ...]]
    national_channels: tuple[str, ...]
    sparse_channels: tuple[str, ...]
    noise_sigma: float
    true_mu_c: dict[str, float]
    true_tau_c: dict[str, float]
    true_beta_gc: dict[str, dict[str, float]]
    true_alpha_g: dict[str, float]
    transform_truth: dict[str, dict[str, float]]
    control_truth: dict[str, Any]
    panel_seed: int = H6_SEED
    mcmc_seed: int = H6_SEED
    expected_diagnostic_behavior: dict[str, Any] = field(default_factory=dict)

    @property
    def geo_order(self) -> tuple[str, ...]:
        return tuple(f"DMA_{i:03d}" for i in range(self.n_geos))

    @property
    def active_controls(self) -> tuple[str, ...]:
        return tuple(self.control_truth.get("active_controls") or ())

    @property
    def known_truth(self) -> dict[str, Any]:
        return {
            "true_mu_c": dict(self.true_mu_c),
            "true_tau_c": dict(self.true_tau_c),
            "true_beta_gc": {g: dict(ch) for g, ch in self.true_beta_gc.items()},
            "true_alpha_g": dict(self.true_alpha_g),
            "noise_sigma": self.noise_sigma,
            "transform_truth": dict(self.transform_truth),
            "control_truth": dict(self.control_truth),
            "collinearity_blocks": {k: list(v) for k, v in self.collinearity_blocks.items()},
            "national_channels": list(self.national_channels),
            "sparse_channels": list(self.sparse_channels),
            "stress_variant": self.stress_variant,
            "vertical_id": self.vertical_id,
            "scale": self.scale,
        }


def _scale_dims(scale: ScaleProfile) -> tuple[int, int, tuple[str, ...]]:
    if scale == "pilot":
        return _PILOT_GEOS, _PILOT_WEEKS, _PILOT_CHANNELS
    return _PROD_GEOS, _PROD_WEEKS, _PROD_CHANNELS


def _geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t in range(len(x)):
        carry = x[t] + decay * carry
        out[t] = carry
    return out


def _hill_transform(x: np.ndarray, half: float, slope: float) -> np.ndarray:
    x_pos = np.maximum(x, 0.0)
    return x_pos**slope / (half**slope + x_pos**slope + 1e-9)


def _default_transform_truth(channels: tuple[str, ...]) -> dict[str, dict[str, float]]:
    truth: dict[str, dict[str, float]] = {}
    for ch in channels:
        truth[ch] = {
            "decay": 0.5 if ch in ("tv", "ctv", "audio") else 0.3,
            "hill_half": 2.5,
            "hill_slope": 1.8,
        }
    return truth


def _build_beta_surfaces(
    geos: tuple[str, ...],
    channels: tuple[str, ...],
    mu: dict[str, float],
    tau: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    beta: dict[str, dict[str, float]] = {}
    for geo in geos:
        beta[geo] = {}
        for ch in channels:
            beta[geo][ch] = float(mu[ch] + rng.normal(0.0, tau[ch]))
    return beta


def _active_controls_for_variant(
    profile: VerticalControlProfile,
    variant: StressVariant,
) -> tuple[str, ...]:
    if variant == "full_controls":
        return profile.all_controls
    if variant == "omitted_controls":
        return profile.optional_controls
    if variant == "mis_specified_controls":
        # Include optional + one wrong proxy not in profile truth
        return profile.optional_controls + ("mis_specified_proxy",)
    if variant == "media_correlated_controls":
        return profile.required_controls + profile.optional_controls
    raise ValueError(f"unknown stress variant: {variant}")


def build_production_shaped_world(
    *,
    world_id: str,
    scale: ScaleProfile,
    vertical_id: str,
    stress_variant: StressVariant = "full_controls",
    panel_seed: int = H6_SEED,
) -> ProductionShapedWorldSpec:
    """Construct a production-shaped world spec with known truth."""
    n_geos, n_weeks, channels = _scale_dims(scale)
    profile = get_vertical_profile(vertical_id)
    rng = np.random.default_rng(panel_seed)
    geos = tuple(f"DMA_{i:03d}" for i in range(n_geos))

    mu = {ch: 0.28 + 0.04 * (i % 5) for i, ch in enumerate(channels)}
    tau = {ch: (0.12 if ch in _SPARSE_CHANNELS else 0.07) for ch in channels}
    beta = _build_beta_surfaces(geos, channels, mu, tau, rng)
    alpha = {g: 8.5 + 0.15 * (int(g.split("_")[1]) % 10) for g in geos}

    sparse_tail = set(geos[int(0.7 * n_geos) :])
    for g in sparse_tail:
        for ch in _SPARSE_CHANNELS:
            if ch in beta[g]:
                beta[g][ch] = float(beta[g][ch] + rng.normal(0.4, 0.15))

    active_controls = _active_controls_for_variant(profile, stress_variant)
    ctrl_truth = control_truth_for_profile(profile, active_controls=active_controls)

    if stress_variant == "omitted_controls":
        ctrl_truth["omitted_required_controls"] = list(profile.required_controls)
    if stress_variant == "mis_specified_controls":
        ctrl_truth["mis_specification_note"] = (
            "mis_specified_proxy included in panel; true effects only on profile controls"
        )
    if stress_variant == "media_correlated_controls":
        ctrl_truth["media_correlation_target"] = "tv"

    exp: dict[str, Any] = {
        "generative_transform": "adstock_saturation",
        "h6_lane": True,
        "scale": scale,
        "vertical_id": vertical_id,
        "stress_variant": stress_variant,
        "transform_mismatch_warning_expected": True,
        "collinearity_warning_expected": True,
    }
    if stress_variant == "omitted_controls":
        exp["omitted_control_stress"] = True
        exp["forbidden_claims_when_controls_missing"] = [
            "do_not_attribute_omitted_control_lift_to_media",
            "do_not_publish_incrementality_without_required_controls",
        ]
    if stress_variant in ("media_correlated_controls", "mis_specified_controls"):
        exp["confounding_stress"] = True

    blocks = _collinearity_blocks_for_channels(channels)

    return ProductionShapedWorldSpec(
        world_id=world_id,
        scale=scale,
        vertical_id=vertical_id,
        stress_variant=stress_variant,
        n_geos=n_geos,
        n_weeks=n_weeks,
        channels=channels,
        collinearity_blocks=blocks,
        national_channels=tuple(c for c in _NATIONAL_CHANNELS if c in channels),
        sparse_channels=tuple(c for c in _SPARSE_CHANNELS if c in channels),
        noise_sigma=0.14,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g=alpha,
        transform_truth=_default_transform_truth(channels),
        control_truth=ctrl_truth,
        panel_seed=panel_seed,
        mcmc_seed=panel_seed,
        expected_diagnostic_behavior=exp,
    )


def materialize_h6_panel(
    spec: ProductionShapedWorldSpec,
    *,
    panel_seed: int | None = None,
) -> pd.DataFrame:
    """Materialize production-shaped panel with known generative truth."""
    rng = np.random.default_rng(panel_seed if panel_seed is not None else spec.panel_seed)
    geos = spec.geo_order
    n_weeks = spec.n_weeks
    channels = spec.channels
    profile = get_vertical_profile(spec.vertical_id)
    active_controls = spec.active_controls

    week0 = pd.Timestamp("2022-01-03")
    rows: list[dict[str, Any]] = []

    national_tv = rng.uniform(8.0, 25.0, size=n_weeks)

    block_latent = rng.normal(0, 1, size=n_weeks)
    digital_block = spec.collinearity_blocks.get("digital_block", ())

    for gi, geo in enumerate(geos):
        is_sparse_geo = geo in set(geos[int(0.7 * len(geos)) :])
        media_raw: dict[str, np.ndarray] = {}

        for ch in channels:
            if ch in spec.national_channels:
                noise = rng.normal(0, 0.02, size=n_weeks)
                media_raw[ch] = national_tv * (1.0 + noise)
            elif ch in digital_block:
                media_raw[ch] = np.maximum(0.5, 3.0 * block_latent + rng.normal(0, 0.4, n_weeks) + 4.0)
            elif ch in spec.sparse_channels and is_sparse_geo:
                media_raw[ch] = rng.uniform(0.0, 0.3, size=n_weeks)
            else:
                media_raw[ch] = rng.uniform(1.0, 8.0, size=n_weeks)

        transformed: dict[str, np.ndarray] = {}
        for ch in channels:
            tr = spec.transform_truth[ch]
            ad = _geometric_adstock(media_raw[ch], tr["decay"])
            transformed[ch] = _hill_transform(ad, tr["hill_half"], tr["hill_slope"])

        ctrl_vals: dict[str, np.ndarray] = {}
        for ctrl in active_controls:
            if ctrl == "mis_specified_proxy":
                ctrl_vals[ctrl] = 0.5 * national_tv + rng.normal(0, 0.1, n_weeks)
                continue
            if ctrl in profile.control_effects_log or ctrl in profile.required_controls:
                if spec.stress_variant == "media_correlated_controls" and ctrl == profile.required_controls[0]:
                    ctrl_vals[ctrl] = 0.35 * national_tv + rng.normal(0, 0.15, n_weeks)
                else:
                    ctrl_vals[ctrl] = rng.normal(0, 1, n_weeks)
            else:
                ctrl_vals[ctrl] = rng.normal(0, 1, n_weeks)

        season = 0.06 * np.sin(2 * np.pi * np.arange(n_weeks) / 52.0)
        shock = np.zeros(n_weeks)
        if gi % 17 == 0:
            shock_start = int(rng.integers(10, max(11, n_weeks - 5)))
            shock[shock_start : shock_start + 2] = rng.uniform(0.15, 0.35)

        alpha = spec.true_alpha_g[geo]
        log_y = alpha + season + shock
        for ch in channels:
            log_y = log_y + spec.true_beta_gc[geo][ch] * transformed[ch]
        for ctrl, x in ctrl_vals.items():
            eff = spec.control_truth.get("control_effects_log", {}).get(ctrl, 0.0)
            log_y = log_y + eff * ((x - x.mean()) / (x.std() + 1e-6))
        log_y = log_y + rng.normal(0.0, spec.noise_sigma, size=n_weeks)

        revenue = np.exp(log_y)
        for t in range(n_weeks):
            row: dict[str, Any] = {
                "geo_id": geo,
                "week_start_date": week0 + pd.Timedelta(weeks=t),
                "revenue": float(max(revenue[t], 1e-3)),
            }
            for ch in channels:
                row[ch] = float(max(media_raw[ch][t], 0.0))
            for ctrl, arr in ctrl_vals.items():
                row[ctrl] = float(arr[t])
            rows.append(row)

    return pd.DataFrame(rows)


def h6_panel_schema(spec: ProductionShapedWorldSpec) -> PanelSchema:
    return PanelSchema(
        geo_column="geo_id",
        week_column="week_start_date",
        target_column="revenue",
        channel_columns=spec.channels,
        control_columns=spec.active_controls,
    )


def h6_ridge_config(spec: ProductionShapedWorldSpec) -> MMMConfig:
    """Production Ridge baseline config (research lane — not prod promotion)."""
    min_train = max(6, min(12, spec.n_weeks - 4))
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        run_environment=RunEnvironment.RESEARCH,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        random_seed=spec.panel_seed,
        data={
            "path": None,
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": list(spec.channels),
            "control_columns": list(spec.active_controls),
            "data_version_id": f"{spec.world_id}-h6",
        },
        cv=CVConfig(
            mode="rolling",
            n_splits=2,
            min_train_weeks=min_train,
            horizon_weeks=2,
        ),
        ridge_bo={"n_trials": 8, "sampler_seed": spec.panel_seed},
        transforms={"adstock": "geometric", "saturation": "hill"},
    )


def h6_h5_config(
    spec: ProductionShapedWorldSpec,
    *,
    fast_mcmc: bool = True,
    nuts_seed: int | None = None,
) -> MMMConfig:
    """Research Bayes-H5 sandbox config for H6 worlds."""
    bayesian: dict[str, Any] = {
        "backend": BayesianBackend.PYMC,
        "nuts_seed": int(nuts_seed if nuts_seed is not None else spec.mcmc_seed),
        "prior_predictive_draws": 0,
        "posterior_predictive_draws": 0,
    }
    if fast_mcmc:
        bayesian.update(SAMPLER_FAST)
    return MMMConfig(
        framework=Framework.BAYESIAN,
        run_environment=RunEnvironment.RESEARCH,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        data={
            "path": None,
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": list(spec.channels),
            "control_columns": list(spec.active_controls),
        },
        bayesian=bayesian,
    )


def as_recovery_world_spec(spec: ProductionShapedWorldSpec) -> RecoveryWorldSpec:
    """Map H6 world to RecoveryWorldSpec for shared H5 recovery metrics."""
    weeks = {g: spec.n_weeks for g in spec.geo_order}
    return RecoveryWorldSpec(
        world_id=spec.world_id,
        geo_order=spec.geo_order,
        channels=spec.channels,
        weeks_by_geo=weeks,
        noise_sigma=spec.noise_sigma,
        true_mu_c=spec.true_mu_c,
        true_tau_c=spec.true_tau_c,
        true_beta_gc=spec.true_beta_gc,
        true_alpha_g=spec.true_alpha_g,
        geo_hierarchy={g: {"region": f"R{int(g.split('_')[1]) % 5}"} for g in spec.geo_order},
        sparse_geos=tuple(g for g in spec.geo_order[int(0.7 * len(spec.geo_order)) :]),
        expected_diagnostic_behavior=dict(spec.expected_diagnostic_behavior),
        mcmc_seed=spec.mcmc_seed,
    )


def forbidden_claims_for_h6_world(spec: ProductionShapedWorldSpec) -> list[str]:
    """Governed forbidden claims for H6 stress variants (H6d/H6f)."""
    claims = list(
        spec.expected_diagnostic_behavior.get("forbidden_claims_when_controls_missing") or []
    )
    if spec.stress_variant == "omitted_controls":
        claims.extend(
            [
                "do_not_attribute_omitted_control_lift_to_media",
                "do_not_publish_incrementality_without_required_controls",
            ]
        )
    if spec.stress_variant == "media_correlated_controls":
        claims.append("media_correlated_controls_may_inflate_attribution_to_tv")
    return sorted(set(claims))


def materialize_h6_truth_artifact(spec: ProductionShapedWorldSpec) -> dict[str, Any]:
    """Full known-truth artifact for H6 validation archives."""
    return {
        "world_id": spec.world_id,
        "lane": "bayes_h6_synthetic",
        "known_truth": spec.known_truth,
        "production_flags": {
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "optimizer_enabled": False,
            "decision_surface_enabled": False,
            "recommendations_enabled": False,
        },
        "panel_shape": {
            "n_geos": spec.n_geos,
            "n_weeks": spec.n_weeks,
            "n_channels": len(spec.channels),
            "n_controls": len(spec.active_controls),
        },
        "forbidden_claims": forbidden_claims_for_h6_world(spec),
        "outputs_are_diagnostic_only": True,
    }


_WORLDS: dict[str, ProductionShapedWorldSpec] = {}


def _register_worlds() -> None:
    if _WORLDS:
        return
    _WORLDS[WORLD_H6_PILOT_RETAIL_FULL] = build_production_shaped_world(
        world_id=WORLD_H6_PILOT_RETAIL_FULL,
        scale="pilot",
        vertical_id="retail",
        stress_variant="full_controls",
        panel_seed=H6_SEED,
    )
    _WORLDS[WORLD_H6_PILOT_RETAIL_OMITTED] = build_production_shaped_world(
        world_id=WORLD_H6_PILOT_RETAIL_OMITTED,
        scale="pilot",
        vertical_id="retail",
        stress_variant="omitted_controls",
        panel_seed=H6_SEED + 1,
    )
    _WORLDS[WORLD_H6_PILOT_RETAIL_MEDIA_CORR] = build_production_shaped_world(
        world_id=WORLD_H6_PILOT_RETAIL_MEDIA_CORR,
        scale="pilot",
        vertical_id="retail",
        stress_variant="media_correlated_controls",
        panel_seed=H6_SEED + 2,
    )
    _WORLDS[WORLD_H6_PILOT_CPG_FULL] = build_production_shaped_world(
        world_id=WORLD_H6_PILOT_CPG_FULL,
        scale="pilot",
        vertical_id="cpg",
        stress_variant="full_controls",
        panel_seed=H6_SEED + 3,
    )
    _WORLDS[WORLD_H6_PILOT_AUTO_OMITTED] = build_production_shaped_world(
        world_id=WORLD_H6_PILOT_AUTO_OMITTED,
        scale="pilot",
        vertical_id="auto",
        stress_variant="omitted_controls",
        panel_seed=H6_SEED + 4,
    )


def get_h6_world(world_id: str) -> ProductionShapedWorldSpec:
    _register_worlds()
    if world_id not in _WORLDS:
        raise KeyError(f"unknown H6 world: {world_id!r}; known: {sorted(_WORLDS)}")
    return _WORLDS[world_id]


def list_h6_world_ids() -> tuple[str, ...]:
    _register_worlds()
    return tuple(_WORLDS)


def list_h6_confounding_world_ids() -> tuple[str, ...]:
    """Worlds used for H6d omitted/mis-specified/media-correlated stress."""
    return (
        WORLD_H6_PILOT_RETAIL_FULL,
        WORLD_H6_PILOT_RETAIL_OMITTED,
        WORLD_H6_PILOT_RETAIL_MEDIA_CORR,
        WORLD_H6_PILOT_AUTO_OMITTED,
    )
