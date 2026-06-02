"""Research-only media transforms for Bayes-H5 sandbox (not production feature engineering)."""

from __future__ import annotations

from typing import Any

import numpy as np

TRANSFORM_REGISTRY_ID = "bayes_h5_media_transform_registry_v1"

TRANSFORM_IDS: tuple[str, ...] = (
    "identity",
    "geometric_adstock",
    "hill_saturation",
    "adstock_then_saturation",
)

# Generative world kinds where H5 fit uses identity on observed media (not a transform probe).
GENERATIVE_KINDS_IDENTITY_FIT: frozenset[str] = frozenset(
    {
        "identity",
        "linear",
        "correlated",
        "weak_signal",
    }
)

# Real historical panels have no known generative transform truth (shadow / research).
REAL_PANEL_GENERATIVE_KINDS: frozenset[str] = frozenset(
    {
        "real_panel",
        "unknown",
        "none",
    }
)

# Registry transform ids used in aligned transform-probe worlds.
MEDIA_TRANSFORM_IDS: frozenset[str] = frozenset(
    {
        "identity",
        "geometric_adstock",
        "hill_saturation",
        "adstock_then_saturation",
        "adstock",
        "saturation",
    }
)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-6)


def apply_identity(x: np.ndarray, *, params: dict[str, Any] | None = None) -> np.ndarray:
    del params
    return _standardize(x)


def apply_geometric_adstock(x: np.ndarray, *, params: dict[str, Any] | None = None) -> np.ndarray:
    p = params or {}
    decay = float(p.get("decay", 0.7))
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = float(v) + decay * carry
        out[i] = carry
    return _standardize(out)


def apply_hill_saturation(x: np.ndarray, *, params: dict[str, Any] | None = None) -> np.ndarray:
    p = params or {}
    half = float(p.get("half", 2.0))
    slope = float(p.get("slope", 1.5))
    x_pos = np.maximum(np.asarray(x, dtype=float), 0.0)
    saturated = (x_pos**slope) / (half**slope + x_pos**slope + 1e-6)
    return _standardize(saturated)


def apply_adstock_then_saturation(x: np.ndarray, *, params: dict[str, Any] | None = None) -> np.ndarray:
    p = params or {}
    adstocked = apply_geometric_adstock(x, params={"decay": float(p.get("decay", 0.7))})
    # adstock helper already standardizes; re-run hill on de-standardized proxy via abs+offset
    raw_proxy = np.maximum(adstocked - adstocked.min() + 0.1, 0.0)
    return apply_hill_saturation(raw_proxy, params=p)


_TRANSFORM_FNS: dict[str, Any] = {
    "identity": apply_identity,
    "geometric_adstock": apply_geometric_adstock,
    "hill_saturation": apply_hill_saturation,
    "adstock_then_saturation": apply_adstock_then_saturation,
}


def apply_channel_transform(
    x: np.ndarray,
    transform_id: str,
    *,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply a registered transform to a 1d channel series (deterministic)."""
    if transform_id not in _TRANSFORM_FNS:
        raise KeyError(f"unknown H5 transform {transform_id!r}; known: {sorted(_TRANSFORM_FNS)}")
    return _TRANSFORM_FNS[transform_id](x, params=params)


def apply_media_transforms_matrix(
    x: np.ndarray,
    channels: list[str],
    transforms_by_channel: dict[str, str],
    *,
    transform_params_by_channel: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply per-channel transforms; x shape (n_obs, n_channels) aligned to channels."""
    params_map = transform_params_by_channel or {}
    out = np.zeros_like(x, dtype=float)
    for ci, ch in enumerate(channels):
        tid = transforms_by_channel.get(ch, "identity")
        out[:, ci] = apply_channel_transform(x[:, ci], tid, params=params_map.get(ch))
    return out


def transforms_aligned(generative_transform: str, fitted_transform_id: str) -> bool:
    """Whether generative outcome and fitted media transform are aligned for H5 diagnostics."""
    gen = str(generative_transform)
    fitted = str(fitted_transform_id)
    if gen == fitted:
        return True
    if gen in GENERATIVE_KINDS_IDENTITY_FIT and fitted == "identity":
        return True
    if gen == "adstock_then_saturation" and fitted == "geometric_adstock":
        return False
    return (gen in ("adstock", "geometric_adstock") and fitted == "geometric_adstock") or (
        gen in ("saturation", "hill_saturation") and fitted == "hill_saturation"
    )


def is_real_panel_generative(generative_transform: str) -> bool:
    return str(generative_transform) in REAL_PANEL_GENERATIVE_KINDS


def compute_transform_mismatch_detected(
    generative_transform: str,
    fitted_transform_id: str,
    *,
    transform_mismatch_mode: str = "aligned",
) -> bool:
    """Whether the fit should flag transform mismatch (intentional probe or true misalignment)."""
    if is_real_panel_generative(generative_transform):
        return transform_mismatch_mode == "intentional_mismatch"
    if transform_mismatch_mode == "intentional_mismatch":
        return True
    return not transforms_aligned(generative_transform, fitted_transform_id)


def is_transform_probe_generative(generative_transform: str) -> bool:
    """True when generative transform is a media-transform probe (not linear/weak-ID kind)."""
    return str(generative_transform) in MEDIA_TRANSFORM_IDS or str(generative_transform) in (
        "adstock",
        "saturation",
    )


def list_transform_registry() -> dict[str, Any]:
    return {
        "registry_id": TRANSFORM_REGISTRY_ID,
        "transform_ids": list(TRANSFORM_IDS),
        "research_only": True,
        "wired_to_production_feature_engineering": False,
    }
