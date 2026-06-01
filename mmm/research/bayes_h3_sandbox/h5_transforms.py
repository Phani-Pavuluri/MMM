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
    """Whether generative and fitted transform ids are considered aligned for H5 diagnostics."""
    if generative_transform == fitted_transform_id:
        return True
    if generative_transform == "adstock_then_saturation" and fitted_transform_id == "geometric_adstock":
        return False
    return (
        generative_transform in ("adstock", "geometric_adstock")
        and fitted_transform_id == "geometric_adstock"
    ) or (
        generative_transform in ("saturation", "hill_saturation")
        and fitted_transform_id == "hill_saturation"
    )


def list_transform_registry() -> dict[str, Any]:
    return {
        "registry_id": TRANSFORM_REGISTRY_ID,
        "transform_ids": list(TRANSFORM_IDS),
        "research_only": True,
        "wired_to_production_feature_engineering": False,
    }
