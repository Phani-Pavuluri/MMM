"""Real-panel collinearity-aware preprocessing for H5 shadow runs (research only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema

CHANNEL_POLICY_KEEP_ALL = "keep_all_channels"
CHANNEL_POLICY_SINGLE = "single_channel"
CHANNEL_POLICY_DROP_COLLINEAR = "drop_collinear_channels"
CHANNEL_POLICY_DROP_SPARSE = "drop_sparse_channels"
CHANNEL_POLICY_COMPOSITE = "composite_media_channel"
CHANNEL_POLICY_POOLED = "pooled_channel_effects"

CHANNEL_POLICY_MODES: frozenset[str] = frozenset(
    {
        CHANNEL_POLICY_KEEP_ALL,
        CHANNEL_POLICY_SINGLE,
        CHANNEL_POLICY_DROP_COLLINEAR,
        CHANNEL_POLICY_DROP_SPARSE,
        CHANNEL_POLICY_COMPOSITE,
        CHANNEL_POLICY_POOLED,
    }
)

COMPOSITE_METHODS: frozenset[str] = frozenset(
    {
        "sum_scaled_media",
        "first_principal_component",
    }
)


class H5RealPanelPreprocessingError(ValueError):
    """Real-panel preprocessing failed — fail closed."""


def compute_media_correlation_matrix(df: pd.DataFrame, channel_columns: tuple[str, ...] | list[str]) -> dict[str, Any]:
    channels = list(channel_columns)
    if len(channels) < 2:
        return {"channels": channels, "matrix": {}, "max_abs_off_diagonal": 0.0}
    media = df[channels].to_numpy(dtype=float)
    corr = np.corrcoef(media.T)
    matrix: dict[str, dict[str, float]] = {}
    for i, a in enumerate(channels):
        matrix[a] = {b: float(corr[i, j]) for j, b in enumerate(channels)}
    off = corr[np.triu_indices_from(corr, k=1)]
    return {
        "channels": channels,
        "matrix": matrix,
        "max_abs_off_diagonal": float(np.max(np.abs(off))) if len(off) else 0.0,
    }


def detect_collinear_channel_groups(
    df: pd.DataFrame,
    channel_columns: tuple[str, ...] | list[str],
    *,
    max_abs_corr_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Groups of channels with pairwise |ρ| >= threshold (union-find style merge)."""
    channels = list(channel_columns)
    if len(channels) < 2:
        return []

    media = df[channels].to_numpy(dtype=float)
    corr = np.corrcoef(media.T)
    n = len(channels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(corr[i, j])) >= max_abs_corr_threshold:
                union(i, j)

    groups_map: dict[int, list[str]] = {}
    for idx, ch in enumerate(channels):
        root = find(idx)
        groups_map.setdefault(root, []).append(ch)

    groups: list[dict[str, Any]] = []
    for members in groups_map.values():
        if len(members) < 2:
            continue
        pairs: list[dict[str, Any]] = []
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                ia, ib = channels.index(a), channels.index(b)
                pairs.append({"a": a, "b": b, "correlation": float(corr[ia, ib])})
        groups.append(
            {
                "channels": sorted(members),
                "max_abs_corr_threshold": max_abs_corr_threshold,
                "pairwise": pairs,
            }
        )
    return groups


def _mean_abs_corr_to_others(ch: str, channels: list[str], corr: np.ndarray) -> float:
    idx = channels.index(ch)
    others = [channels.index(o) for o in channels if o != ch]
    if not others:
        return 0.0
    return float(np.mean([abs(corr[idx, j]) for j in others]))


def _channels_to_drop_from_groups(
    df: pd.DataFrame,
    channels: list[str],
    groups: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Drop the most redundant channel per collinear group (highest mean |ρ| to others)."""
    media = df[channels].to_numpy(dtype=float)
    corr = np.corrcoef(media.T)
    dropped: list[dict[str, str]] = []
    for group in groups:
        members = list(group["channels"])
        if len(members) < 2:
            continue
        scores = {ch: _mean_abs_corr_to_others(ch, members, corr) for ch in members}
        drop_ch = max(scores, key=scores.get)
        for ch in members:
            if ch == drop_ch:
                dropped.append(
                    {
                        "channel": ch,
                        "reason": (
                            f"collinear_group_drop_redundant: highest mean |rho| to group "
                            f"({scores[ch]:.4f})"
                        ),
                    }
                )
    return dropped


def validate_collinearity_config(channel_policy: dict[str, Any] | None) -> None:
    if channel_policy is None:
        return
    if not isinstance(channel_policy, dict):
        raise H5RealPanelPreprocessingError("channel_policy must be an object")
    mode = channel_policy.get("mode")
    if mode not in CHANNEL_POLICY_MODES:
        raise H5RealPanelPreprocessingError(
            f"channel_policy.mode must be one of {sorted(CHANNEL_POLICY_MODES)}"
        )
    if mode == CHANNEL_POLICY_SINGLE:
        if not str(channel_policy.get("channel") or "").strip():
            raise H5RealPanelPreprocessingError("single_channel requires channel_policy.channel")
    elif mode == CHANNEL_POLICY_DROP_COLLINEAR:
        dropped = channel_policy.get("dropped_channels") or channel_policy.get(
            "explicit_dropped_channels"
        )
        kept = channel_policy.get("kept_channels") or channel_policy.get("explicit_kept_channels")
        if dropped:
            if not isinstance(dropped, list) or not dropped:
                raise H5RealPanelPreprocessingError(
                    "drop_collinear_channels requires non-empty dropped_channels when explicit"
                )
            if not isinstance(kept, list) or not kept:
                raise H5RealPanelPreprocessingError(
                    "drop_collinear_channels requires non-empty kept_channels when explicit"
                )
        thr = channel_policy.get("max_abs_corr_threshold")
        if thr is None or not (0.0 < float(thr) < 1.0):
            raise H5RealPanelPreprocessingError(
                "drop_collinear_channels requires 0 < max_abs_corr_threshold < 1"
            )
    elif mode == CHANNEL_POLICY_DROP_SPARSE:
        dropped = channel_policy.get("dropped_channels")
        kept = channel_policy.get("kept_channels")
        if not isinstance(dropped, list) or not dropped:
            raise H5RealPanelPreprocessingError(
                "drop_sparse_channels requires non-empty dropped_channels"
            )
        if not isinstance(kept, list) or not kept:
            raise H5RealPanelPreprocessingError(
                "drop_sparse_channels requires non-empty kept_channels"
            )
        reason = str(channel_policy.get("reason") or channel_policy.get("sparse_drop_reason") or "").strip()
        if not reason:
            raise H5RealPanelPreprocessingError(
                "drop_sparse_channels requires non-empty reason documenting sparse-channel drop"
            )
        if channel_policy.get("no_silent_dropping") is not True:
            raise H5RealPanelPreprocessingError(
                "drop_sparse_channels requires no_silent_dropping=true"
            )
    elif mode == CHANNEL_POLICY_COMPOSITE:
        sources = channel_policy.get("source_channels")
        if not sources or not isinstance(sources, list):
            raise H5RealPanelPreprocessingError("composite_media_channel requires source_channels list")
        method = channel_policy.get("method")
        if method not in COMPOSITE_METHODS:
            raise H5RealPanelPreprocessingError(
                f"composite method must be one of {sorted(COMPOSITE_METHODS)}"
            )
        if not str(channel_policy.get("output_channel") or "").strip():
            raise H5RealPanelPreprocessingError("composite_media_channel requires output_channel")
    elif mode == CHANNEL_POLICY_POOLED:  # noqa: SIM102 - preserve reserved branch structure
        if channel_policy.get("implemented") is True:
            pass  # reserved for future pooled ablation wiring


def build_single_channel_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    channel: str,
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any]]:
    if channel not in schema.channel_columns:
        raise H5RealPanelPreprocessingError(f"single_channel {channel!r} not in panel channels")
    cols = [schema.geo_column, schema.week_column, schema.target_column, channel, *schema.control_columns]
    out_df = df[cols].copy()
    out_schema = PanelSchema(
        schema.geo_column,
        schema.week_column,
        schema.target_column,
        (channel,),
        schema.control_columns,
    )
    record = {
        "mode": CHANNEL_POLICY_SINGLE,
        "kept_channels": [channel],
        "dropped_channels": [c for c in schema.channel_columns if c != channel],
        "ablation_only": True,
    }
    return out_df, out_schema, record


def build_drop_collinear_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    max_abs_corr_threshold: float = 0.95,
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any]]:
    channels = list(schema.channel_columns)
    groups = detect_collinear_channel_groups(df, channels, max_abs_corr_threshold=max_abs_corr_threshold)
    to_drop_info = _channels_to_drop_from_groups(df, channels, groups)
    drop_set = {d["channel"] for d in to_drop_info}
    kept = [c for c in channels if c not in drop_set]
    if not kept:
        raise H5RealPanelPreprocessingError(
            "drop_collinear_channels would remove all media channels — fail closed"
        )
    cols = [schema.geo_column, schema.week_column, schema.target_column, *kept, *schema.control_columns]
    out_df = df[cols].copy()
    out_schema = PanelSchema(
        schema.geo_column,
        schema.week_column,
        schema.target_column,
        tuple(kept),
        schema.control_columns,
    )
    record = {
        "mode": CHANNEL_POLICY_DROP_COLLINEAR,
        "max_abs_corr_threshold": max_abs_corr_threshold,
        "collinear_groups": groups,
        "dropped_channels": to_drop_info,
        "kept_channels": kept,
        "ablation_only": True,
    }
    return out_df, out_schema, record


def build_explicit_channel_drop_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    dropped_channels: list[str],
    kept_channels: list[str] | None = None,
    reason: str = "explicit_frozen_policy_drop",
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any]]:
    """Drop explicitly listed channels — no heuristic silent dropping."""
    drop_set = {str(ch) for ch in dropped_channels}
    unknown = drop_set - set(schema.channel_columns)
    if unknown:
        raise H5RealPanelPreprocessingError(
            f"explicit dropped_channels not in panel: {sorted(unknown)!r}"
        )
    kept = [ch for ch in schema.channel_columns if ch not in drop_set]
    if kept_channels is not None:
        expected = list(kept_channels)
        if set(kept) != set(expected):
            raise H5RealPanelPreprocessingError(
                f"explicit kept_channels {expected!r} does not match drop list (got {kept!r})"
            )
        kept = expected
    if not kept:
        raise H5RealPanelPreprocessingError(
            "explicit channel drop would remove all media channels — fail closed"
        )
    cols = [schema.geo_column, schema.week_column, schema.target_column, *kept, *schema.control_columns]
    out_df = df[cols].copy()
    out_schema = PanelSchema(
        schema.geo_column,
        schema.week_column,
        schema.target_column,
        tuple(kept),
        schema.control_columns,
    )
    record = {
        "mode": CHANNEL_POLICY_DROP_COLLINEAR,
        "dropped_channels": [{"channel": ch, "reason": reason} for ch in sorted(drop_set)],
        "kept_channels": kept,
        "explicit_drop": True,
        "ablation_only": False,
    }
    return out_df, out_schema, record


def build_sparse_channel_drop_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    dropped_channels: list[str],
    kept_channels: list[str],
    reason: str,
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any]]:
    """Governed drop of near-zero / weakly identified sparse media channels."""
    if not str(reason).strip():
        raise H5RealPanelPreprocessingError(
            "drop_sparse_channels requires documented sparse_drop reason"
        )
    out_df, out_schema, record = build_explicit_channel_drop_panel(
        df,
        schema,
        dropped_channels=dropped_channels,
        kept_channels=kept_channels,
        reason=reason,
    )
    record["mode"] = CHANNEL_POLICY_DROP_SPARSE
    record["sparse_drop_reason"] = reason
    record["explicit_drop"] = True
    record["ablation_only"] = False
    return out_df, out_schema, record


def _composite_column(
    df: pd.DataFrame,
    source_channels: list[str],
    *,
    method: str,
) -> np.ndarray:
    mat = df[source_channels].to_numpy(dtype=float)
    scaled = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-6)
    if method == "sum_scaled_media":
        return scaled.sum(axis=1)
    if method == "first_principal_component":
        u, s, _vt = np.linalg.svd(scaled, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        return (pc1 - pc1.mean()) / (pc1.std() + 1e-6)
    raise H5RealPanelPreprocessingError(f"unsupported composite method: {method!r}")


def build_composite_media_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    source_channels: list[str],
    method: str,
    output_channel: str,
    remaining_channels: list[str] | None = None,
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any]]:
    for ch in source_channels:
        if ch not in schema.channel_columns:
            raise H5RealPanelPreprocessingError(f"composite source {ch!r} not in panel")
    if method not in COMPOSITE_METHODS:
        raise H5RealPanelPreprocessingError(f"unsupported composite method: {method!r}")

    remaining = list(remaining_channels or [])
    for ch in remaining:
        if ch in source_channels:
            raise H5RealPanelPreprocessingError("remaining_channels must not overlap source_channels")
        if ch not in schema.channel_columns:
            raise H5RealPanelPreprocessingError(f"remaining channel {ch!r} not in panel")

    out_df = df[
        [schema.geo_column, schema.week_column, schema.target_column, *schema.control_columns]
    ].copy()
    for ch in remaining:
        out_df[ch] = df[ch].values
    composite = _composite_column(df, source_channels, method=method)
    composite = composite - float(np.min(composite)) + 1.0
    out_df[output_channel] = composite
    final_channels = [*remaining, output_channel]

    out_schema = PanelSchema(
        schema.geo_column,
        schema.week_column,
        schema.target_column,
        tuple(final_channels),
        schema.control_columns,
    )
    record = {
        "mode": CHANNEL_POLICY_COMPOSITE,
        "method": method,
        "source_channels": list(source_channels),
        "output_channel": output_channel,
        "remaining_channels": remaining,
        "kept_channels": final_channels,
        "dropped_channels": [
            {"channel": ch, "reason": "merged_into_composite"}
            for ch in schema.channel_columns
            if ch not in final_channels
        ],
        "ablation_only": True,
        "diagnostic_only": True,
    }
    return out_df, out_schema, record


def apply_channel_policy(
    df: pd.DataFrame,
    schema: PanelSchema,
    transform_config: dict[str, Any],
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any], dict[str, Any]]:
    """
    Apply explicit channel_policy from transform_config.

    Returns (panel, schema, updated_transform_config, policy_record).
    """
    policy = transform_config.get("channel_policy")
    if not policy:
        return df, schema, transform_config, {
            "mode": CHANNEL_POLICY_KEEP_ALL,
            "kept_channels": list(schema.channel_columns),
        }

    validate_collinearity_config(policy)
    mode = policy["mode"]
    base_transforms = dict(transform_config.get("media_transforms_by_channel") or {})

    if mode == CHANNEL_POLICY_KEEP_ALL:
        record = {"mode": mode, "kept_channels": list(schema.channel_columns), "ablation_only": False}
        return df, schema, transform_config, record

    if mode == CHANNEL_POLICY_SINGLE:
        ch = str(policy["channel"])
        out_df, out_schema, record = build_single_channel_panel(df, schema, ch)
    elif mode == CHANNEL_POLICY_DROP_SPARSE:
        out_df, out_schema, record = build_sparse_channel_drop_panel(
            df,
            schema,
            dropped_channels=list(policy["dropped_channels"]),
            kept_channels=list(policy["kept_channels"]),
            reason=str(policy.get("reason") or policy.get("sparse_drop_reason") or ""),
        )
        record["no_silent_dropping"] = policy.get("no_silent_dropping", True)
    elif mode == CHANNEL_POLICY_DROP_COLLINEAR:
        explicit_drop = policy.get("dropped_channels")
        if explicit_drop:
            out_df, out_schema, record = build_explicit_channel_drop_panel(
                df,
                schema,
                dropped_channels=list(explicit_drop),
                kept_channels=list(policy["kept_channels"]) if policy.get("kept_channels") else None,
                reason=str(policy.get("reason") or "explicit_frozen_policy_drop"),
            )
            record["max_abs_corr_threshold"] = policy.get("max_abs_corr_threshold")
            record["no_silent_dropping"] = policy.get("no_silent_dropping", True)
        else:
            out_df, out_schema, record = build_drop_collinear_panel(
                df,
                schema,
                max_abs_corr_threshold=float(policy["max_abs_corr_threshold"]),
            )
    elif mode == CHANNEL_POLICY_COMPOSITE:
        out_df, out_schema, record = build_composite_media_panel(
            df,
            schema,
            source_channels=list(policy["source_channels"]),
            method=str(policy["method"]),
            output_channel=str(policy["output_channel"]),
            remaining_channels=list(policy.get("remaining_channels") or []),
        )
    elif mode == CHANNEL_POLICY_POOLED:
        raise H5RealPanelPreprocessingError(
            "pooled_channel_effects ablation is not implemented in preprocessing — use geometry runner note"
        )
    else:
        raise H5RealPanelPreprocessingError(f"unsupported channel_policy.mode: {mode!r}")

    new_transforms = {ch: base_transforms.get(ch, "identity") for ch in out_schema.channel_columns}

    updated_config = {**transform_config, "media_transforms_by_channel": new_transforms}
    record["explicit_policy"] = True
    record["no_silent_dropping"] = True
    return out_df, out_schema, updated_config, record


def apply_preprocessing_config(
    df: pd.DataFrame,
    schema: PanelSchema,
    preprocessing: dict[str, Any] | None,
) -> pd.DataFrame:
    """Optional prescale steps on media columns (research ablation only)."""
    if not preprocessing:
        return df
    work = df.copy()
    if preprocessing.get("media_prescale") == "zscore_panel":
        for ch in schema.channel_columns:
            vals = work[ch].to_numpy(dtype=float)
            work[ch] = (vals - vals.mean()) / (vals.std() + 1e-6)
    return work
