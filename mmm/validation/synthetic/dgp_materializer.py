"""Rich DGP materialization — deterministic KPI from world_truth (Phase 4B-1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation
from mmm.validation.synthetic._io import sha256_file, write_json
from mmm.validation.synthetic.materializer import (
    CHECKSUM_VERSION,
    DEFAULT_GEO_COLUMN,
    DEFAULT_WEEK_COLUMN,
    build_decision_truth_index,
    build_metadata,
    load_world_truth,
)
from mmm.validation.synthetic.replay_units import build_replay_units_payload

DGP_MATERIALIZATION_VERSION = "dgp_materialize_v1.0.0"
DGP_DIAGNOSTICS_PARQUET = "dgp_diagnostics.parquet"
DGP_DIAGNOSTICS_MANIFEST = "dgp_diagnostics_manifest.json"

# Ridge semi_log on modeling scale: log(y) = intercept + sum(beta_j * hill(adstock(spend_j))) + eps
DGP_FORMULA_DOC = {
    "model_form": "semi_log",
    "adstock": "geometric_carry[t] = spend[t] + decay * carry[t-1]",
    "saturation": "hill(x) = x^slope / (half_max^slope + x^slope + 1e-12)",
    "linear_predictor": "eta = intercept + sum_c beta_c * hill_c",
    "kpi": "y = exp(eta + eps); eps ~ N(0, observation_noise_std) deterministic at 0 for exact worlds",
}


@dataclass(frozen=True)
class DgpMaterializeResult:
    bundle_dir: Path
    world_id: str
    files_written: tuple[str, ...]
    checksums: dict[str, Any]


def observation_noise_std(outcome_truth: dict[str, Any]) -> float:
    if "observation_noise_std" in outcome_truth:
        return float(outcome_truth["observation_noise_std"])
    level = str(outcome_truth.get("observation_noise_level", "low"))
    return {"low": 0.0, "medium": 0.02, "high": 0.05}.get(level, 0.0)


def geometric_adstock_series(spend: np.ndarray, decay: float) -> np.ndarray:
    """Canonical geometric adstock (matches ``GeometricAdstock``)."""
    return GeometricAdstock(decay).transform(np.asarray(spend, dtype=float))


def hill_saturation_series(adstocked: np.ndarray, *, half_max: float, slope: float) -> np.ndarray:
    """Canonical Hill saturation (matches ``HillSaturation``)."""
    return HillSaturation(half_max=half_max, slope=slope).transform(np.asarray(adstocked, dtype=float))


def _week_dates(truth: dict[str, Any]) -> list[str]:
    from datetime import datetime, timedelta

    time_t = truth["time_truth"]
    start = datetime.strptime(str(time_t["start_date"]), "%Y-%m-%d").date()
    n = int(time_t["n_periods"])
    if str(time_t.get("date_frequency", "weekly")) != "weekly":
        raise ValueError("dgp_materialize_v1 supports weekly date_frequency only")
    return [(start + timedelta(weeks=i)).isoformat() for i in range(n)]


def _spend_by_geo_week(
    truth: dict[str, Any],
    *,
    geos: list[str],
    weeks: list[str],
    channels: list[str],
) -> dict[tuple[str, str], dict[str, float]]:
    media = truth["media_truth"]
    base = dict(media.get("baseline_spend_by_channel") or {})
    spec = media.get("spend_process_spec") or {}
    if not base:
        level = float(spec.get("level", 10.0))
        base = {c: level for c in channels}
    kind = str(spec.get("kind", "constant"))
    if kind == "pre_impulse_constant":
        impulse_weeks = int(spec.get("impulse_periods", 4))
        impulse_level = float(spec.get("impulse_level", 20.0))
        baseline_level = float(spec.get("baseline_level", 10.0))
        out_imp: dict[tuple[str, str], dict[str, float]] = {}
        for wi, w in enumerate(weeks):
            level = impulse_level if wi < impulse_weeks else baseline_level
            for g in geos:
                out_imp[(g, w)] = {c: level for c in channels}
        return _apply_experiment_spend_overlays(truth, out_imp, weeks)
    if kind == "collinear_block":
        primary = str(spec.get("primary_channel", channels[0]))
        secondary = str(spec.get("secondary_channel", channels[1] if len(channels) > 1 else channels[0]))
        scale = float(spec.get("scale", 0.98))
        level = float(spec.get("level", base.get(primary, 10.0)))
        out_col: dict[tuple[str, str], dict[str, float]] = {}
        n_weeks = max(len(weeks), 1)
        for wi, w in enumerate(weeks):
            jitter = 1.0 + 0.02 * np.sin(2.0 * np.pi * wi / n_weeks)
            for g in geos:
                row = {c: float(base.get(c, level)) for c in channels}
                p_spend = level * jitter
                row[primary] = p_spend
                if secondary in row:
                    row[secondary] = p_spend * scale
                out_col[(g, w)] = row
        return _apply_experiment_spend_overlays(truth, out_col, weeks)
    if kind == "channel_modulated":
        mod = spec.get("by_channel") or {}
        out: dict[tuple[str, str], dict[str, float]] = {}
        n_weeks = max(len(weeks), 1)
        for wi, w in enumerate(weeks):
            phase = 2.0 * np.pi * wi / n_weeks
            for g in geos:
                row: dict[str, float] = {}
                for c in channels:
                    cfg = mod.get(c) or {}
                    b = float(cfg.get("base", base.get(c, 10.0)))
                    amp = float(cfg.get("amplitude", 0.0))
                    row[c] = max(0.0, b + amp * np.sin(phase))
                out[(g, w)] = row
        return _apply_experiment_spend_overlays(truth, out, weeks)
    out = {}
    for g in geos:
        for w in weeks:
            out[(g, w)] = {c: float(base.get(c, 0.0)) for c in channels}
    return _apply_experiment_spend_overlays(truth, out, weeks)


def _effective_betas_at_week_index(
    truth: dict[str, Any],
    channels: list[str],
    week_index: int,
) -> dict[str, float]:
    """Apply ``drift_truth.coefficient_drift`` schedule on top of baseline ``true_beta_by_channel``."""
    coef = truth["coefficient_truth"]
    betas = {c: float(coef["true_beta_by_channel"][c]) for c in channels}
    for drift in (truth.get("drift_truth") or {}).get("coefficient_drift") or []:
        ch = str(drift.get("channel", ""))
        if ch not in betas:
            continue
        start = int(drift.get("start_period_index", 0))
        if week_index < start:
            if "pre_beta" in drift:
                betas[ch] = float(drift["pre_beta"])
            continue
        if "post_beta" in drift:
            betas[ch] = float(drift["post_beta"])
        elif "delta_beta" in drift:
            pre = float(drift.get("pre_beta", betas[ch]))
            betas[ch] = pre + float(drift["delta_beta"])
    return betas


def _apply_experiment_spend_overlays(
    truth: dict[str, Any],
    spend_map: dict[tuple[str, str], dict[str, float]],
    weeks: list[str],
) -> dict[tuple[str, str], dict[str, float]]:
    """Apply observed spend shocks inside experiment windows on treated geos."""
    units = (truth.get("experiment_truth") or {}).get("units") or []
    if not units:
        return spend_map
    for unit in units:
        channel = str(unit["channel"])
        treated = set(str(g) for g in unit.get("geos") or [])
        ws = str(unit["week_start"])
        we = str(unit["week_end"])
        shock = unit.get("spend_shock") or {}
        obs_mult = float(shock.get("observed_multiplier", 1.0))
        if obs_mult == 1.0:
            continue
        for (geo, week), row in spend_map.items():
            if geo not in treated or not (ws <= week <= we):
                continue
            if channel in row:
                row[channel] = float(row[channel]) * obs_mult
    return spend_map


def compute_dgp_series(
    truth: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build panel KPI and long-form diagnostics from ``world_truth``.

    Returns ``(panel_df, diagnostics_df)``. Does not read or write bundle files.
    """
    if str(truth.get("outcome_truth", {}).get("model_form", "")) != "semi_log":
        raise ValueError("dgp_materialize_v1 requires outcome_truth.model_form == semi_log")
    transform = truth["transform_truth"]
    if str(transform.get("adstock_family", "")) != "geometric":
        raise ValueError("dgp_materialize_v1 requires geometric adstock")
    if str(transform.get("saturation_family", "")) != "hill":
        raise ValueError("dgp_materialize_v1 requires Hill saturation")

    geo = truth["geo_truth"]
    media = truth["media_truth"]
    outcome = truth["outcome_truth"]
    coef = truth["coefficient_truth"]
    geos = list(geo["geos"])
    channels = list(media["channels"])
    weeks = _week_dates(truth)
    geo_col = str(geo.get("geo_column_name") or DEFAULT_GEO_COLUMN)
    week_col = str(truth["time_truth"].get("week_column_name") or DEFAULT_WEEK_COLUMN)
    target_col = str(outcome["target_column"])
    intercept = float(coef["intercept"])
    noise_std = observation_noise_std(outcome)

    spend_grid = _spend_by_geo_week(truth, geos=geos, weeks=weeks, channels=channels)
    diag_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []

    for g in geos:
        week_list = list(weeks)
        # Per-channel time series for transforms (causal within geo)
        saturated_by_ch: dict[str, np.ndarray] = {}
        adstocked_by_ch: dict[str, np.ndarray] = {}
        raw_by_ch: dict[str, np.ndarray] = {}
        for ch in channels:
            raw = np.array([spend_grid[(g, w)][ch] for w in week_list], dtype=float)
            decay = float(transform["adstock_decay_by_channel"][ch])
            half = float(transform["hill_half_max_by_channel"][ch])
            slope = float(transform["hill_slope_by_channel"][ch])
            ad = GeometricAdstock(decay)
            sat = HillSaturation(half_max=half, slope=slope)
            adstocked = apply_adstock_saturation_series(raw, ad, sat)
            # apply_adstock_saturation_series returns saturated; recover adstocked
            ad_only = geometric_adstock_series(raw, decay)
            sat_only = hill_saturation_series(ad_only, half_max=half, slope=slope)
            raw_by_ch[ch] = raw
            adstocked_by_ch[ch] = ad_only
            saturated_by_ch[ch] = sat_only
            np.testing.assert_allclose(adstocked, sat_only, rtol=1e-9, atol=1e-9)

        for t, w in enumerate(week_list):
            betas_t = _effective_betas_at_week_index(truth, channels, t)
            eta = intercept + sum(
                betas_t[ch] * float(saturated_by_ch[ch][t]) for ch in channels
            )
            eps = 0.0 if noise_std == 0.0 else 0.0  # deterministic exact worlds: no stochastic draw
            log_y = eta + eps
            y = float(np.exp(log_y))
            panel_row: dict[str, Any] = {
                geo_col: g,
                week_col: pd.Timestamp(w),
                target_col: y,
            }
            for ch in channels:
                panel_row[ch] = float(raw_by_ch[ch][t])
            panel_rows.append(panel_row)

            for ch in channels:
                diag_rows.append(
                    {
                        geo_col: g,
                        week_col: pd.Timestamp(w),
                        "channel": ch,
                        "effective_beta": float(betas_t[ch]),
                        "week_index": int(t),
                        "raw_spend": float(raw_by_ch[ch][t]),
                        "adstocked_spend": float(adstocked_by_ch[ch][t]),
                        "saturated_feature": float(saturated_by_ch[ch][t]),
                        "linear_predictor": float(eta),
                        "log_kpi": float(log_y),
                        "generated_kpi": float(y),
                        "noise_epsilon": float(eps),
                        "derived_artifact": True,
                    }
                )

    panel_df = pd.DataFrame(panel_rows).sort_values([geo_col, week_col], kind="mergesort").reset_index(drop=True)
    diag_df = pd.DataFrame(diag_rows).sort_values([geo_col, week_col, "channel"], kind="mergesort").reset_index(
        drop=True
    )
    return panel_df, diag_df


def build_dgp_diagnostics_manifest(truth: dict[str, Any]) -> dict[str, Any]:
    meta = truth["metadata"]
    return {
        "artifact_kind": "derived_dgp_diagnostics_manifest",
        "authoritative": False,
        "not_world_truth": True,
        "world_id": str(meta["world_id"]),
        "dgp_materialization_version": DGP_MATERIALIZATION_VERSION,
        "formulas": DGP_FORMULA_DOC,
        "columns_in_parquet": [
            "geo_id",
            "week_start_date",
            "channel",
            "raw_spend",
            "adstocked_spend",
            "saturated_feature",
            "linear_predictor",
            "log_kpi",
            "generated_kpi",
            "noise_epsilon",
            "derived_artifact",
        ],
    }


def compute_dgp_checksums(bundle_dir: Path, *, truth_path: Path) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    checksums: dict[str, Any] = {
        "checksum_version": CHECKSUM_VERSION,
        "world_truth_sha256": sha256_file(truth_path),
        "dgp_materialization_version": DGP_MATERIALIZATION_VERSION,
    }
    for name, key in (
        ("panel.parquet", "panel_sha256"),
        (DGP_DIAGNOSTICS_PARQUET, "dgp_diagnostics_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("decision_truth.json", "decision_truth_sha256"),
    ):
        p = bundle / name
        if p.is_file():
            checksums[key] = sha256_file(p)
    replay_path = bundle / "replay_units.json"
    if replay_path.is_file():
        replay_digest = sha256_file(replay_path)
        checksums["replay_sha256"] = replay_digest
        checksums["experiment_sha256"] = replay_digest
    else:
        checksums["replay_sha256"] = None
        checksums["experiment_sha256"] = None

    manifest_entries = []
    for p in sorted(bundle.iterdir()):
        if p.name == "checksums.json" or not p.is_file():
            continue
        manifest_entries.append({"path": p.name, "sha256": sha256_file(p)})
    manifest_body = json.dumps(manifest_entries, sort_keys=True, separators=(",", ":"))
    checksums["manifest_hash"] = hashlib.sha256(manifest_body.encode("utf-8")).hexdigest()
    return checksums


def materialize_dgp_world(
    bundle_dir: str | Path,
    *,
    truth_path: str | Path | None = None,
    overwrite: bool = True,
) -> DgpMaterializeResult:
    """
    Materialize a rich DGP panel and diagnostics from ``world_truth.json``.

    Never modifies ``world_truth.json``. Diagnostics are derived artifacts only.
    """
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    truth_file = Path(truth_path) if truth_path is not None else bundle / "world_truth.json"
    if not truth_file.is_file():
        raise FileNotFoundError(f"world_truth.json not found: {truth_file}")

    truth = load_world_truth(truth_file)
    world_id = str(truth["metadata"]["world_id"])
    written: list[str] = []

    panel_df, diag_df = compute_dgp_series(truth)

    panel_path = bundle / "panel.parquet"
    if panel_path.exists() and not overwrite:
        raise FileExistsError(f"{panel_path} exists; pass overwrite=True")
    panel_df.to_parquet(panel_path, index=False)
    written.append("panel.parquet")

    diag_path = bundle / DGP_DIAGNOSTICS_PARQUET
    diag_df.to_parquet(diag_path, index=False)
    written.append(DGP_DIAGNOSTICS_PARQUET)

    manifest_path = bundle / DGP_DIAGNOSTICS_MANIFEST
    write_json(manifest_path, build_dgp_diagnostics_manifest(truth))
    written.append(DGP_DIAGNOSTICS_MANIFEST)

    replay_payload = build_replay_units_payload(truth)
    if replay_payload is not None:
        replay_path = bundle / "replay_units.json"
        replay_path.write_text(
            json.dumps(replay_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append("replay_units.json")

    decision_index = build_decision_truth_index(truth)
    if decision_index is not None:
        decision_path = bundle / "decision_truth.json"
        write_json(decision_path, decision_index)
        written.append("decision_truth.json")

    metadata = build_metadata(truth, bundle_dir=bundle, materialized_files=written)
    metadata["materialization_version"] = DGP_MATERIALIZATION_VERSION
    metadata["dgp_materialization"] = True
    metadata["derived_artifacts"] = [DGP_DIAGNOSTICS_PARQUET, DGP_DIAGNOSTICS_MANIFEST]
    meta_path = bundle / "metadata.json"
    write_json(meta_path, metadata)
    written.append("metadata.json")

    checksums = compute_dgp_checksums(bundle, truth_path=truth_file)
    write_json(bundle / "checksums.json", checksums)
    written.append("checksums.json")

    return DgpMaterializeResult(
        bundle_dir=bundle,
        world_id=world_id,
        files_written=tuple(written),
        checksums=checksums,
    )
