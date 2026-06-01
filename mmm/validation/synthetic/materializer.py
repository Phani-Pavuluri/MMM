"""Materialize derived bundle artifacts from authoritative world_truth.json (no DGP)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.validation.synthetic._io import read_json, sha256_file, write_json
from mmm.validation.synthetic.replay_units import build_replay_units_payload

MATERIALIZATION_VERSION = "materialize_v1.0.0"
WORLD_CONTRACT_VERSION = "groundtruth_world_v1"
CHECKSUM_VERSION = "checksums_v1"
DEFAULT_WEEK_COLUMN = "week_start_date"
DEFAULT_GEO_COLUMN = "geo_id"


@dataclass(frozen=True)
class MaterializeResult:
    bundle_dir: Path
    world_id: str
    files_written: tuple[str, ...]
    checksums: dict[str, Any]


def load_world_truth(path: Path) -> dict[str, Any]:
    return read_json(path)


def _week_dates(truth: dict[str, Any]) -> list[str]:
    time_t = truth["time_truth"]
    start = datetime.strptime(str(time_t["start_date"]), "%Y-%m-%d").date()
    n = int(time_t["n_periods"])
    freq = str(time_t.get("date_frequency", "weekly"))
    if freq != "weekly":
        raise ValueError(f"unsupported date_frequency {freq!r}; only weekly supported in materialize_v1")
    return [(start + timedelta(weeks=i)).isoformat() for i in range(n)]


def build_panel_dataframe(truth: dict[str, Any]) -> pd.DataFrame:
    """
    Render a minimal panel from authored truth (constant spend and KPI level).

    Does not estimate coefficients or Δμ from the panel — see INV-005.
    """
    geo = truth["geo_truth"]
    media = truth["media_truth"]
    outcome = truth["outcome_truth"]
    geos = list(geo["geos"])
    channels = list(media["channels"])
    weeks = _week_dates(truth)
    geo_col = str(geo.get("geo_column_name") or DEFAULT_GEO_COLUMN)
    week_col = str(truth["time_truth"].get("week_column_name") or DEFAULT_WEEK_COLUMN)
    target_col = str(outcome["target_column"])
    base_spend = dict(media.get("baseline_spend_by_channel") or {})
    if not base_spend:
        spec = media.get("spend_process_spec") or {}
        level = float(spec.get("level", 10.0))
        base_spend = {c: level for c in channels}
    kpi_level = float(outcome.get("base_level_mean", 100.0))

    rows: list[dict[str, Any]] = []
    for g in geos:
        for w in weeks:
            row: dict[str, Any] = {geo_col: g, week_col: w, target_col: kpi_level}
            for ch in channels:
                row[ch] = float(base_spend.get(ch, 0.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    sort_cols = [geo_col, week_col]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_decision_truth_index(truth: dict[str, Any]) -> dict[str, Any] | None:
    decision = truth.get("decision_truth") or {}
    scenarios = decision.get("scenarios") or []
    if not scenarios:
        return None
    meta = truth["metadata"]
    index = []
    for i, sc in enumerate(scenarios):
        index.append(
            {
                "scenario_id": str(sc["scenario_id"]),
                "ref": f"decision_truth.scenarios[{i}]",
            }
        )
    return {
        "world_id": str(meta["world_id"]),
        "world_version": str(meta["world_version"]),
        "world_contract_version": str(meta["world_contract_version"]),
        "world_generator_version": str(meta["world_generator_version"]),
        "materialization_version": MATERIALIZATION_VERSION,
        "scenario_index": index,
    }


def build_metadata(
    truth: dict[str, Any],
    *,
    bundle_dir: Path,
    materialized_files: list[str],
) -> dict[str, Any]:
    meta = truth["metadata"]
    return {
        "world_id": str(meta["world_id"]),
        "world_version": str(meta["world_version"]),
        "world_contract_version": str(meta["world_contract_version"]),
        "world_generator_version": str(meta["world_generator_version"]),
        "materialization_version": MATERIALIZATION_VERSION,
        "seed": int(meta["generation_seed"]),
        "creation_timestamp": str(meta["creation_timestamp"]),
        "scenario_tags": list(meta.get("scenario_tags") or []),
        "checksum_version": CHECKSUM_VERSION,
        "bundle_path": str(bundle_dir.as_posix()),
        "materialized_files": sorted(materialized_files),
        "catalog_ref": str(meta["world_id"]),
    }


def compute_checksums(bundle_dir: Path, *, truth_path: Path) -> dict[str, Any]:
    panel_path = bundle_dir / "panel.parquet"
    replay_path = bundle_dir / "replay_units.json"
    decision_path = bundle_dir / "decision_truth.json"
    meta_path = bundle_dir / "metadata.json"

    checksums: dict[str, Any] = {
        "checksum_version": CHECKSUM_VERSION,
        "world_truth_sha256": sha256_file(truth_path),
    }
    if panel_path.is_file():
        checksums["panel_sha256"] = sha256_file(panel_path)
    if replay_path.is_file():
        replay_digest = sha256_file(replay_path)
        checksums["replay_sha256"] = replay_digest
        checksums["experiment_sha256"] = replay_digest
    else:
        checksums["replay_sha256"] = None
        checksums["experiment_sha256"] = None
    if decision_path.is_file():
        checksums["decision_truth_sha256"] = sha256_file(decision_path)
    if meta_path.is_file():
        checksums["metadata_sha256"] = sha256_file(meta_path)

    manifest_entries = []
    for p in sorted(bundle_dir.iterdir()):
        if p.name == "checksums.json" or not p.is_file():
            continue
        manifest_entries.append({"path": p.name, "sha256": sha256_file(p)})
    manifest_body = json.dumps(manifest_entries, sort_keys=True, separators=(",", ":"))
    checksums["manifest_hash"] = hashlib.sha256(manifest_body.encode("utf-8")).hexdigest()
    return checksums


def materialize_world(
    bundle_dir: str | Path,
    *,
    truth_path: str | Path | None = None,
    overwrite: bool = True,
) -> MaterializeResult:
    """
    Materialize derived artifacts under ``bundle_dir`` from ``world_truth.json``.

    Does not modify ``world_truth.json``. Re-running with the same truth yields identical
    checksums when ``creation_timestamp`` is fixed in truth metadata.
    """
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    truth_file = Path(truth_path) if truth_path is not None else bundle / "world_truth.json"
    if not truth_file.is_file():
        raise FileNotFoundError(f"world_truth.json not found: {truth_file}")

    truth = load_world_truth(truth_file)
    world_id = str(truth["metadata"]["world_id"])
    if bundle.name != world_id and bundle.resolve().name != world_id:
        pass  # allow bundle_dir path ending in world_id parent

    written: list[str] = []

    panel_df = build_panel_dataframe(truth)
    panel_path = bundle / "panel.parquet"
    if panel_path.exists() and not overwrite:
        raise FileExistsError(f"{panel_path} exists; pass overwrite=True")
    panel_df.to_parquet(panel_path, index=False)
    written.append("panel.parquet")

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
    meta_path = bundle / "metadata.json"
    write_json(meta_path, metadata)
    written.append("metadata.json")

    checksums = compute_checksums(bundle, truth_path=truth_file)
    checksum_path = bundle / "checksums.json"
    write_json(checksum_path, checksums)
    written.append("checksums.json")

    return MaterializeResult(
        bundle_dir=bundle,
        world_id=world_id,
        files_written=tuple(written),
        checksums=checksums,
    )
