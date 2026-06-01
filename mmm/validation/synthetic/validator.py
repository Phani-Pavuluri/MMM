"""L1–L3 validation for synthetic world bundles (documentation: world_validator_spec.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmm.validation.synthetic._io import read_json, sha256_file
from mmm.validation.synthetic.dgp_materializer import DGP_MATERIALIZATION_VERSION
from mmm.validation.synthetic.materializer import (
    CHECKSUM_VERSION,
    MATERIALIZATION_VERSION,
    WORLD_CONTRACT_VERSION,
)
from mmm.validation.synthetic.replay_units import (
    SUPPORTED_REPLAY_TRANSFORM_MODES,
    lift_scale_supported,
    week_window_inside_time_truth,
)

ALLOWED_MATERIALIZATION_VERSIONS = frozenset(
    {MATERIALIZATION_VERSION, DGP_MATERIALIZATION_VERSION}
)

REQUIRED_TRUTH_SECTIONS = (
    "metadata",
    "time_truth",
    "geo_truth",
    "media_truth",
    "outcome_truth",
    "transform_truth",
    "coefficient_truth",
    "experiment_truth",
    "decision_truth",
    "drift_truth",
    "artifact_truth",
    "governance_truth",
)


@dataclass
class ValidationResult:
    passed: bool
    max_level: int
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    world_id: str | None = None

    def fail(self, check_id: str) -> None:
        self.hard_failures.append(check_id)
        self.passed = False


def verify_checksums(bundle_dir: Path, *, truth_path: Path | None = None) -> list[str]:
    """Return list of checksum mismatch check ids (empty if ok)."""
    bundle = Path(bundle_dir)
    failures: list[str] = []
    checksum_path = bundle / "checksums.json"
    if not checksum_path.is_file():
        return ["L3-012-missing-checksums"]
    recorded = read_json(checksum_path)
    truth_file = truth_path or bundle / "world_truth.json"

    expected_truth = sha256_file(truth_file)
    if recorded.get("world_truth_sha256") != expected_truth:
        failures.append("L3-012-world_truth_sha256")

    panel_path = bundle / "panel.parquet"
    if panel_path.is_file() and recorded.get("panel_sha256") != sha256_file(panel_path):
        failures.append("L3-012-panel_sha256")
    replay_path = bundle / "replay_units.json"
    if replay_path.is_file():
        digest = sha256_file(replay_path)
        if recorded.get("replay_sha256") != digest:
            failures.append("L3-012-replay_sha256")
        exp = recorded.get("experiment_sha256")
        if exp not in (digest, None) and exp != digest:
            failures.append("L3-012-experiment_sha256")
    decision_path = bundle / "decision_truth.json"
    if (
        decision_path.is_file()
        and "decision_truth_sha256" in recorded
        and recorded.get("decision_truth_sha256") != sha256_file(decision_path)
    ):
        failures.append("L3-012-decision_truth_sha256")
    meta_path = bundle / "metadata.json"
    if (
        meta_path.is_file()
        and "metadata_sha256" in recorded
        and recorded.get("metadata_sha256") != sha256_file(meta_path)
    ):
        failures.append("L3-012-metadata_sha256")
    return failures


def validate_bundle(bundle_dir: str | Path, *, max_level: int = 3) -> ValidationResult:
    """Run structural (L1), semantic (L2), and cross-object (L3) checks up to ``max_level``."""
    bundle = Path(bundle_dir)
    result = ValidationResult(passed=True, max_level=max_level)
    truth_path = bundle / "world_truth.json"
    if not truth_path.is_file():
        result.fail("L1-002")
        return result

    try:
        truth = read_json(truth_path)
    except (json.JSONDecodeError, ValueError):
        result.fail("L1-002-parse")
        return result

    if max_level >= 1:
        _validate_level1(bundle, truth, result)
    if result.passed and max_level >= 2:
        _validate_level2(truth, result)
    if result.passed and max_level >= 3:
        _validate_level3(bundle, truth, result)

    return result


def _validate_level1(bundle: Path, truth: dict[str, Any], result: ValidationResult) -> None:
    meta = truth.get("metadata")
    if not isinstance(meta, dict):
        result.fail("L1-005-metadata")
        return
    result.world_id = str(meta.get("world_id", ""))

    for section in REQUIRED_TRUTH_SECTIONS:
        if section not in truth:
            result.fail(f"L1-003-{section}")

    if meta.get("world_contract_version") != WORLD_CONTRACT_VERSION:
        result.fail("L1-004")

    world_id = str(meta.get("world_id", ""))
    if bundle.name != world_id:
        result.fail("L1-001-world_id_dir")

    for key in (
        "world_id",
        "world_version",
        "world_contract_version",
        "world_generator_version",
        "materialization_version",
        "generation_seed",
        "scenario_tags",
        "creation_timestamp",
    ):
        if key not in meta:
            result.fail(f"L1-005-metadata-{key}")

    if not (bundle / "metadata.json").is_file():
        result.fail("L1-006")
    else:
        md = read_json(bundle / "metadata.json")
        for key in (
            "world_id",
            "world_version",
            "world_contract_version",
            "world_generator_version",
            "materialization_version",
            "seed",
            "creation_timestamp",
            "scenario_tags",
            "checksum_version",
        ):
            if key not in md:
                result.fail(f"L1-006-metadata-{key}")
        if md.get("checksum_version") != CHECKSUM_VERSION:
            result.fail("L1-008")

    if not (bundle / "checksums.json").is_file():
        result.fail("L1-007")
    else:
        cs = read_json(bundle / "checksums.json")
        if cs.get("checksum_version") != CHECKSUM_VERSION:
            result.fail("L1-008-checksums")

    if not (bundle / "panel.parquet").is_file():
        result.fail("L1-009-panel")


def _validate_level2(truth: dict[str, Any], result: ValidationResult) -> None:
    geo = truth.get("geo_truth") or {}
    media = truth.get("media_truth") or {}
    coef = truth.get("coefficient_truth") or {}
    transform = truth.get("transform_truth") or {}

    weights = geo.get("weights") or {}
    if weights:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-9:
            result.fail("L2-003")

    geos = list(geo.get("geos") or [])
    if int(geo.get("n_geos", -1)) != len(geos):
        result.fail("L2-004")
    if len(geos) != len(set(geos)):
        result.fail("L2-004-unique")

    channels = list(media.get("channels") or [])
    if not channels or len(channels) != len(set(channels)):
        result.fail("L2-005")

    betas = coef.get("true_beta_by_channel") or {}
    if set(betas.keys()) != set(channels):
        result.fail("L2-006")

    for v in [coef.get("intercept"), *betas.values()]:
        if v is None:
            result.fail("L2-007")
            continue
        fv = float(v)
        if fv != fv:  # NaN
            result.fail("L2-007")
        if abs(fv) == float("inf"):
            result.fail("L2-007-inf")

    for ch in channels:
        decay = (transform.get("adstock_decay_by_channel") or {}).get(ch)
        if decay is not None and not (0.0 < float(decay) < 1.0):
            result.fail("L2-008")
        half = (transform.get("hill_half_max_by_channel") or {}).get(ch)
        slope = (transform.get("hill_slope_by_channel") or {}).get(ch)
        if half is not None and float(half) <= 0:
            result.fail("L2-009")
        if slope is not None and float(slope) <= 0:
            result.fail("L2-009-slope")

    spend = media.get("baseline_spend_by_channel") or {}
    for v in spend.values():
        if float(v) < 0:
            result.fail("L2-010")

    time_t = truth.get("time_truth") or {}
    if (
        time_t.get("start_date")
        and time_t.get("end_date")
        and str(time_t["end_date"]) < str(time_t["start_date"])
    ):
        result.fail("L2-001")

    n_periods = int(time_t.get("n_periods", 0))
    tw = time_t.get("train_window") or {}
    if tw:
        end_idx = int(tw.get("end_period_index", -1))
        if end_idx >= n_periods:
            result.fail("L2-002")


def _validate_level3(bundle: Path, truth: dict[str, Any], result: ValidationResult) -> None:
    meta_truth = truth["metadata"]
    meta_bundle = read_json(bundle / "metadata.json")

    if meta_bundle.get("world_id") != meta_truth.get("world_id"):
        result.fail("L3-001-world_id")
    for key in ("world_contract_version", "world_generator_version", "world_version"):
        if meta_bundle.get(key) != meta_truth.get(key):
            result.fail(f"L3-001-{key}")
    if int(meta_bundle.get("seed", -1)) != int(meta_truth.get("generation_seed", -2)):
        result.fail("L3-001-seed")
    if meta_bundle.get("materialization_version") not in ALLOWED_MATERIALIZATION_VERSIONS:
        result.fail("L3-001-materialization_version")

    channels = set(truth["media_truth"]["channels"])
    geo_col = str(truth["geo_truth"].get("geo_column_name") or "geo_id")
    week_col = str(truth["time_truth"].get("week_column_name") or "week_start_date")
    target_col = str(truth["outcome_truth"]["target_column"])

    import pandas as pd

    panel = pd.read_parquet(bundle / "panel.parquet")
    required_cols = {geo_col, week_col, target_col} | channels
    if not required_cols.issubset(panel.columns):
        result.fail("L3-008")

    decision_idx_path = bundle / "decision_truth.json"
    if decision_idx_path.is_file():
        if "true_beta" in decision_idx_path.read_text(encoding="utf-8").lower():
            result.fail("L3-007-duplicate-beta")
        scenarios = (truth.get("decision_truth") or {}).get("scenarios") or []
        for sc in scenarios:
            cand = sc.get("candidate_spend_by_channel") or {}
            if not set(cand.keys()).issubset(channels):
                result.fail("L3-002")

    _validate_replay_units(bundle, truth, result)

    for fail_id in verify_checksums(bundle):
        result.fail(fail_id)


def _validate_replay_units(bundle: Path, truth: dict[str, Any], result: ValidationResult) -> None:
    truth_units = (truth.get("experiment_truth") or {}).get("units") or []
    replay_path = bundle / "replay_units.json"
    if not truth_units:
        if replay_path.is_file():
            result.fail("L3-replay-unexpected-file")
        return
    if not replay_path.is_file():
        result.fail("L3-replay-missing-file")
        return

    raw = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        result.fail("L3-replay-not-list")
        return

    truth_by_id = {str(u["unit_id"]): u for u in truth_units}
    channels = set(truth["media_truth"]["channels"])
    geos = set(truth["geo_truth"]["geos"])
    time_t = truth["time_truth"]
    world_id = str(truth["metadata"]["world_id"])

    for row in raw:
        if not isinstance(row, dict):
            result.fail("L3-replay-row-type")
            continue
        uid = str(row.get("unit_id", ""))
        if uid not in truth_by_id:
            result.fail(f"L3-replay-unknown-unit:{uid}")
            continue

        if str(row.get("world_id", "")) != world_id:
            result.fail(f"L3-replay-world_id:{uid}")

        ch = str(row.get("channel", ""))
        if ch not in channels:
            result.fail(f"L3-replay-channel:{uid}")
        for tch in row.get("treated_channel_names") or []:
            if str(tch) not in channels:
                result.fail(f"L3-replay-treated-channel:{uid}")

        for g in row.get("geo_ids") or []:
            if str(g) not in geos:
                result.fail(f"L3-replay-geo:{uid}")

        tw = row.get("time_window") or {}
        ws = str(tw.get("week_start", row.get("week_start", "")))
        we = str(tw.get("week_end", row.get("week_end", "")))
        if not week_window_inside_time_truth(week_start=ws, week_end=we, time_truth=time_t):
            result.fail(f"L3-replay-window:{uid}")

        ls = str(row.get("lift_scale", ""))
        if not lift_scale_supported(ls):
            result.fail(f"L3-replay-lift-scale:{uid}")

        if not str(row.get("estimand", "")).strip():
            result.fail(f"L3-replay-estimand:{uid}")

        mode = str(row.get("replay_transform_mode", ""))
        re = row.get("replay_estimand") or {}
        re_mode = str(re.get("replay_transform_mode", mode))
        if re_mode not in SUPPORTED_REPLAY_TRANSFORM_MODES:
            result.fail(f"L3-replay-transform-mode:{uid}")

        truth_lift = float((truth_by_id[uid].get("lift_definition") or {}).get("value", 0))
        row_lift = float(row.get("lift", row.get("observed_lift", 0)))
        if abs(row_lift - truth_lift) > 1e-9:
            result.fail(f"L3-replay-lift-mismatch:{uid}")
