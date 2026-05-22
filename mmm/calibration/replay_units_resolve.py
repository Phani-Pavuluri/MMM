"""Resolve train vs holdout replay unit sets from calibration config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmm.calibration.replay_prod_gate import assert_replay_production_ready
from mmm.calibration.replay_split import split_replay_units
from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


def resolve_replay_unit_sets(
    config: MMMConfig,
    schema: PanelSchema,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """
    Return ``(train_units, holdout_units, split_meta)``.

    When ``use_replay_holdout_split`` is false, all units are in ``train_units`` and holdout is empty.
    When true, explicit paths or an auto-split from ``replay_units_path`` is used.
    """
    cal = config.calibration
    meta: dict[str, Any] = {"use_replay_holdout_split": bool(cal.use_replay_holdout_split)}
    if not cal.use_replay_calibration:
        meta["holdout_not_available_reason"] = "replay_calibration_not_enabled"
        return [], [], meta

    train_path = cal.train_replay_units_path
    holdout_path = cal.holdout_replay_units_path
    if train_path and holdout_path:
        train = load_calibration_units_from_json(Path(train_path))
        holdout = load_calibration_units_from_json(Path(holdout_path))
        assert_replay_production_ready(config, train + holdout, schema=schema)
        meta.update(
            {
                "split_mode": "explicit_paths",
                "n_train_replay_units": len(train),
                "n_holdout_replay_units": len(holdout),
                "holdout_not_available_reason": None if holdout else "explicit_holdout_path_empty",
            }
        )
        return train, holdout, meta

    if not cal.replay_units_path:
        meta["holdout_not_available_reason"] = "replay_units_path_missing"
        return [], [], meta

    all_units = load_calibration_units_from_json(Path(cal.replay_units_path))
    assert_replay_production_ready(config, all_units, schema=schema)
    if not cal.use_replay_holdout_split:
        meta.update({"split_mode": "all_units_train", "n_train_replay_units": len(all_units)})
        return all_units, [], meta

    train, holdout, split_meta = split_replay_units(
        all_units,
        holdout_fraction=float(cal.replay_holdout_fraction),
        min_train_units=int(cal.replay_holdout_min_train_units),
        min_holdout_units=int(cal.replay_holdout_min_holdout_units),
        seed=int(config.random_seed or 0),
    )
    meta.update({"split_mode": "auto_fraction", **split_meta})
    if split_meta.get("holdout_not_available_reason"):
        meta["holdout_not_available_reason"] = split_meta["holdout_not_available_reason"]
        return all_units, [], meta
    return train, holdout, meta
