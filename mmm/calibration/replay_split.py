"""Deterministic train/holdout splits for replay calibration units."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.calibration.contracts import CalibrationUnit


def split_replay_units(
    units: list[CalibrationUnit],
    *,
    holdout_fraction: float,
    min_train_units: int,
    min_holdout_units: int,
    seed: int,
) -> tuple[list[CalibrationUnit], list[CalibrationUnit], dict[str, Any]]:
    """
    Split replay units into train (BO objective) and holdout (diagnostics only).

    Split is reproducible: sort by ``unit_id``, then permute with ``seed``.
    """
    n = len(units)
    if n < min_train_units + min_holdout_units:
        return (
            [],
            [],
            {
                "holdout_not_available_reason": (
                    f"insufficient_replay_units: need>={min_train_units + min_holdout_units}, got {n}"
                ),
                "n_total_units": n,
            },
        )
    n_holdout = max(min_holdout_units, int(round(n * holdout_fraction)))
    n_holdout = min(n_holdout, n - min_train_units)
    if n_holdout < min_holdout_units:
        return (
            [],
            [],
            {
                "holdout_not_available_reason": (
                    f"holdout_fraction_yields_too_few_holdout_units: n_holdout={n_holdout}, "
                    f"min_holdout_units={min_holdout_units}"
                ),
                "n_total_units": n,
            },
        )

    ordered = sorted(units, key=lambda u: str(u.unit_id))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    holdout_idx = set(int(i) for i in perm[:n_holdout])
    train = [u for i, u in enumerate(ordered) if i not in holdout_idx]
    holdout = [u for i, u in enumerate(ordered) if i in holdout_idx]
    return (
        train,
        holdout,
        {
            "holdout_not_available_reason": None,
            "n_total_units": n,
            "n_train_replay_units": len(train),
            "n_holdout_replay_units": len(holdout),
            "holdout_fraction_requested": float(holdout_fraction),
            "split_seed": int(seed),
        },
    )
