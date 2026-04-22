from __future__ import annotations

from typing import Any

from mmm.calibration.engine import CalibrationEngine
from mmm.causal.estimand import EstimandValidator
from mmm.config.schema import MMMConfig


def run_calibration_extensions(config: MMMConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config.calibration.enabled and config.calibration.experiments_path:
        eng = CalibrationEngine(config.calibration.experiments_path)
        exps = eng.load()
        ev = EstimandValidator(config.extensions.estimand, config.calibration.experiment_target_kpi)
        evr = ev.validate_experiments(exps)
        out["estimand_validation"] = {"ok": evr.ok, "messages": evr.messages}
    return out
