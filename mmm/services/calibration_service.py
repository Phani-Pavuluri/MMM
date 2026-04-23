from __future__ import annotations

from typing import Any

import pandas as pd

from mmm.calibration.admissibility import experiment_admissibility_violations
from mmm.calibration.engine import CalibrationEngine
from mmm.calibration.matching import compute_experiment_weight_audit
from mmm.causal.estimand import EstimandValidator
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


def run_calibration_extensions(
    config: MMMConfig,
    *,
    panel: pd.DataFrame | None = None,
    schema: PanelSchema | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config.calibration.enabled and config.calibration.experiments_path:
        eng = CalibrationEngine(config.calibration.experiments_path)
        exps = eng.load()
        adm_samples: list[dict[str, Any]] = []
        for ex in exps:
            viol = experiment_admissibility_violations(ex, run_environment=config.run_environment)
            if viol:
                adm_samples.append({"experiment_id": ex.experiment_id, "violations": viol})
        out["calibration_admissibility"] = {
            "n_experiments_loaded": len(exps),
            "n_inadmissible": len(adm_samples),
            "inadmissible_sample": adm_samples[:25],
            "policy_note": "Rows with violations are excluded from prod matching paths; review before enabling calibration.enabled in prod.",
        }
        ev = EstimandValidator(config.extensions.estimand, config.calibration.experiment_target_kpi)
        evr = ev.validate_experiments(exps)
        out["estimand_validation"] = {"ok": evr.ok, "messages": evr.messages}
        if panel is not None and schema is not None:
            geos = {str(x) for x in panel[schema.geo_column].unique()}
            chans = set(schema.channel_columns)
            _matched, trace = eng.match_with_trace(
                exps,
                geos=geos,
                channels=chans,
                levels=list(config.calibration.match_levels),
                apply_quality=config.calibration.use_quality_weights,
                panel=panel,
                schema=schema,
                run_environment=config.run_environment,
            )
            audit = compute_experiment_weight_audit(_matched)
            n_match = len(_matched)
            mx = float(audit.get("max_inverse_se_share") or 0.0)
            if n_match >= 6 and mx <= 0.45:
                strength = "strong"
            elif n_match >= 3:
                strength = "moderate"
            else:
                strength = "weak"
            out["experiment_matching"] = {
                "n_matched": n_match,
                "matching_trace": trace,
                "weight_audit": audit,
                "evidence_strength": strength,
                "evidence_strength_rationale": {
                    "n_matched": n_match,
                    "max_inverse_se_share": mx,
                    "note": "Machine-readable heuristic for promotion hooks; not a substitute for design review.",
                },
            }
        else:
            out["experiment_matching"] = {
                "skipped": True,
                "reason": "panel_and_schema_required_for_matching_trace",
            }
    return out
