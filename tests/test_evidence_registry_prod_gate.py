"""PR 2.5: production governance for evidence-registry replay."""

from __future__ import annotations

import pytest

from mmm.calibration.replay_prod_gate import (
    assert_evidence_registry_replay_production_ready,
    assert_replay_production_ready,
)
from tests.test_replay_prod_gate import _minimal_unit, _prod_schema_cfg


def _valid_summary(**overrides: object) -> dict:
    base = {
        "replay_mode_used": "evidence_registry",
        "n_evidence_units_loaded": 1,
        "n_evidence_units_used": 1,
        "n_evidence_units_rejected": 0,
        "unit_governance": [
            {
                "experiment_id": "e1",
                "channel": "c1",
                "quality_tier": "high",
                "compatibility_status": "compatible",
                "supports_subgeo_claims": False,
                "allocation_role": "computational_bridge_only",
                "allocation_method": "observed_spend_weighted",
                "allocation_required": False,
                "lift_se": 0.02,
                "evidence_weight": 0.9,
            }
        ],
        "rejected_evidence_reasons": [],
    }
    base.update(overrides)
    return base


def test_prod_evidence_registry_valid_summary_passes() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "evidence_weighting_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    summary = _valid_summary()
    units = [_minimal_unit(experiment_id="e1")]
    assert_evidence_registry_replay_production_ready(cfg, summary, schema=schema, units=units)
    assert_replay_production_ready(cfg, units, schema=schema, evidence_summary=summary)


def test_prod_evidence_registry_missing_summary_fails() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="evidence_weighted_replay_summary"):
        assert_replay_production_ready(cfg, [_minimal_unit()], schema=schema, evidence_summary=None)


def test_prod_evidence_registry_zero_used_fails() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    with pytest.raises(ValueError, match="n_evidence_units_used"):
        assert_evidence_registry_replay_production_ready(
            cfg, _valid_summary(n_evidence_units_used=0), schema=schema
        )


def test_prod_evidence_registry_missing_se_fails() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    row = dict(_valid_summary()["unit_governance"][0])
    row["lift_se"] = None
    with pytest.raises(ValueError, match="lift_se"):
        assert_evidence_registry_replay_production_ready(
            cfg, _valid_summary(unit_governance=[row]), schema=schema
        )


def test_prod_aggregate_subgeo_claim_fails() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    row = dict(_valid_summary()["unit_governance"][0])
    row["compatibility_status"] = "aggregate_only"
    row["supports_subgeo_claims"] = True
    with pytest.raises(ValueError, match="supports_subgeo_claims=false"):
        assert_evidence_registry_replay_production_ready(
            cfg, _valid_summary(unit_governance=[row]), schema=schema
        )


def test_prod_allocated_shock_without_bridge_role_fails() -> None:
    cfg, schema = _prod_schema_cfg(None)
    cfg = cfg.model_copy(
        update={
            "calibration": cfg.calibration.model_copy(
                update={
                    "replay_mode": "evidence_registry",
                    "evidence_registry_path": "/tmp/ev.json",
                    "compatibility_resolver_enabled": True,
                    "use_replay_calibration": True,
                }
            )
        }
    )
    row = dict(_valid_summary()["unit_governance"][0])
    row["allocation_required"] = True
    row["compatibility_status"] = "allocation_required"
    row["allocation_role"] = "experimental_dma_truth"
    with pytest.raises(ValueError, match="computational_bridge_only"):
        assert_evidence_registry_replay_production_ready(
            cfg, _valid_summary(unit_governance=[row]), schema=schema
        )


def test_legacy_replay_gate_unchanged_without_evidence_summary() -> None:
    cfg, schema = _prod_schema_cfg(None)
    assert_replay_production_ready(cfg, [_minimal_unit()], schema=schema)
