"""Prod override_unsafe requires waiver artifact."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mmm.config.schema import MMMConfig, RunEnvironment


def test_prod_override_unsafe_requires_waiver_path() -> None:
    with pytest.raises(ValueError, match="override_unsafe_waiver_path"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            override_unsafe=True,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": [], "data_version_id": "snap-1"},
            cv={"mode": "rolling"},
            objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
        )


def test_prod_override_unsafe_accepts_valid_waiver(tmp_path: Path) -> None:
    exp = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    waiver = tmp_path / "waiver.json"
    waiver.write_text(
        json.dumps(
            {
                "waiver_id": "waiver-001",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "reason": "incident response identifiability cap",
                "owner": "test",
                "expires_at": exp,
            }
        ),
        encoding="utf-8",
    )
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        override_unsafe=True,
        override_unsafe_waiver_path=str(waiver),
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "control_columns": [], "data_version_id": "snap-1"},
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
    )
    assert cfg.override_unsafe is True
