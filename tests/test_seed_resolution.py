"""Unified seed contract: inheritance, overrides, artifacts."""

from __future__ import annotations

import json

from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract


def test_identical_config_identical_resolution() -> None:
    a = MMMConfig(random_seed=7, data={"channel_columns": ["c1"]})
    b = MMMConfig(random_seed=7, data={"channel_columns": ["c1"]})
    ra = resolve_seed_contract(a)
    rb = resolve_seed_contract(b)
    assert ra["master_seed"] == 7
    assert ra["resolved_child_seeds"] == rb["resolved_child_seeds"]
    assert ra["inheritance_source"] == rb["inheritance_source"]


def test_child_override_behavior() -> None:
    cfg = MMMConfig(
        random_seed=11,
        ridge_bo={"sampler_seed": 99},
        bootstrap_seed=55,
        data={"channel_columns": ["c1"]},
    )
    art = resolve_seed_contract(cfg)
    assert art["resolved_child_seeds"]["ridge_bo.sampler_seed"] == 99
    assert art["inheritance_source"]["ridge_bo.sampler_seed"] == "explicit"
    assert art["resolved_child_seeds"]["bootstrap_seed"] == 55
    assert art["inheritance_source"]["bootstrap_seed"] == "explicit"
    assert art["resolved_child_seeds"]["extension_seed"] != 11
    assert art["inheritance_source"]["extension_seed"] == "inherited_from_master"


def test_direct_inherit_sampler_from_master() -> None:
    cfg = MMMConfig(random_seed=42, data={"channel_columns": ["c1"]})
    art = resolve_seed_contract(cfg)
    assert art["resolved_child_seeds"]["ridge_bo.sampler_seed"] == 42
    assert art["inheritance_source"]["ridge_bo.sampler_seed"] == "inherited_from_master"


def test_reproducibility_across_runs() -> None:
    payloads = []
    for _ in range(3):
        cfg = MMMConfig(random_seed=3, data={"channel_columns": ["c1"]})
        payloads.append(resolve_seed_contract(cfg))
    assert payloads[0] == payloads[1] == payloads[2]


def test_artifact_fields_present() -> None:
    cfg = MMMConfig(random_seed=1, data={"channel_columns": ["c1"]})
    art = resolve_seed_contract(cfg)
    blob = json.dumps(art, sort_keys=True)
    assert "master_seed" in blob
    assert "resolved_child_seeds" in blob
    assert "inheritance_source" in blob
