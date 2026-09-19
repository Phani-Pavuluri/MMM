"""Focused tests for the Tier-1 batch runner (deterministic, fast subset)."""

from __future__ import annotations

from mmm.validation.synthetic.tier1_batch_runner import (
    ANCHOR_WORLD_IDS,
    TIER1_MIN_WORLDS,
    build_tier1_characterization,
    run_tier1_batch,
    tier1_behavioral_specs,
    tier1_manifest,
    tier1_smoke_specs,
)


def test_manifest_is_deterministic_and_sized() -> None:
    first, second = tier1_manifest(), tier1_manifest()
    assert first == second
    assert first["world_count"] >= TIER1_MIN_WORLDS
    smoke = first["strata"]["smoke_structural"]
    behavioral = first["strata"]["behavioral_rich_dgp"]
    assert len(smoke) >= 70
    assert len(behavioral) >= 20
    assert list(first["strata"]["anchors"]) == list(ANCHOR_WORLD_IDS)
    world_ids = (
        [s["world_id"] for s in smoke]
        + [s["world_id"] for s in behavioral]
        + list(ANCHOR_WORLD_IDS)
    )
    assert len(set(world_ids)) == len(world_ids)


def test_smoke_subset_executes_structural_only(tmp_path) -> None:
    specs = tier1_smoke_specs()[:2]
    batch = run_tier1_batch(
        repo_root=tmp_path,
        work_root=tmp_path / "work",
        smoke=specs,
        behavioral=(),
        include_anchors=False,
    )
    assert batch["world_count"] == 2
    for outcome in batch["outcomes"]:
        assert outcome["structural_status"] in ("pass", "fail")
        assert outcome["error"] is None
        assert all(v is None for v in outcome["recovery"].values())


def test_behavioral_single_world_scores_recovery(tmp_path) -> None:
    specs = [s for s in tier1_behavioral_specs() if s.world_type == "exact_recovery"][:1]
    assert specs
    batch = run_tier1_batch(
        repo_root=tmp_path,
        work_root=tmp_path / "work",
        smoke=(),
        behavioral=tuple(specs),
        include_anchors=False,
    )
    assert batch["world_count"] == 1
    recovery = batch["outcomes"][0]["recovery"]
    assert recovery["coefficient_recovery"] in ("pass", "fail")


def test_characterization_is_report_only(tmp_path) -> None:
    specs = tier1_smoke_specs()[:2]
    batch = run_tier1_batch(
        repo_root=tmp_path,
        work_root=tmp_path / "work",
        smoke=specs,
        behavioral=(),
        include_anchors=False,
    )
    characterization = build_tier1_characterization(batch)
    assert characterization["tier"] == "tier_1_calibration"
    assert "provisional" in characterization["recommendation_disclaimer"]
    for rec in characterization["threshold_recommendations"]:
        assert rec["approval_status"] == "recommendation_only_not_approved"
