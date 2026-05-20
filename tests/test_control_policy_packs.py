"""Control policy packs are guidance-only."""

from __future__ import annotations

from mmm.governance.control_policy_packs import all_pack_summaries, policy_pack_recommendations


def test_all_packs_listed() -> None:
    summaries = all_pack_summaries(configured_controls=["holiday_indicator"])
    ids = {s["pack_id"] for s in summaries}
    assert "generic" in ids
    assert "b2b" in ids
    assert "retail" in ids
    assert "subscription" in ids
    assert "geo_experimentation" in ids
    b2b = policy_pack_recommendations("b2b", configured_controls=[])
    assert b2b["guidance_only"] is True
    assert "sp500_index" in b2b["recommended_controls"]
