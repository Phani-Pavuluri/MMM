"""Split-channel governance blocks optimization when separability is low."""

from __future__ import annotations

from mmm.governance.split_channel_policy import apply_split_channel_governance


def test_low_separability_high_importance_blocks_optimization() -> None:
    er = {
        "governance": {"approved_for_optimization": True, "notes": []},
        "feature_separability_report": {
            "skipped": False,
            "unsupported_split_level_claims": [
                {
                    "feature_group": "Meta",
                    "member_columns": ["Meta_a", "Meta_b"],
                    "claim": "split_level_attribution_or_incrementality",
                }
            ],
            "feature_groups": [
                {
                    "feature_group": "Meta",
                    "separability_classification": "low",
                    "business_importance": {
                        "importance_band": "high",
                        "material_spend_share": True,
                    },
                }
            ],
        },
    }
    apply_split_channel_governance(er)
    gov = er["governance"]
    assert gov["approved_for_optimization"] is False
    assert gov["split_level_claims_supported"] is False
    assert gov["split_level_optimization_blocked"] is True
    assert any("split_level_attribution_blocked" in n for n in gov["notes"])
