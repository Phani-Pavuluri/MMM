"""Bayesian extension surfaces: typed posterior exploration quantity (research-only semantics)."""

from mmm.contracts.quantity_models import PosteriorExplorationQuantityResult


def test_posterior_exploration_quantity_section_contract() -> None:
    sec = PosteriorExplorationQuantityResult(
        draw_summary={"surface": "extension_train_diagnostic", "note": "fixture"},
        validity_diagnostics={"framework": "bayesian"},
    ).section_dict()
    assert sec["quantity_contract_version"] == "mmm_quantity_envelope_v1"
    assert sec["estimand_kind"] == "posterior_exploration"
    assert sec["prod_decisioning_allowed"] is False
    assert "research" in sec["allowed_surfaces"]
