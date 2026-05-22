"""Docs must not claim unsupported production transforms (audit fix P4)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_canonical_transform_stack() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "geometric" in text.lower()
    assert "hill" in text.lower()
    assert "production" in text.lower() or "canonical" in text.lower()
    assert "weibull" not in text.lower() or "not supported" in text.lower() or "research" in text.lower()


def test_config_yaml_documents_canonical_only() -> None:
    text = (ROOT / "docs/01_getting_started/config_yaml.md").read_text(encoding="utf-8")
    assert "geometric" in text
    assert "Hill" in text or "hill" in text
    assert "production" in text.lower() or "canonical" in text.lower() or "Ridge+BO" in text


def test_config_yaml_documents_replay_gap_controls() -> None:
    text = (ROOT / "docs/01_getting_started/config_yaml.md").read_text(encoding="utf-8")
    assert "replay_generalization_gap_threshold" in text
    assert "block_on_severe_replay_gap" in text
    assert "full_panel_transform_estimand_mask" in text
    assert "legacy_replay_deprecated_use_evidence_registry" in text
    assert "not" in text.lower() and "causal" in text.lower()


def test_calibration_docs_full_panel_replay_semantics() -> None:
    text = (ROOT / "docs/02_concepts/calibration.md").read_text(encoding="utf-8")
    assert "full_panel_transform_estimand_mask" in text
    assert "adstock" in text.lower()
    assert "estimand mask" in text.lower() or "estimand_mask" in text
    assert "legacy_replay_deprecated_use_evidence_registry" in text
    assert "causal" in text.lower()


def test_prod_checklist_replay_release_items() -> None:
    text = (ROOT / "docs/04_governance/prod_safety_checklist.md").read_text(encoding="utf-8")
    assert "full_panel_transform_estimand_mask" in text or "full-panel replay transform" in text.lower()
    assert "replay_holdout_available" in text
    assert "block_on_severe_replay_gap" in text
    assert "legacy_replay_deprecated" in text


def test_artifact_schema_replay_disclosure_fields() -> None:
    text = (ROOT / "docs/04_governance/artifact_schema.md").read_text(encoding="utf-8")
    for field in (
        "replay_transform_mode",
        "replay_uses_full_panel_transform",
        "lift_evaluated_on_estimand_mask_only",
        "calibration_refit_mode",
        "replay_generalization_gap",
        "replay_overfit_warning",
    ):
        assert field in text
