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
