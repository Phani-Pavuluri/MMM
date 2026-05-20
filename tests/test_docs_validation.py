"""Documentation inventory and link validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_inventory_exists() -> None:
    inv = ROOT / "docs" / "DOCUMENTATION_INVENTORY.md"
    assert inv.is_file()
    text = inv.read_text(encoding="utf-8")
    assert "Canonical location" in text
    assert "04_governance/artifact_schema.md" in text


def test_validate_docs_script_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_docs.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
