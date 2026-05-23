"""Package and CI Python version contracts (3.11+; no 3.10 support)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_requires_python_311_plus() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    spec = data["project"]["requires-python"]
    assert spec == ">=3.11", spec
    classifiers = data["project"].get("classifiers") or []
    py_classifiers = [c for c in classifiers if c.startswith("Programming Language :: Python :: 3.")]
    assert "Programming Language :: Python :: 3.10" not in py_classifiers
    assert "Programming Language :: Python :: 3.11" in py_classifiers
    assert "Programming Language :: Python :: 3.12" in py_classifiers


def test_ci_matrix_excludes_python_310() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix_match = re.search(r"python-version:\s*\[([^\]]+)\]", ci)
    assert matrix_match is not None, "ci.yml test matrix python-version not found"
    versions = matrix_match.group(1)
    assert "3.10" not in versions
    assert "3.11" in versions
    assert "3.12" in versions
