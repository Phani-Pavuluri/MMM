#!/usr/bin/env python3
"""Validate docs tree: internal links, inventory coverage, orphan detection."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

LINK_RE = re.compile(r"\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def _md_files() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if p.is_file())


def _resolve_link(source: Path, target: str) -> Path | None:
    t = target.strip().split("#", 1)[0]
    if not t or any(t.startswith(p) for p in SKIP_PREFIXES):
        return None
    candidate = ROOT / t.lstrip("/") if t.startswith("/") else (source.parent / t).resolve()
    if candidate.is_dir():
        candidate = candidate / "README.md"
    return candidate


def validate_links() -> list[str]:
    errors: list[str] = []
    for md in _md_files():
        text = md.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            raw = match.group(1).split()[0]
            resolved = _resolve_link(md, raw)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(f"{md.relative_to(ROOT)}: broken link -> {raw}")
    return errors


def find_orphans(inventory_canonical: set[str]) -> list[str]:
    orphans: list[str] = []
    for md in _md_files():
        rel = md.relative_to(DOCS).as_posix()
        if rel in ("README.md", "documentation_truth_audit.md", "DOCUMENTATION_INVENTORY.md"):
            continue
        if rel.startswith("_archive/"):
            continue
        if rel not in inventory_canonical:
            orphans.append(rel)
    return orphans


def load_inventory_paths() -> set[str]:
    inv = DOCS / "DOCUMENTATION_INVENTORY.md"
    if not inv.exists():
        return set()
    paths: set[str] = set()
    for line in inv.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `docs/"):
            part = line.split("|")[1].strip().strip("`")
            if part.startswith("docs/"):
                paths.add(part.removeprefix("docs/"))
    return paths


def main() -> int:
    errors = validate_links()
    inventory = load_inventory_paths()
    orphans = find_orphans(inventory) if inventory else []

    if errors:
        print("Broken links:")
        for e in errors:
            print(f"  {e}")
    if orphans:
        print("Orphan pages (not in DOCUMENTATION_INVENTORY.md):")
        for o in orphans:
            print(f"  {o}")

    if errors:
        return 1
    print(f"OK: {len(_md_files())} markdown files, links resolve.")
    if orphans:
        print(f"Note: {len(orphans)} orphan(s) listed (informational).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
