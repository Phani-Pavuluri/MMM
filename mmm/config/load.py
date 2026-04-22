"""Load and persist YAML configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mmm.config.schema import MMMConfig


def load_config(path: str | Path) -> MMMConfig:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {p} must be a mapping")
    return MMMConfig.model_validate(raw)


def resolve_config(cfg: MMMConfig) -> MMMConfig:
    """Apply environment-derived defaults (normalization profile) before training."""
    from mmm.config.validators import apply_environment_objective_profile

    return apply_environment_objective_profile(cfg)


def dump_resolved_config(cfg: MMMConfig, dest: str | Path) -> None:
    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump_resolved()
    if d.suffix in {".yaml", ".yml"}:
        d.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        d.write_text(json.dumps(payload, indent=2), encoding="utf-8")
