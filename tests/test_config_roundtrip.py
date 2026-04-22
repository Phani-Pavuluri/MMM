from pathlib import Path

from mmm.config.load import dump_resolved_config, load_config
from mmm.config.schema import MMMConfig


def test_yaml_roundtrip(tmp_path: Path):
    cfg = MMMConfig(
        data={
            "path": "x.csv",
            "geo_column": "geo_id",
            "week_column": "week",
            "target_column": "y",
            "channel_columns": ["a", "b"],
        }
    )
    p = tmp_path / "cfg.yaml"
    dump_resolved_config(cfg, p)
    cfg2 = load_config(p)
    assert cfg2.data.channel_columns == ["a", "b"]
