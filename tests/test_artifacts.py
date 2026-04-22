from pathlib import Path

from mmm.artifacts.stores.local import LocalArtifactStore


def test_local_store(tmp_path: Path):
    s = LocalArtifactStore(tmp_path)
    s.start_run("run1", metadata={"a": 1})
    s.log_metrics({"x": 1.0})
    s.log_dict("foo", {"b": 2})
    s.end_run()
    assert (s.run_path / "foo.json").exists()
