"""Artifact backend factory and disclosure."""

from __future__ import annotations

from mmm.artifacts.factory import artifact_backend_disclosure, resolve_artifact_store
from mmm.artifacts.stores.local import LocalArtifactStore
from mmm.config.schema import ArtifactBackend, MMMConfig


def test_local_backend_resolves_to_local_store() -> None:
    cfg = MMMConfig(
        data={"channel_columns": ["c1"]},
        artifacts={"backend": "local", "run_dir": "./tmp_test_runs"},
    )
    store = resolve_artifact_store(cfg)
    assert isinstance(store, LocalArtifactStore)


def test_mlflow_backend_marked_experimental(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    cfg = MMMConfig(
        data={"channel_columns": ["c1"]},
        artifacts={"backend": "mlflow", "mlflow_experiment": "test"},
    )
    disc = artifact_backend_disclosure(cfg)
    assert disc["backend"] == ArtifactBackend.MLFLOW.value
    assert disc["support_tier"] == "experimental"
    assert disc["remote_tracking"] is False
    assert disc["ci_contract_tested"] is True
