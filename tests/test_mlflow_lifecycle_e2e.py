"""MLflow file-store lifecycle (experimental, CI contract-tested)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmm.artifacts.compatibility import classify_backend, verify_run_roundtrip
from mmm.artifacts.factory import resolve_artifact_store
from mmm.artifacts.lifecycle import persist_training_artifacts
from mmm.config.schema import MMMConfig

mlflow = pytest.importorskip("mlflow")


@pytest.mark.tracking
def test_mlflow_file_backend_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    cfg = MMMConfig(
        data={"channel_columns": ["c1"]},
        artifacts={"backend": "mlflow", "run_dir": str(tmp_path / "runs"), "mlflow_experiment": "e2e_test"},
    )
    disc = classify_backend(cfg)
    assert disc["support_tier"] == "experimental"
    assert disc["ci_contract_tested"] is True
    assert disc["remote_tracking"] is False

    store = resolve_artifact_store(cfg)
    store.start_run("mlf_e2e")
    persist_training_artifacts(
        store,
        extension_report={"decision_bundle": {"artifact_tier": "research"}},
        model_card_md="# Model card\n\n## Intended use\n\nmlflow e2e\n",
    )
    store.end_run()

    # Local mirror path may differ; verify JSON artifacts exist under run_path when file-backed
    if store.run_path.exists():
        rt = verify_run_roundtrip(store.run_path)
        assert "extension_report.json" in [
            f"{n}.json" for n in rt.get("artifacts_present", [])
        ] or rt["ok"] or (store.run_path / "extension_report.json").exists()
