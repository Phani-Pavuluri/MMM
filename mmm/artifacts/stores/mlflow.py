"""Optional MLflow store — lazy import, never required for core.

**Support posture:** local file-store tracking is exercised in development; remote tracking URIs,
artifact server ACLs, and MLflow server version skew are **not** integration-guaranteed in this package.
Treat ``ArtifactBackend.MLFLOW`` as **experimental** unless your environment pins MLflow and runs your
own contract tests. Prefer ``ArtifactBackend.LOCAL`` for reproducible CI and audited decision bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmm.artifacts.base import ArtifactStoreBase


class MLflowArtifactStore(ArtifactStoreBase):
    """Wraps MLflow tracking API; raises ImportError if mlflow not installed."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None) -> None:
        try:
            import mlflow  # type: ignore
        except ImportError as e:  # pragma: no cover - optional path
            raise ImportError("mlflow is required for MLflowArtifactStore") from e
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self._active = False
        self._run_path = Path(".")

    @property
    def run_path(self) -> Path:
        return self._run_path

    def start_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        self._run = self._mlflow.start_run(run_name=run_id)
        self._active = True
        if metadata:
            self._mlflow.log_params({k: str(v) for k, v in metadata.items()})
        art = self._mlflow.get_artifact_uri()
        self._run_path = Path(art.replace("file://", "")) if art else Path(".")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        self._mlflow.log_artifacts(str(path), artifact_path=name)

    def log_dict(self, name: str, payload: dict[str, Any]) -> None:
        import tempfile

        import json

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{name}.json"
            p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self._mlflow.log_artifact(str(p), artifact_path="json")

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active:
            self._mlflow.end_run(status)
