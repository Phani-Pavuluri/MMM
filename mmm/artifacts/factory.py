"""Resolve artifact stores from config — local is supported; MLflow is experimental."""

from __future__ import annotations

import os
from typing import Any

from mmm.artifacts.base import ArtifactStoreBase
from mmm.artifacts.compatibility import classify_backend, validate_backend_config
from mmm.artifacts.stores.local import LocalArtifactStore
from mmm.config.schema import ArtifactBackend, MMMConfig


def artifact_backend_disclosure(config: MMMConfig) -> dict[str, Any]:
    """Machine-readable artifact backend posture for extension reports and model cards."""
    return classify_backend(config)


def resolve_artifact_store(config: MMMConfig) -> ArtifactStoreBase:
    """
    Return an artifact store for training runs.

    ``local`` is fully wired (trainer, fingerprints, extension_report, model_card).
    ``mlflow`` delegates to optional MLflow tracking — file URIs are CI-tested; remote is not guaranteed.
    """
    validate_backend_config(config)
    if config.artifacts.backend == ArtifactBackend.MLFLOW:
        from mmm.artifacts.stores.mlflow import MLflowArtifactStore

        experiment = config.artifacts.mlflow_experiment or "mmm_runs"
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            root = os.path.abspath(config.artifacts.run_dir)
            tracking_uri = f"file:{root}/_mlflow_tracking"
            os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
        return MLflowArtifactStore(experiment_name=experiment, tracking_uri=tracking_uri)
    return LocalArtifactStore(config.artifacts.run_dir)
