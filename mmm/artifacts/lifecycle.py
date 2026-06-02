"""Persist canonical training-run artifacts through the artifact store API."""

from __future__ import annotations

from typing import Any

from mmm.artifacts.base import ArtifactStoreBase
from mmm.artifacts.compatibility import MODEL_CARD_FILENAME


def persist_training_artifacts(
    store: ArtifactStoreBase,
    *,
    extension_report: dict[str, Any],
    data_fingerprint: dict[str, Any] | None = None,
    seed_resolution: dict[str, Any] | None = None,
    config_fingerprint: dict[str, Any] | None = None,
    model_card_md: str | None = None,
) -> dict[str, str]:
    """
    Write lifecycle artifacts via ``log_dict`` / ``log_artifact``.

    Returns map of logical name → persisted path label.
    """
    written: dict[str, str] = {}
    if data_fingerprint is not None:
        store.log_dict("data_fingerprint", data_fingerprint)
        written["data_fingerprint"] = "data_fingerprint.json"
    if seed_resolution is not None:
        store.log_dict("seed_resolution", seed_resolution)
        written["seed_resolution"] = "seed_resolution.json"
    if config_fingerprint is not None:
        store.log_dict("config_fingerprint", config_fingerprint)
        written["config_fingerprint"] = "config_fingerprint.json"
    bundle = extension_report.get("decision_bundle")
    if isinstance(bundle, dict):
        store.log_dict("decision_bundle", bundle)
        written["decision_bundle"] = "decision_bundle.json"
    from mmm.diagnostics.ridge_diagnostic_summary import export_ridge_diagnostic_artifacts

    ridge_written = export_ridge_diagnostic_artifacts(store, extension_report)
    written.update(ridge_written)
    store.log_dict("extension_report", extension_report)
    written["extension_report"] = "extension_report.json"
    if model_card_md is not None:
        card_path = store.run_path / MODEL_CARD_FILENAME
        card_path.write_text(model_card_md, encoding="utf-8")
        store.log_artifact("model_card", card_path)
        written["model_card"] = MODEL_CARD_FILENAME
    return written
