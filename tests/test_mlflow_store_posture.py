"""MLflow backend is optional and not integration-guaranteed for remote servers (Phase 10)."""


def test_mlflow_module_documents_experimental_posture() -> None:
    import mmm.artifacts.stores.mlflow as mlflow_store

    doc = (mlflow_store.__doc__ or "").lower()
    assert "experimental" in doc or "not" in doc
