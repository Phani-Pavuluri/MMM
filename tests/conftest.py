"""Pytest configuration."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _decision_bundle_git_sha_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod CLI decision bundles require ``git_sha``; CI / sdist trees may lack ``.git``."""
    if not os.environ.get("MMM_GIT_SHA"):
        monkeypatch.setenv("MMM_GIT_SHA", "test-git-sha-not-from-repo")


@pytest.fixture(autouse=True)
def _prod_dataset_lineage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod configs accept ``MMM_DATA_VERSION_ID`` when ``data.data_version_id`` is unset (CI/tests)."""
    if not os.environ.get("MMM_DATA_VERSION_ID"):
        monkeypatch.setenv("MMM_DATA_VERSION_ID", "pytest-dataset-snapshot-id")


@pytest.fixture(autouse=True)
def _dependency_lock_digest_for_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision-tier lineage requires a lock digest when no uv.lock/poetry.lock is present."""
    if not os.environ.get("MMM_DEPENDENCY_LOCK_DIGEST"):
        monkeypatch.setenv("MMM_DEPENDENCY_LOCK_DIGEST", "pytest-dependency-lock-digest")
