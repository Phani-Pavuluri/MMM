#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mmm-dev-validation:python3.11"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required for repository-standard local validation." >&2
  exit 1
}

docker build --file "${ROOT}/.devcontainer/Dockerfile" --tag "${IMAGE}" "${ROOT}/.devcontainer"
docker run --rm \
  --env MLFLOW_ALLOW_FILE_STORE=true \
  --mount "type=bind,source=${ROOT},target=/source,readonly" \
  "${IMAGE}" \
  bash -lc '
    cp -a /source /tmp/mmm
    cd /tmp/mmm
    python -m pip install -U pip
    python -m pip install -e ".[dev,tracking]"
    ruff check mmm tests scripts
    python -m mmm.validation.synthetic.hierarchy_evidence_validator --smoke VAL-BAYES-H2B-SMOKE
    pytest tests/research/test_bayes_h3_sandbox_guardrails.py -q -m "not slow"
    pytest tests/ -q -m "not slow"
  '
