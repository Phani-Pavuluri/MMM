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
    set -euo pipefail
    run() {
      local name="$1"
      shift
      echo "### STEP: ${name}"
      "$@" && echo "### PASS: ${name}" || { echo "### FAIL: ${name}"; return 1; }
    }
    run copy-source cp -a /source /tmp/mmm
    run enter-tree cd /tmp/mmm
    run pip-setup python -m pip install -U pip
    run install-package python -m pip install -e ".[dev,tracking]"
    run ruff-check ruff check mmm tests scripts
    run hierarchy-smoke python -m mmm.validation.synthetic.hierarchy_evidence_validator --smoke VAL-BAYES-H2B-SMOKE
    run guardrails-pytest pytest tests/research/test_bayes_h3_sandbox_guardrails.py -q -m "not slow"
    run full-pytest pytest tests/ -q -m "not slow"
    echo "### ALL STEPS PASSED"
  '
