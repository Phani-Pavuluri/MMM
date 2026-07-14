# Development validation workflow

## Standard local command

Run the repository validation suite with:

```bash
make validate
```

This builds the existing Python 3.11 devcontainer image and runs validation in a disposable Docker container. The
repository is mounted read-only and copied inside the container before dependencies or tests run, so validation does
not rewrite host-side generated artifacts.

`make validate-docker` is an explicit alias for the same Docker path.

## Validation authority

Host-local Poetry environments are not the source of truth. They may be used for quick developer feedback when they
work, but a stale or broken host interpreter must not block repository validation.

The preferred local validation environment is the repository devcontainer/Docker image using Python 3.11. It runs:

1. dependency installation from `.[dev,tracking]`;
2. Ruff over `mmm`, `tests`, and `scripts`;
3. the Bayes-H2b hierarchy smoke validator;
4. Bayes-H3 sandbox guardrail tests; and
5. the complete non-slow test suite.

GitHub Actions remains the final authority. CI validates Python 3.11 and Python 3.12, then runs the release-gate job.
Passing `make validate` is necessary local evidence, but does not replace those matrix and release-gate results.

## Cleanup

To remove common local cache and OS metadata files explicitly:

```bash
make clean-junk
```

This target removes `.DS_Store`, `__pycache__`, and repository-local pytest, Ruff, and mypy caches. It is not run by
`make validate`.
