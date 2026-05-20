# Development environment setup

## Dependency system

MMM uses **PEP 621 + setuptools** (`pyproject.toml`, `[project.optional-dependencies]`). CI installs with:

```bash
pip install -e ".[dev,tracking]"
```

There is **no** `poetry.lock`. A Poetry-based devcontainer is not used until the project adopts Poetry as the single source of truth.

## Dev container (recommended)

Open the repo in VS Code / Cursor and **Reopen in Container**. The image:

- Base: `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
- Apt: `git`, `build-essential` only (no Node/Yarn; stale Yarn apt lists removed)
- No Docker socket, privileged mode, or host networking
- Workspace bind-mount: this repo → `/workspace/mmm`

`postCreateCommand` runs `pip install -e ".[dev,tracking]"` (includes `pytest`, `ruff`, `mypy`, MLflow for tracking tests).

## Local (without container)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,tracking]"
ruff check mmm tests scripts
pytest tests/ -q -m "not slow"
```

## Optional extras

```bash
pip install -e ".[dev,tracking,bayesian,bo]"
```
