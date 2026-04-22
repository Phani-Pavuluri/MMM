"""Bayesian experiment likelihood helpers (extend with PyMC symbolic ops as needed)."""

from __future__ import annotations

# Intentionally lightweight: PyMC-specific likelihood wiring lives next to the
# generative model in `pymc_trainer` to keep tensors in-model scope.
