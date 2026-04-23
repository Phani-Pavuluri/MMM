"""Context gate: full-panel budget optimizer must run inside an approved decision pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_decision_pipeline: ContextVar[bool] = ContextVar("mmm_decision_pipeline", default=False)


def decision_pipeline_active() -> bool:
    return _decision_pipeline.get()


@contextmanager
def allow_decision_pipeline() -> None:
    """Allow ``optimize_budget_via_simulation`` (used by tests and ``run_decision_optimization``)."""
    tok = _decision_pipeline.set(True)
    try:
        yield
    finally:
        _decision_pipeline.reset(tok)
