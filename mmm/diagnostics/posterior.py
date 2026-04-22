"""Posterior / sampling diagnostics."""

from __future__ import annotations

from typing import Any


def summarize_idata(idata: Any) -> dict:
    try:
        import arviz as az  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("arviz required") from e
    summ = az.summary(idata, round_to=None)
    return {"summary_table": summ.to_dict()}
