"""Posterior draw export from PyMC-style xarray posteriors."""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config.schema import PoolingMode
from mmm.diagnostics.bayesian_draw_export import linear_coef_draws_from_pymc_idata


def test_linear_coef_draws_full_pooling_ok():
    xarray = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    chains, draws, p = 2, 30, 4
    beta = rng.normal(size=(chains, draws, p)).astype(np.float64)
    post = xarray.Dataset(
        {"beta": (("chain", "draw", "dim0"), beta)},
        coords={"chain": np.arange(chains), "draw": np.arange(draws), "dim0": np.arange(p)},
    )
    idata = type("Id", (), {"posterior": post})()
    out, meta = linear_coef_draws_from_pymc_idata(
        idata,
        pooling=PoolingMode.FULL,
        n_media=3,
        n_controls=1,
    )
    assert meta["export_status"] == "ok"
    assert out is not None
    assert out.shape == (chains * draws, 4)


def test_linear_coef_draws_partial_pooling_rejected():
    xarray = pytest.importorskip("xarray")
    post = xarray.Dataset()
    idata = type("Id", (), {"posterior": post})()
    out, meta = linear_coef_draws_from_pymc_idata(
        idata,
        pooling=PoolingMode.PARTIAL,
        n_media=1,
        n_controls=0,
    )
    assert out is None
    assert meta["export_status"] == "unsupported"
