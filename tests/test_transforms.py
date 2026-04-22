import numpy as np

from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.adstock.weibull import WeibullAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation


def test_geometric_adstock_smoke():
    x = np.array([1.0, 0.0, 2.0])
    ad = GeometricAdstock(0.5)
    y = ad.fit(x).transform(x)
    assert y.shape == x.shape


def test_weibull_adstock_smoke():
    x = np.linspace(0, 3, 20)
    ad = WeibullAdstock()
    y = ad.fit(x).transform(x)
    assert len(y) == len(x)


def test_hill_monotone():
    x = np.linspace(0, 5, 50)
    h = HillSaturation(1.0, 2.0)
    y = h.fit(x).transform(x)
    assert np.all(np.diff(y) >= -1e-9)


def test_stack_series():
    spend = np.random.RandomState(0).lognormal(0, 0.2, size=30)
    ad = GeometricAdstock(0.6)
    sat = HillSaturation(1.0, 2.0)
    z = apply_adstock_saturation_series(spend, ad, sat)
    assert z.shape == spend.shape
