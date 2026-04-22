from mmm.utils.synthetic import SyntheticGeoPanelSpec, known_dgp_parameters


def test_known_dgp():
    s = SyntheticGeoPanelSpec(decay=0.7)
    p = known_dgp_parameters(s)
    assert p["decay"] == 0.7
