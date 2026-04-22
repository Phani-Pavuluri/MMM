from mmm.data.schema import validate_panel
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_synthetic_generates_valid_panel():
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec())
    validate_panel(df, schema)
    assert len(df) == 4 * 80
