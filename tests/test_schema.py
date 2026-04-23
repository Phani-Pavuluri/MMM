import pandas as pd
import pytest

from mmm.data.schema import PanelSchema, PanelValidationError, validate_panel


def test_validate_panel_ok():
    df = pd.DataFrame(
        {
            "geo_id": ["A", "A"],
            "week": [1, 2],
            "revenue": [10.0, 11.0],
            "c1": [1.0, 2.0],
        }
    )
    schema = PanelSchema("geo_id", "week", "revenue", ("c1",))
    validate_panel(df, schema)


def test_validate_panel_calendar_strict_rejects_bad_week_strings() -> None:
    df = pd.DataFrame(
        {
            "geo_id": ["A", "A"],
            "week": ["2024-01-01", "not-a-date"],
            "revenue": [10.0, 11.0],
            "c1": [1.0, 2.0],
        }
    )
    schema = PanelSchema("geo_id", "week", "revenue", ("c1",))
    with pytest.raises(PanelValidationError, match="parseable"):
        validate_panel(df, schema, calendar_strict=True)


def test_validate_negative_spend_raises():
    df = pd.DataFrame(
        {
            "geo_id": ["A", "A"],
            "week": [1, 2],
            "revenue": [10.0, 11.0],
            "c1": [1.0, -1.0],
        }
    )
    schema = PanelSchema("geo_id", "week", "revenue", ("c1",))
    with pytest.raises(PanelValidationError):
        validate_panel(df, schema)
