"""Control template helper — illustrative CSV scaffolds only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mmm.cli.main import app
from mmm.helpers.control_templates import (
    ControlDomain,
    ControlFrequency,
    build_control_template_dataframe,
    generate_control_template,
    metadata_path_for_csv,
    parse_domain,
    parse_frequency,
    template_metadata,
)


def test_csv_generation(tmp_path: Path):
    out = tmp_path / "controls.csv"
    result = generate_control_template(
        domain=ControlDomain.B2B,
        frequency=ControlFrequency.WEEKLY,
        n_rows=12,
        out=out,
    )
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 12
    assert "week" in df.columns
    assert "sp500_index" in df.columns
    assert result["metadata_path"] is not None


def test_deterministic_output(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    generate_control_template(
        domain="retail", frequency="weekly", n_rows=20, out=a
    )
    generate_control_template(
        domain="retail", frequency="weekly", n_rows=20, out=b
    )
    assert pd.read_csv(a).equals(pd.read_csv(b))


def test_domain_specific_columns():
    df, period, cols = build_control_template_dataframe(
        domain=ControlDomain.TRAVEL,
        frequency=ControlFrequency.WEEKLY,
        n_rows=4,
    )
    assert period == "week"
    assert cols == ["holiday_indicator", "weather_index", "fuel_price", "consumer_sentiment"]
    assert set(cols).issubset(df.columns)


def test_frequency_monthly_uses_month_column():
    df, period, _ = build_control_template_dataframe(
        domain=ControlDomain.GENERIC,
        frequency=ControlFrequency.MONTHLY,
        n_rows=6,
    )
    assert period == "month"
    assert "month" in df.columns
    assert "week" not in df.columns


def test_metadata_generation(tmp_path: Path):
    out = tmp_path / "controls.csv"
    generate_control_template(domain="saas", frequency="monthly", n_rows=3, out=out)
    meta_path = metadata_path_for_csv(out)
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["template_only"] is True
    assert meta["illustrative_values_only"] is True
    assert meta["modeling_default"] is False
    assert meta["replace_with_real_historical_data"] is True
    assert meta["diagnostic_only"] is True
    assert meta["auto_model_integration_forbidden"] is True
    assert meta["domain"] == "saas"
    assert meta["frequency"] == "monthly"


def test_template_metadata_helper():
    meta = template_metadata(
        domain=ControlDomain.ECOMMERCE,
        frequency=ControlFrequency.WEEKLY,
        n_rows=10,
        period_column="week",
        control_columns=["promo_intensity"],
    )
    assert meta["policy_version"] == "control_template_v1"


def test_parse_domain_and_frequency_errors():
    with pytest.raises(ValueError, match="Unknown domain"):
        parse_domain("unknown")
    with pytest.raises(ValueError, match="Unknown frequency"):
        parse_frequency("daily")


def test_cli_smoke(tmp_path: Path):
    out = tmp_path / "controls.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate-control-template",
            "--domain",
            "b2b",
            "--frequency",
            "weekly",
            "--rows",
            "52",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert out.exists()
    assert metadata_path_for_csv(out).exists()
