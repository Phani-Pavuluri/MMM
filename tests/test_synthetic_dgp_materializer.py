"""Phase 4B-1 — rich DGP world materialization for exact-recovery worlds."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.saturation.hill import HillSaturation
from mmm.validation.synthetic.dgp_materializer import (
    DGP_MATERIALIZATION_VERSION,
    compute_dgp_series,
    geometric_adstock_series,
    hill_saturation_series,
    materialize_dgp_world,
)
from mmm.validation.synthetic.validator import validate_bundle, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_008 = REPO_ROOT / "validation" / "worlds" / "WORLD-008-exact-recovery"

RTOL = 1e-9


@pytest.fixture(scope="module")
def world_008_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("dgp") / "WORLD-008-exact-recovery"
    shutil.copytree(WORLD_008, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_world_008_truth_present() -> None:
    assert (WORLD_008 / "world_truth.json").is_file()


def test_materialize_writes_derived_artifacts(world_008_bundle: Path) -> None:
    for name in (
        "panel.parquet",
        "dgp_diagnostics.parquet",
        "dgp_diagnostics_manifest.json",
        "metadata.json",
        "checksums.json",
        "decision_truth.json",
    ):
        assert (world_008_bundle / name).is_file(), name
    manifest = json.loads((world_008_bundle / "dgp_diagnostics_manifest.json").read_text(encoding="utf-8"))
    assert manifest["not_world_truth"] is True
    assert manifest["authoritative"] is False


def test_world_truth_bytes_unchanged_after_materialize(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-008-exact-recovery"
    shutil.copytree(WORLD_008, bundle)
    before = (bundle / "world_truth.json").read_bytes()
    materialize_dgp_world(bundle, overwrite=True)
    after = (bundle / "world_truth.json").read_bytes()
    assert before == after


def test_geometric_adstock_matches_canonical_formula() -> None:
    decay = 0.6
    x = np.array([1.0, 0.0, 2.0, 0.0, 1.5])
    expected = GeometricAdstock(decay).transform(x)
    np.testing.assert_allclose(geometric_adstock_series(x, decay), expected, rtol=RTOL)


def test_hill_saturation_matches_canonical_formula() -> None:
    half, slope = 10.0, 2.0
    x = np.linspace(0.5, 20.0, 10)
    expected = HillSaturation(half_max=half, slope=slope).transform(x)
    np.testing.assert_allclose(hill_saturation_series(x, half_max=half, slope=slope), expected, rtol=RTOL)


def test_linear_predictor_matches_coefficients(world_008_bundle: Path) -> None:
    truth = json.loads((world_008_bundle / "world_truth.json").read_text(encoding="utf-8"))
    diag = pd.read_parquet(world_008_bundle / "dgp_diagnostics.parquet")
    intercept = float(truth["coefficient_truth"]["intercept"])
    betas = truth["coefficient_truth"]["true_beta_by_channel"]
    for _, grp in diag.groupby(["geo_id", "week_start_date"], sort=False):
        expected_eta = intercept + sum(
            float(betas[ch]) * float(grp.loc[grp["channel"] == ch, "saturated_feature"].iloc[0])
            for ch in betas
        )
        np.testing.assert_allclose(grp["linear_predictor"].iloc[0], expected_eta, rtol=RTOL)


def test_kpi_matches_inverse_semi_log(world_008_bundle: Path) -> None:
    diag = pd.read_parquet(world_008_bundle / "dgp_diagnostics.parquet")
    panel = pd.read_parquet(world_008_bundle / "panel.parquet")
    merged = diag.drop_duplicates(["geo_id", "week_start_date"])[
        ["geo_id", "week_start_date", "log_kpi", "generated_kpi"]
    ].merge(panel, on=["geo_id", "week_start_date"])
    np.testing.assert_allclose(merged["generated_kpi"], np.exp(merged["log_kpi"]), rtol=RTOL)
    np.testing.assert_allclose(merged["generated_kpi"], merged["revenue"], rtol=RTOL)


def test_diagnostics_adstock_and_saturation_columns(world_008_bundle: Path) -> None:
    truth = json.loads((world_008_bundle / "world_truth.json").read_text(encoding="utf-8"))
    diag = pd.read_parquet(world_008_bundle / "dgp_diagnostics.parquet")
    transform = truth["transform_truth"]
    for ch in truth["media_truth"]["channels"]:
        decay = float(transform["adstock_decay_by_channel"][ch])
        half = float(transform["hill_half_max_by_channel"][ch])
        slope = float(transform["hill_slope_by_channel"][ch])
        for _, sub in diag[diag["channel"] == ch].groupby("geo_id", sort=True):
            sub = sub.sort_values("week_start_date")
            raw = sub["raw_spend"].to_numpy()
            ad = geometric_adstock_series(raw, decay)
            np.testing.assert_allclose(sub["adstocked_spend"], ad, rtol=RTOL)
            np.testing.assert_allclose(
                sub["saturated_feature"],
                hill_saturation_series(ad, half_max=half, slope=slope),
                rtol=RTOL,
            )


def test_materialization_is_deterministic(world_008_bundle: Path) -> None:
    first = json.loads((world_008_bundle / "checksums.json").read_text(encoding="utf-8"))
    materialize_dgp_world(world_008_bundle, overwrite=True)
    second = json.loads((world_008_bundle / "checksums.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["panel_sha256"] == second["panel_sha256"]
    assert first["dgp_diagnostics_sha256"] == second["dgp_diagnostics_sha256"]


def test_validator_l3_passes(world_008_bundle: Path) -> None:
    result = validate_bundle(world_008_bundle, max_level=3)
    assert result.passed, result.hard_failures


def test_verify_checksums_clean(world_008_bundle: Path) -> None:
    assert verify_checksums(world_008_bundle) == []


def test_compute_dgp_series_panel_shape() -> None:
    truth = json.loads((WORLD_008 / "world_truth.json").read_text(encoding="utf-8"))
    panel, diag = compute_dgp_series(truth)
    n_geos = len(truth["geo_truth"]["geos"])
    n_weeks = int(truth["time_truth"]["n_periods"])
    n_ch = len(truth["media_truth"]["channels"])
    assert len(panel) == n_geos * n_weeks
    assert len(diag) == n_geos * n_weeks * n_ch
    assert bool(diag["derived_artifact"].all())


def test_metadata_records_dgp_version(world_008_bundle: Path) -> None:
    meta = json.loads((world_008_bundle / "metadata.json").read_text(encoding="utf-8"))
    assert meta["materialization_version"] == DGP_MATERIALIZATION_VERSION
    assert meta["dgp_materialization"] is True
