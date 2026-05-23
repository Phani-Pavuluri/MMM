"""Frozen train→decide E2E fixture builder (deterministic seed, no fabricated extension JSON)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mmm.calibration.replay_etl import SpendShiftSpec, build_replay_units_from_panel_shifts
from mmm.calibration.units_io import write_calibration_units_to_json
from mmm.config.schema import DataConfig
from mmm.data.loader import DatasetBuilder
from mmm.data.schema import PanelSchema
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

E2E_SEED = 4242
E2E_CHANNELS = ("search",)


def write_train_decide_fixture(root: Path) -> dict[str, Path]:
    """
    Write panel CSV, replay units JSON, train config YAML, and BAU scenario YAML under ``root``.

    Returns paths: panel_csv, replay_units, train_config, scenario_yaml, run_dir.
    """
    df, schema0 = generate_geo_panel(
        SyntheticGeoPanelSpec(
            n_geos=4,
            n_weeks=80,
            channels=E2E_CHANNELS,
            betas=(0.75,),
            decay=0.55,
            noise=0.02,
        ),
        seed=E2E_SEED,
    )
    schema = PanelSchema(
        geo_column=schema0.geo_column,
        week_column=schema0.week_column,
        target_column=schema0.target_column,
        channel_columns=schema0.channel_columns,
        control_columns=(),
    )
    cfg_data = DataConfig(
        path=None,
        geo_column=schema.geo_column,
        week_column=schema.week_column,
        target_column=schema.target_column,
        channel_columns=list(schema.channel_columns),
        control_columns=[],
    )
    panel = DatasetBuilder(cfg_data, schema).build(df)

    panel_csv = root / "panel.csv"
    panel.to_csv(panel_csv, index=False)

    weeks_sorted = sorted(panel[schema.week_column].unique())
    specs = [
        SpendShiftSpec(
            unit_id="e2e_u1",
            channel="search",
            spend_multiplier=0.85,
            observed_lift=0.02,
            lift_se=0.08,
            geo_ids=[str(panel[schema.geo_column].iloc[0])],
            week_start=weeks_sorted[8],
            week_end=weeks_sorted[20],
            estimand="geo_time_ATT",
            lift_scale="mean_kpi_level_delta",
        )
    ]
    units = build_replay_units_from_panel_shifts(panel, schema, specs, target_kpi=schema.target_column)
    replay_json = root / "replay_units.json"
    write_calibration_units_to_json(units, replay_json)

    run_dir = root / "mmm_runs"
    train_yaml = root / "train_config.yaml"
    train_yaml.write_text(
        yaml.dump(
            {
                "run_environment": "prod",
                "allow_unsafe_decision_apis": False,
                "framework": "ridge_bo",
                "model_form": "semi_log",
                "random_seed": E2E_SEED,
                "prod_canonical_modeling_contract_id": "ridge_bo_semi_log_calendar_cv_v1",
                "data": {
                    "path": str(panel_csv.resolve()),
                    "geo_column": schema.geo_column,
                    "week_column": schema.week_column,
                    "target_column": schema.target_column,
                    "channel_columns": list(schema.channel_columns),
                    "control_columns": [],
                    "data_version_id": "train-decide-e2e-v1",
                },
                "cv": {"mode": "rolling", "n_splits": 2, "min_train_weeks": 20, "horizon_weeks": 4},
                "ridge_bo": {"n_trials": 12, "sampler_seed": E2E_SEED},
                "transforms": {
                    "adstock": "geometric",
                    "saturation": "hill",
                    "adstock_params": {"decay": 0.55},
                    "saturation_params": {"half_max": 1.0, "slope": 2.0},
                },
                "objective": {
                    "normalization_profile": "strict_prod",
                    "named_profile": "ridge_bo_standard_v1",
                },
                "calibration": {
                    "use_replay_calibration": True,
                    "replay_units_path": str(replay_json.resolve()),
                    "replay_refit_mode": "fold_aligned",
                },
                "budget": {"total_budget": 500.0},
                "artifacts": {"backend": "local", "run_dir": str(run_dir.resolve())},
                "extensions": {
                    "optimization_gates": {"enabled": True},
                    "optimizer_certification": {"enabled": True},
                    "governance": {
                        "max_replay_calibration_chi2": 50.0,
                        "require_falsification_pass": False,
                        "falsification_max_allowed_flags_for_optimization": 8,
                        "require_beats_baselines_for_approval": False,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    decide_yaml = root / "decide_config.yaml"
    decide_yaml.write_text(
        yaml.dump(
            {
                "run_environment": "prod",
                "allow_unsafe_decision_apis": False,
                "framework": "ridge_bo",
                "model_form": "semi_log",
                "random_seed": E2E_SEED,
                "prod_canonical_modeling_contract_id": "ridge_bo_semi_log_calendar_cv_v1",
                "data": {
                    "path": str(panel_csv.resolve()),
                    "geo_column": schema.geo_column,
                    "week_column": schema.week_column,
                    "target_column": schema.target_column,
                    "channel_columns": list(schema.channel_columns),
                    "control_columns": [],
                    "data_version_id": "train-decide-e2e-v1",
                },
                "budget": {"total_budget": 500.0},
                "cv": {"mode": "rolling", "n_splits": 2, "min_train_weeks": 20, "horizon_weeks": 4},
                "objective": {
                    "normalization_profile": "strict_prod",
                    "named_profile": "ridge_bo_standard_v1",
                },
                "extensions": {"optimization_gates": {"enabled": True}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    scenario_yaml = root / "scenario_bau.yaml"
    scenario_yaml.write_text(
        "candidate_spend:\n  search: 15.0\n",
        encoding="utf-8",
    )

    return {
        "panel_csv": panel_csv,
        "replay_units": replay_json,
        "train_config": train_yaml,
        "decide_config": decide_yaml,
        "scenario_yaml": scenario_yaml,
        "run_dir": run_dir,
    }
