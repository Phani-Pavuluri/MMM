"""High-level trainers aligned with YAML config."""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm.artifacts.stores.local import LocalArtifactStore
from mmm.config.load import dump_resolved_config, load_config, resolve_config
from mmm.config.schema import Framework, MMMConfig
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.loader import DatasetBuilder
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.panel_qa import run_panel_qa
from mmm.decomposition.engine import DecompositionEngine
from mmm.features.design_matrix import build_design_matrix
from mmm.economics.canonical import assert_planner_scope_supported, build_economics_contract
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.governance.decision_safety import MSG_ANALYSIS_ONLY, decision_safety_artifact
from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


class MMMTrainer:
    """Routes to Ridge+BO or Bayesian trainer."""

    def __init__(self, config: MMMConfig) -> None:
        self.config = resolve_config(config)
        self.schema = DatasetBuilder(self.config.data).schema()

    @classmethod
    def from_yaml(cls, path: str | Path) -> MMMTrainer:
        return cls(load_config(path))

    def run(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        run_id = self.config.run_id or str(uuid.uuid4())
        store = LocalArtifactStore(self.config.artifacts.run_dir)
        store.start_run(run_id, metadata={"framework": self.config.framework.value})
        dump_resolved_config(self.config, store.run_path / "resolved_config.yaml")
        cfg_blob = json.dumps(self.config.model_dump_resolved(), sort_keys=True, default=str)
        store.log_dict(
            "config_fingerprint",
            {
                "sha256": hashlib.sha256(cfg_blob.encode()).hexdigest(),
                "version": "mmm_config_v1",
                "package_version": __import__("mmm.version", fromlist=["__version__"]).__version__,
            },
        )
        run_warnings: list[str] = []
        assert_planner_scope_supported(build_economics_contract(self.config))
        if not self.config.allow_unsafe_decision_apis:
            warnings.warn(MSG_ANALYSIS_ONLY, UserWarning, stacklevel=2)
            run_warnings.append(MSG_ANALYSIS_ONLY)
        store.log_dict(
            "decision_safety",
            decision_safety_artifact(allow_unsafe_decision_apis=self.config.allow_unsafe_decision_apis),
        )

        builder = DatasetBuilder(self.config.data, self.schema)
        panel = builder.build(df)
        panel_work = sort_panel_for_modeling(panel, self.schema)
        store.log_dict("data_fingerprint", fingerprint_panel(panel_work, self.schema))
        store.log_dict(
            "data_validation_preview",
            {
                "panel_qa": run_panel_qa(panel_work, self.schema, self.config.extensions.panel_qa),
                "note": "Structural QA before fit; extension_report.panel_qa is canonical post-sort.",
            },
        )

        if self.config.framework == Framework.RIDGE_BO:
            trainer = RidgeBOMMMTrainer(self.config, self.schema)
        else:
            trainer = BayesianMMMTrainer(self.config, self.schema)
        fit_out = trainer.fit(panel_work)
        yhat = trainer.predict(panel_work)
        resid = panel_work[self.config.data.target_column].to_numpy(dtype=float) - yhat
        store.log_metrics({"mae": float(np.mean(np.abs(resid)))})
        if self.config.framework == Framework.RIDGE_BO and isinstance(fit_out, dict):
            if fit_out.get("ridge_bo_telemetry"):
                store.log_dict("ridge_bo_telemetry", fit_out["ridge_bo_telemetry"])
        elif isinstance(fit_out, dict) and fit_out.get("bayesian_prior_policy"):
            store.log_dict("bayesian_prior_policy", fit_out["bayesian_prior_policy"])

        if self.config.framework == Framework.RIDGE_BO:
            art = fit_out["artifacts"]
            dec = DecompositionEngine(self.schema, self.config.model_form)
            decomp = dec.ridge_decompose(
                panel_work,
                art.coef,
                float(art.intercept[0]),
                self.config,
                decay=art.best_params["decay"],
                hill_half=art.best_params["hill_half"],
                hill_slope=art.best_params["hill_slope"],
            )
            store.log_dict("decomposition_head", decomp.channel_contributions.head().to_dict())
            store.log_dict(
                "decomposition_semantics",
                {
                    "scale": decomp.scale,
                    "is_exact_additive": decomp.is_exact_additive,
                    "safe_for_budgeting": decomp.safe_for_budgeting,
                    "notes": decomp.notes,
                    "economics_output_metadata": decomp.economics_output_metadata,
                },
            )
            store.log_dict("leaderboard", {"top": art.leaderboard[:5]})
            lineage_bundle = build_design_matrix(
                panel_work,
                self.schema,
                self.config,
                decay=art.best_params["decay"],
                hill_half=art.best_params["hill_half"],
                hill_slope=art.best_params["hill_slope"],
            )
            store.log_dict("feature_lineage", lineage_bundle.to_lineage_json())
        ext_report = run_post_fit_extensions(
            panel=panel_work,
            schema=self.schema,
            config=self.config,
            fit_out=fit_out,
            yhat=yhat,
            store=store,
        )
        store.end_run()
        return {
            "run_id": run_id,
            "fit": fit_out,
            "predictions": yhat,
            "store": str(store.run_path),
            "extensions": ext_report,
            "allow_unsafe_decision_apis": self.config.allow_unsafe_decision_apis,
            "warnings": run_warnings,
        }


class MMMComparator:
    """Compare multiple resolved configs on the same panel."""

    def __init__(self, configs: list[MMMConfig]) -> None:
        self.configs = configs

    def run_all(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for cfg in self.configs:
            rows.append(MMMTrainer(cfg).run(df))
        return rows
