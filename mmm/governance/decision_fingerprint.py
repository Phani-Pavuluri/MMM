"""Fail-closed train ↔ decide panel fingerprint alignment (prod Ridge paths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.governance.policy import PolicyError

FingerprintKind = Literal["v2_combined", "legacy_panel_only", "missing"]


class DecisionFingerprintMismatchWaiverArtifact(BaseModel):
    waiver_id: str = Field(min_length=4)
    created_at: str
    reason: str = Field(min_length=8)
    owner: str | None = None
    training_fingerprint_sha256: str | None = Field(
        default=None,
        description="Optional expected training sha256_combined; mismatch still requires explicit waiver.",
    )

    @field_validator("created_at")
    @classmethod
    def _parse_created(cls, v: str) -> str:
        from datetime import datetime

        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


def load_decision_fingerprint_mismatch_waiver(path: str | Path) -> DecisionFingerprintMismatchWaiverArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DecisionFingerprintMismatchWaiverArtifact.model_validate(raw)


def training_fingerprint_from_extension_report(er: dict[str, Any]) -> dict[str, Any]:
    fp = er.get("data_fingerprint") or er.get("panel_fingerprint")
    if isinstance(fp, dict):
        return fp
    rfs = er.get("ridge_fit_summary")
    if isinstance(rfs, dict):
        nested = rfs.get("data_fingerprint") or rfs.get("panel_fingerprint")
        if isinstance(nested, dict):
            return nested
    return {}


def _fingerprint_token(fp: dict[str, Any]) -> tuple[str | None, FingerprintKind]:
    if not fp:
        return None, "missing"
    combined = fp.get("sha256_combined")
    if combined:
        return str(combined), "v2_combined"
    legacy = fp.get("sha256_panel_keycols_sorted_csv")
    if legacy:
        return str(legacy), "legacy_panel_only"
    return None, "missing"


def compare_training_and_decision_fingerprints(
    training_fp: dict[str, Any],
    decision_fp: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare training artifact fingerprint to the panel loaded at decide time.

    Prefers ``sha256_combined`` (fingerprint v2). Legacy artifacts compare panel keycols hash only.
    """
    train_tok, train_kind = _fingerprint_token(training_fp)
    decide_tok, decide_kind = _fingerprint_token(decision_fp)
    warnings: list[str] = []
    matched = False
    comparison = "none"

    if train_kind == "missing":
        return {
            "matched": False,
            "comparison": "training_fingerprint_missing",
            "training_kind": train_kind,
            "decision_kind": decide_kind,
            "warnings": ["extension_report has no data_fingerprint; cannot verify train↔decide panel identity"],
        }

    if decide_kind == "missing":
        return {
            "matched": False,
            "comparison": "decision_fingerprint_missing",
            "training_kind": train_kind,
            "decision_kind": decide_kind,
            "warnings": ["decision-time fingerprint could not be computed"],
        }

    if train_kind == "v2_combined" and decide_kind == "v2_combined":
        comparison = "sha256_combined"
        matched = train_tok == decide_tok
    elif train_kind == "legacy_panel_only" or decide_kind == "legacy_panel_only":
        comparison = "sha256_panel_keycols_sorted_csv"
        warnings.append(
            "legacy panel-only fingerprint comparison (training or decision artifact predates fingerprint_v2); "
            "config/transform drift is not fully covered"
        )
        train_legacy = str(training_fp.get("sha256_panel_keycols_sorted_csv") or train_tok or "")
        decide_legacy = str(decision_fp.get("sha256_panel_keycols_sorted_csv") or decide_tok or "")
        matched = bool(train_legacy) and train_legacy == decide_legacy
        if train_kind == "v2_combined" and decide_kind == "legacy_panel_only":
            warnings.append("training artifact has v2 combined hash but decision side fell back to legacy panel hash")
    else:
        matched = train_tok == decide_tok

    return {
        "matched": matched,
        "comparison": comparison,
        "training_kind": train_kind,
        "decision_kind": decide_kind,
        "training_sha": train_tok,
        "decision_sha": decide_tok,
        "warnings": warnings,
    }


def require_decision_fingerprint_match(
    cfg: MMMConfig,
    extension_report: dict[str, Any],
    *,
    panel: Any,
    schema: PanelSchema,
    seed_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prod: abort when decide-time panel fingerprint differs from training extension report.

    Override: ``governance.allow_decision_fingerprint_mismatch`` + signed waiver JSON path.
    """
    if cfg.run_environment != RunEnvironment.PROD:
        return {"skipped": True, "matched": True}

    training_fp = training_fingerprint_from_extension_report(extension_report)
    if cfg.data.path:
        from mmm.data.loader import DatasetBuilder
        from mmm.data.panel_order import sort_panel_for_modeling
        from mmm.data.schema import validate_panel

        builder = DatasetBuilder(cfg.data)
        schema = builder.schema()
        panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    decision_fp = fingerprint_panel(panel, schema, config=cfg, seed_resolution=seed_resolution)
    result = compare_training_and_decision_fingerprints(training_fp, decision_fp)
    result["decision_fingerprint"] = decision_fp
    result["training_fingerprint"] = training_fp

    gov = cfg.governance
    if result.get("matched"):
        return result

    if gov.allow_decision_fingerprint_mismatch:
        wpath = str(gov.decision_fingerprint_mismatch_waiver_path or "").strip()
        if not wpath:
            raise PolicyError(
                "governance.allow_decision_fingerprint_mismatch=True requires "
                "governance.decision_fingerprint_mismatch_waiver_path"
            )
        waiver = load_decision_fingerprint_mismatch_waiver(wpath)
        expected = waiver.training_fingerprint_sha256
        train_sha = result.get("training_sha")
        if expected and train_sha and str(expected) != str(train_sha):
            raise PolicyError(
                "decision fingerprint mismatch waiver training_fingerprint_sha256 does not match "
                "extension_report fingerprint"
            )
        result["waiver"] = {
            "waiver_id": waiver.waiver_id,
            "reason": waiver.reason,
            "owner": waiver.owner,
        }
        result["severe_warning"] = (
            "DECISION FINGERPRINT MISMATCH OVERRIDE: train and decide panels/config fingerprints differ; "
            f"waiver_id={waiver.waiver_id!r}; reason={waiver.reason!r}"
        )
        return result

    comp = result.get("comparison", "unknown")
    raise PolicyError(
        "train↔decide fingerprint mismatch: production simulate/optimize requires the same panel and "
        f"modeling config as training (comparison={comp!r}, training_kind={result.get('training_kind')!r}, "
        f"decision_kind={result.get('decision_kind')!r}). "
        "Set governance.allow_decision_fingerprint_mismatch with a signed waiver only for controlled rollbacks."
    )
